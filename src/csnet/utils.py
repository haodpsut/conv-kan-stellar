"""Seeding, logging, metrics."""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


# Harvard spectral classes in conventional temperature order
CLASSES = ["O", "B", "A", "F", "G", "K", "M"]
NUM_CLASSES = len(CLASSES)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG we touch. deterministic=True forces reproducible CUDA ops
    (slower but required for multi-seed statistical comparison)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class RunMetrics:
    """Single training-run result, dumped as metrics.json."""

    run_name: str
    model: str
    config: str
    seed: int
    epochs_completed: int
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    test_acc: float | None = None
    test_macro_f1: float | None = None
    cflib_acc: float | None = None
    cflib_macro_f1: float | None = None
    per_class_f1: dict[str, float] | None = None
    wall_seconds: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def dump(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == target).float().mean().item()


def macro_f1(logits: torch.Tensor, target: torch.Tensor, num_classes: int = NUM_CLASSES) -> tuple[float, list[float]]:
    """Unweighted mean of per-class F1. Returns (macro_F1, per_class_F1_list)."""
    pred = logits.argmax(dim=-1)
    f1s: list[float] = []
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().item()
        fp = ((pred == c) & (target != c)).sum().item()
        fn = ((pred != c) & (target == c)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)), f1s


class StopWatch:
    """Tiny context-manager timer."""

    def __init__(self) -> None:
        self.t0: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "StopWatch":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed = time.perf_counter() - self.t0
