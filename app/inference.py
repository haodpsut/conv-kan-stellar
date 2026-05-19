"""Model loading + inference helpers used by the Streamlit demo.

Designed to work in three modes:

1. **Cold start (no trained weights yet)** — instantiates models with random
   initialisation so the UI is fully runnable for development. A clear
   "RANDOM WEIGHTS — DEMO MODE" banner is surfaced via `is_random_weights()`.

2. **Smoke-test weights** — checkpoints saved by the smoke test runs.

3. **Production H200 weights** — downloaded from a GitHub Release attachment
   to `weights/` and matched by filename pattern `<model_key>_<config>.pt`.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from csnet.models import MODELS, build_model
from csnet.utils import NUM_CLASSES, CLASSES, device_auto


WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


class ModelBundle:
    """Lightweight cache of (model_name -> nn.Module) so the demo loads
    each architecture once."""

    def __init__(self, weights_dir: Path | None = None,
                 model_names: list[str] | None = None,
                 device: torch.device | None = None):
        self.weights_dir = weights_dir or WEIGHTS_DIR
        self.device = device or device_auto()
        self.model_names = model_names or [n for n in MODELS if n != "mamba1d"]
        self.models: dict[str, torch.nn.Module] = {}
        self.weight_sources: dict[str, str] = {}   # model -> "random" / "<path>"
        self._build_all()

    def _build_all(self) -> None:
        for name in self.model_names:
            try:
                model = build_model(name, num_classes=NUM_CLASSES).to(self.device)
            except ImportError:
                # e.g. mamba1d without mamba-ssm installed; skip gracefully.
                continue
            ckpt = self.weights_dir / f"{name}.pt"
            if ckpt.exists():
                try:
                    state = torch.load(ckpt, map_location=self.device)
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    model.load_state_dict(state, strict=True)
                    self.weight_sources[name] = str(ckpt)
                except Exception as e:
                    print(f"[inference] failed loading {ckpt}: {e}; using random init")
                    self.weight_sources[name] = "random"
            else:
                self.weight_sources[name] = "random"
            model.eval()
            self.models[name] = model

    def is_random_weights(self) -> bool:
        return all(v == "random" for v in self.weight_sources.values())

    @torch.no_grad()
    def predict_all(self, spectrum: np.ndarray) -> dict[str, dict[str, Any]]:
        """Return per-model {probs[7], pred_class, pred_class_name, latency_ms}."""
        x = torch.from_numpy(spectrum).float().to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)    # [1, 1, L]
        out: dict[str, dict[str, Any]] = {}
        for name, model in self.models.items():
            t0 = time.perf_counter()
            logits = model(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000.0
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            pred = int(probs.argmax())
            out[name] = {
                "probs": probs,
                "pred_class": pred,
                "pred_class_name": CLASSES[pred],
                "latency_ms": float(elapsed),
            }
        return out
