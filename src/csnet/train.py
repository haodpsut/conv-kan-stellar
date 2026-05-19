"""Single training run + CLI entrypoint.

Usage
-----
    python -m csnet.train --config configs/smoke.yaml --model conv_kan --seed 42

    # Override config values from CLI:
    python -m csnet.train --config configs/sdss_only.yaml \
                          --model conv_kan --seed 7 \
                          --epochs 30 --batch-size 128 --lr 1e-3

What it does
------------
1. Loads YAML config, merges CLI overrides.
2. Seeds all RNGs deterministically (Python, NumPy, torch, CUDA).
3. Builds data loaders (synthetic for smoke, real surveys for production).
4. Builds the requested model.
5. Trains with ENS-Focal loss, AdamW, cosine LR schedule, optional mixed precision.
6. Evaluates on test split + (if available) CFLIB external benchmark.
7. Writes results/<run_name>/metrics.json (per-epoch curves + final test metrics).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from .data import LoaderBundle, SpectrumDataset, build_synthetic_loaders, load_survey
from .losses import ENSFocalLoss, ens_class_weights
from .models import MODELS, build_model, count_params
from .utils import (
    NUM_CLASSES,
    CLASSES,
    RunMetrics,
    StopWatch,
    accuracy,
    device_auto,
    macro_f1,
    set_seed,
)


# --------------------------------------------------------------------------- #
# Config loading                                                               #
# --------------------------------------------------------------------------- #

def load_config(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """CLI flags override anything in the YAML."""
    for key in ("model", "seed", "epochs", "batch_size", "lr", "weight_decay", "amp"):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    return cfg


# --------------------------------------------------------------------------- #
# Loader factory                                                               #
# --------------------------------------------------------------------------- #

def build_loaders_from_config(cfg: dict[str, Any]) -> LoaderBundle:
    """Decide between synthetic and real-survey loaders based on cfg['data']['source']."""
    data_cfg = cfg.get("data", {})
    source = data_cfg.get("source", "synthetic")
    bs = cfg.get("batch_size", 64)
    seed = cfg.get("seed", 0)

    if source == "synthetic":
        return build_synthetic_loaders(
            n_per_class=data_cfg.get("n_per_class", 200),
            seq_len=data_cfg.get("seq_len", 512),
            noise_sigma=data_cfg.get("noise_sigma", 0.05),
            batch_size=bs,
            val_frac=data_cfg.get("val_frac", 0.15),
            test_frac=data_cfg.get("test_frac", 0.15),
            seed=seed,
            num_workers=data_cfg.get("num_workers", 0),
        )

    if source == "real":
        surveys: list[str] = data_cfg.get("surveys", ["sdss"])
        root = data_cfg.get("root", "data/processed")

        train_parts, val_parts, test_parts = [], [], []
        for s in surveys:
            train_parts.append(load_survey(s, root=root, split="train"))
            val_parts.append(load_survey(s, root=root, split="val"))
            test_parts.append(load_survey(s, root=root, split="test"))

        train_ds = _concat(train_parts)
        val_ds = _concat(val_parts)
        test_ds = _concat(test_parts)

        # Class counts from concatenated training data
        counts = np.bincount(train_ds.y.numpy(), minlength=NUM_CLASSES)
        nw = data_cfg.get("num_workers", 4)
        pin = torch.cuda.is_available()

        return LoaderBundle(
            train=DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin, drop_last=True),
            val=DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin),
            test=DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin),
            seq_len=train_ds.X.shape[-1],
            class_counts=counts,
        )

    raise ValueError(f"Unknown data.source = {source!r} (expected 'synthetic' or 'real')")


def _concat(parts: list[SpectrumDataset]) -> SpectrumDataset:
    """Concatenate multiple SpectrumDatasets in-place into one."""
    X = torch.cat([p.X for p in parts], dim=0).squeeze(1).numpy()
    y = torch.cat([p.y for p in parts], dim=0).numpy()
    return SpectrumDataset(X, y)


# --------------------------------------------------------------------------- #
# Train + eval                                                                  #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             criterion: nn.Module | None = None) -> dict[str, Any]:
    model.eval()
    all_logits, all_y = [], []
    losses: list[float] = []
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        if criterion is not None:
            losses.append(criterion(logits, y).item())
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())
    if not all_logits:
        return {"loss": float("nan"), "acc": 0.0, "macro_f1": 0.0, "per_class_f1": [0.0] * NUM_CLASSES}
    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)
    macro, per_class = macro_f1(logits, y)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "acc": accuracy(logits, y),
        "macro_f1": macro,
        "per_class_f1": per_class,
    }


def train_one_run(cfg: dict[str, Any], out_dir: Path) -> RunMetrics:
    seed = cfg.get("seed", 0)
    set_seed(seed, deterministic=cfg.get("deterministic", True))

    device = device_auto()
    print(f"[csnet] device={device}  seed={seed}  model={cfg['model']}")

    # Data
    loaders = build_loaders_from_config(cfg)
    print(f"[csnet] seq_len={loaders.seq_len}  train={len(loaders.train.dataset)}  "
          f"val={len(loaders.val.dataset)}  test={len(loaders.test.dataset)}")

    # Model
    model = build_model(cfg["model"], num_classes=NUM_CLASSES).to(device)
    n_params = count_params(model)
    print(f"[csnet] params={n_params:,}")

    # Loss / optim
    class_weights = ens_class_weights(loaders.class_counts, beta=cfg.get("ens_beta", 0.9999)).to(device)
    criterion = ENSFocalLoss(class_weights, gamma=cfg.get("focal_gamma", 2.0),
                             label_smoothing=cfg.get("label_smoothing", 0.0))
    optimiser = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.get("lr", 1e-3),
                                  weight_decay=cfg.get("weight_decay", 1e-4))
    epochs = cfg.get("epochs", 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    use_amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    metrics = RunMetrics(
        run_name=cfg.get("run_name", f"{cfg['model']}_seed{seed}"),
        model=cfg["model"],
        config=str(cfg.get("__config_path__", "<inline>")),
        seed=seed,
        epochs_completed=0,
    )

    with StopWatch() as sw:
        for epoch in range(1, epochs + 1):
            model.train()
            ep_losses: list[float] = []
            for X, y in loaders.train:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimiser.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(X)
                    loss = criterion(logits, y)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    loss.backward()
                    optimiser.step()
                ep_losses.append(loss.item())
            scheduler.step()
            train_loss = float(np.mean(ep_losses))
            val_stats = evaluate(model, loaders.val, device, criterion)
            metrics.train_loss.append(train_loss)
            metrics.val_loss.append(val_stats["loss"])
            metrics.val_acc.append(val_stats["acc"])
            metrics.epochs_completed = epoch
            print(f"[csnet] epoch {epoch:3d}/{epochs} "
                  f"train_loss={train_loss:.4f} "
                  f"val_loss={val_stats['loss']:.4f} "
                  f"val_acc={val_stats['acc']:.4f} "
                  f"val_macro_f1={val_stats['macro_f1']:.4f}")

    # Final test
    test_stats = evaluate(model, loaders.test, device, criterion)
    metrics.test_acc = test_stats["acc"]
    metrics.test_macro_f1 = test_stats["macro_f1"]
    metrics.per_class_f1 = dict(zip(CLASSES, test_stats["per_class_f1"]))
    metrics.wall_seconds = sw.elapsed

    # Optional CFLIB cross-survey benchmark (only if file exists)
    try:
        cflib_ds = load_survey("cflib", root=cfg.get("data", {}).get("root", "data/processed"),
                               split="test")
        cflib_loader = DataLoader(cflib_ds, batch_size=cfg.get("batch_size", 64),
                                  shuffle=False, num_workers=0)
        cflib_stats = evaluate(model, cflib_loader, device, criterion)
        metrics.cflib_acc = cflib_stats["acc"]
        metrics.cflib_macro_f1 = cflib_stats["macro_f1"]
        print(f"[csnet] CFLIB cross-survey acc={cflib_stats['acc']:.4f} "
              f"macro_f1={cflib_stats['macro_f1']:.4f}")
    except FileNotFoundError:
        pass  # smoke test path

    # Save metrics
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.dump(out_dir / "metrics.json")
    print(f"[csnet] wrote {out_dir/'metrics.json'} "
          f"(test_acc={metrics.test_acc:.4f}, wall={metrics.wall_seconds:.1f}s)")
    return metrics


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="csnet.train", description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="YAML config path")
    p.add_argument("--model", type=str, default=None, choices=sorted(MODELS))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                   help="Enable mixed-precision training on CUDA")
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir (default: results/<config_stem>/<model>/seed_<N>/)")
    p.add_argument("--run-name", type=str, default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    cfg["__config_path__"] = str(args.config)
    cfg = apply_overrides(cfg, args)

    if "model" not in cfg:
        p.error("--model is required (no default in config)")
    if "seed" not in cfg:
        cfg["seed"] = 0

    if args.run_name:
        cfg["run_name"] = args.run_name

    out_dir = args.out or Path("results") / args.config.stem / cfg["model"] / f"seed_{cfg['seed']}"
    train_one_run(cfg, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
