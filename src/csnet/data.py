"""Data loaders.

Two paths:

1. **Synthetic** — `make_synthetic_dataset` generates 7-class Gaussian-mixture
   1D spectra. Used by the smoke test so that the pipeline can be validated
   end-to-end without any download. Each class is given a different
   continuum slope + a different set of emission/absorption "lines" so that
   the classification task is non-trivial but learnable in ~5 epochs.

2. **Real** — `load_survey` reads preprocessed parquet/HDF5 produced by
   `scripts/preprocess.py` from SDSS DR19 BOSS, LAMOST DR8, and SDSS APOGEE
   DR17. The preprocess step resamples every spectrum to a common 4096-px
   grid in [3800, 8000] Å (vacuum wavelengths, Morton 1991 air→vac for CFLIB).

The dataset interface is plain torch.utils.data.Dataset returning
(spectrum [seq_len], label int).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from .utils import NUM_CLASSES


# --------------------------------------------------------------------------- #
# Synthetic generator (used by smoke test)                                    #
# --------------------------------------------------------------------------- #

def _class_template(cls_idx: int, seq_len: int, rng: np.random.Generator) -> np.ndarray:
    """Build a base spectrum for class `cls_idx`. Each class differs in:
      - continuum slope (mimics Planck T_eff dependence)
      - a unique set of "line" positions (emission for early types,
        absorption for late types)
    """
    x = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)

    # Continuum: linear slope. Class 0 (O) is hot/blue-rising, class 6 (M) is cool/red-falling.
    slope = 1.5 - 0.5 * cls_idx  # +1.5 (O) ... -1.5 (M)
    continuum = 1.0 + slope * (x - 0.5)

    # Lines: 3 Gaussians per class at fixed positions (deterministic per class)
    # so that the network has a recognisable signature.
    line_centers = [0.1 + 0.12 * cls_idx, 0.4 + 0.05 * cls_idx, 0.7 - 0.04 * cls_idx]
    line_signs = [-1.0, +1.0, -1.0] if cls_idx < 3 else [-1.0, -1.0, -1.0]  # absorption for K/M
    line_amp = 0.6
    line_width = 0.01

    spec = continuum.copy()
    for c, sgn in zip(line_centers, line_signs):
        spec = spec + sgn * line_amp * np.exp(-((x - c) ** 2) / (2 * line_width ** 2))

    return spec.astype(np.float32)


def make_synthetic_dataset(
    n_per_class: int = 200,
    seq_len: int = 512,
    noise_sigma: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X [N, seq_len], y [N]) with N = n_per_class * NUM_CLASSES."""
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for cls in range(NUM_CLASSES):
        base = _class_template(cls, seq_len, rng)
        for _ in range(n_per_class):
            spec = base + rng.normal(0.0, noise_sigma, size=seq_len).astype(np.float32)
            X_list.append(spec)
            y_list.append(cls)
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    # Shuffle once with the rng
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# --------------------------------------------------------------------------- #
# Torch Dataset wrapper                                                        #
# --------------------------------------------------------------------------- #

class SpectrumDataset(Dataset):
    """Wraps a (X, y) numpy pair as a torch dataset. Adds a channel dim so
    that the spectrum is shaped [1, seq_len] (Conv1D expects channels-first)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2, f"X must be [N, seq_len], got {X.shape}"
        assert len(X) == len(y), f"len mismatch: {len(X)} vs {len(y)}"
        self.X = torch.from_numpy(X).float().unsqueeze(1)  # [N, 1, seq_len]
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# --------------------------------------------------------------------------- #
# Splits & DataLoaders                                                         #
# --------------------------------------------------------------------------- #

@dataclass
class LoaderBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    seq_len: int
    class_counts: np.ndarray   # [NUM_CLASSES] training-split counts (for loss weighting)


def build_synthetic_loaders(
    n_per_class: int = 200,
    seq_len: int = 512,
    noise_sigma: float = 0.05,
    batch_size: int = 64,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
    num_workers: int = 0,
) -> LoaderBundle:
    X, y = make_synthetic_dataset(n_per_class, seq_len, noise_sigma, seed=seed)
    ds = SpectrumDataset(X, y)

    n_total = len(ds)
    n_test = int(test_frac * n_total)
    n_val = int(val_frac * n_total)
    n_train = n_total - n_test - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=gen)

    # Class counts in TRAIN ONLY (for ENS-Focal weighting)
    y_train = np.array([int(ds.y[i]) for i in train_ds.indices])
    counts = np.bincount(y_train, minlength=NUM_CLASSES)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)

    return LoaderBundle(train_loader, val_loader, test_loader, seq_len, counts)


# --------------------------------------------------------------------------- #
# Real-data loaders (stubs that explain how to populate)                       #
# --------------------------------------------------------------------------- #

SurveyName = Literal["sdss", "lamost", "apogee", "cflib"]


def load_survey(
    name: SurveyName,
    root: Path | str = "data/processed",
    split: Literal["train", "val", "test"] = "train",
) -> SpectrumDataset:
    """Load a preprocessed survey. Expected layout after `scripts/preprocess.py`:

        data/processed/
            sdss/{train,val,test}.npz   # arrays 'X' [N, 4096], 'y' [N]
            lamost/{train,val,test}.npz
            apogee/{train,val,test}.npz
            cflib/test.npz              # external benchmark only

    Raises FileNotFoundError with a helpful message if the file is missing.
    """
    root = Path(root)
    path = root / name / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed survey file not found: {path}\n"
            f"Run `python scripts/download_{name}.py` then "
            f"`python scripts/preprocess.py --raw data/raw --out {root}` first.\n"
            f"(For smoke testing without downloads, use --config configs/smoke.yaml instead.)"
        )
    with np.load(path) as npz:
        X = npz["X"]
        y = npz["y"]
    return SpectrumDataset(X, y)
