"""Pre-loaded sample spectra for the demo. Currently uses synthetic generation
so the demo is fully functional without external files. When real survey FITS
files become available, drop them in `app/samples_fits/` and extend
`load_real_samples()`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from csnet.data import _class_template
from csnet.utils import CLASSES, NUM_CLASSES


SAMPLES_DIR = Path(__file__).resolve().parent / "samples_fits"


def synthetic_samples(seq_len: int = 4096, noise_sigma: float = 0.05,
                      seed: int = 42) -> list[dict]:
    """One synthetic spectrum per Harvard class, returned as a list of dicts
    with keys: name, survey, true_class, true_class_name, spectrum, wavelength."""
    rng = np.random.default_rng(seed)
    out = []
    wl = np.linspace(3800.0, 8000.0, seq_len)
    survey_cycle = ["SDSS DR19", "LAMOST DR8", "APOGEE DR17", "CFLIB"]
    for cls in range(NUM_CLASSES):
        base = _class_template(cls, seq_len, rng)
        spec = base + rng.normal(0.0, noise_sigma, size=seq_len).astype(np.float32)
        out.append({
            "name": f"Synthetic {CLASSES[cls]}-type",
            "survey": f"{survey_cycle[cls % 4]} (synthetic)",
            "true_class": cls,
            "true_class_name": CLASSES[cls],
            "spectrum": spec.astype(np.float32),
            "wavelength": wl,
        })
    return out


def load_real_samples() -> list[dict]:
    """Load real FITS samples if present in samples_fits/. Returns [] if none."""
    if not SAMPLES_DIR.exists():
        return []
    try:
        from astropy.io import fits
    except ImportError:
        return []

    out = []
    for fp in sorted(SAMPLES_DIR.glob("*.fits")):
        # Real loaders will need per-survey decoding logic; placeholder for now.
        try:
            with fits.open(fp) as hdul:
                spec = hdul[1].data["flux"].astype(np.float32)
                wl = hdul[1].data["wavelength"].astype(np.float32)
            out.append({
                "name": fp.stem,
                "survey": fp.stem.split("_")[0].upper() if "_" in fp.stem else "real",
                "true_class": None,
                "true_class_name": "?",
                "spectrum": spec,
                "wavelength": wl,
            })
        except Exception as e:
            print(f"[samples] skip {fp}: {e}")
    return out


def all_samples(seq_len: int = 4096) -> list[dict]:
    real = load_real_samples()
    if real:
        return real
    return synthetic_samples(seq_len=seq_len)
