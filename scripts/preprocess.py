"""Resample raw spectra from each survey onto the common 4096-px grid in
[3800, 8000] Å (vacuum wavelengths).

This is a stub with the algorithm spelled out — the file walking and
parallelism need tuning to each survey's actual on-disk layout (filename
patterns, HDU index, wavelength conventions, units).

The common-grid + air→vacuum logic itself is implemented and ready to use
as a helper from elsewhere.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np


# Common grid (matches our IEEE TAI submission)
COMMON_LO_ANG = 3800.0
COMMON_HI_ANG = 8000.0
COMMON_NPIX = 4096
COMMON_GRID_VAC = np.linspace(COMMON_LO_ANG, COMMON_HI_ANG, COMMON_NPIX, dtype=np.float32)


def air_to_vacuum_morton(wl_air_angstrom: np.ndarray) -> np.ndarray:
    """Morton (1991) air -> vacuum conversion (CFLIB wavelengths are in air)."""
    s = 1e4 / wl_air_angstrom
    n = (1.0 + 0.0000834254
              + 0.02406147 / (130.0 - s ** 2)
              + 0.00015998 / (38.9 - s ** 2))
    return wl_air_angstrom * n


def resample_to_common_grid(wl: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """Linear interpolate (wl, flux) onto COMMON_GRID_VAC. Values outside the
    input wl range are zero-padded.
    """
    out = np.zeros(COMMON_NPIX, dtype=np.float32)
    mask = (COMMON_GRID_VAC >= wl[0]) & (COMMON_GRID_VAC <= wl[-1])
    out[mask] = np.interp(COMMON_GRID_VAC[mask], wl, flux).astype(np.float32)
    return out


def normalise_continuum(flux: np.ndarray) -> np.ndarray:
    """Simple median-of-positives normalisation. Real pipelines should use
    iterative continuum fitting (e.g., polyfit + sigma-clip) — placeholder."""
    med = float(np.median(flux[flux > 0])) if (flux > 0).any() else 1.0
    return flux / max(med, 1e-6)


SurveyName = Literal["sdss", "lamost", "apogee", "cflib"]


def preprocess_stub(survey: SurveyName, raw_dir: Path, out_dir: Path) -> None:
    """Per-survey preprocessing entry. Currently raises NotImplementedError
    with the steps that need filling in for each survey."""
    raise NotImplementedError(
        f"Preprocess for survey '{survey}' is a stub. To implement:\n"
        f"  1. Walk {raw_dir}/ for the survey's spectrum file pattern.\n"
        f"  2. For each file: open with astropy.io.fits, extract wavelength + flux\n"
        f"     from the documented HDU (SDSS: HDU 1 'COADD' loglam+flux; "
        f"     LAMOST: HDU 0 BinTableHDU; APOGEE: HDU 1 apStar).\n"
        f"  3. If wavelengths are in air, apply Morton (1991) air->vacuum.\n"
        f"  4. resample_to_common_grid(...) then normalise_continuum(...).\n"
        f"  5. Look up the Harvard label from the catalog (use TEFF for APOGEE,\n"
        f"     'subclass' column for LAMOST, 'CLASS' for SDSS).\n"
        f"  6. Append to per-split arrays.\n"
        f"  7. Random-split 80/10/10 (stratified by class) -> "
        f"{out_dir}/{survey}/{{train,val,test}}.npz with arrays 'X' and 'y'.\n"
        f"\n"
        f"For multi-process speed-up, use multiprocessing.Pool over the file list."
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True, help="Raw downloads root")
    p.add_argument("--out", type=Path, required=True, help="Preprocessed output root")
    p.add_argument("--survey", type=str, choices=["sdss", "lamost", "apogee", "cflib", "all"],
                   default="all")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    surveys = ["sdss", "lamost", "apogee", "cflib"] if args.survey == "all" else [args.survey]
    for s in surveys:
        print(f"\n[preprocess] {s}: NOT YET IMPLEMENTED (stub).")
        try:
            preprocess_stub(s, args.raw / s, args.out)
        except NotImplementedError as e:
            print(str(e))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
