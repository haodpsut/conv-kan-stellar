"""Download SDSS APOGEE DR17 high-resolution spectra.

STUB: instructions only. The APOGEE DR17 corpus is ~700K spectra at R~22500
(~50 GB raw, ~12 GB after preprocessing to 4096 px).
"""
from __future__ import annotations

import argparse
from pathlib import Path


INSTRUCTIONS = """
================================================================
SDSS APOGEE DR17 download — manual steps
================================================================

1. Public access — no account needed.

2. Catalog (allStar):
       wget -c -P {outdir} \\
         https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec/allStar-dr17-synspec.fits

3. Spectra (apStar, ~50 GB):
       rsync -avP --partial \\
         rsync://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/ \\
         {outdir}/stars/

   Note: the apStar files contain wavelength-calibrated, sky-subtracted,
   barycentric-corrected, telluric-corrected combined spectra in HDU[1].

4. For classification labels, use the allStar TEFF + LOGG to assign a
   Harvard spectral type via Pecaut & Mamajek (2013) temperature bins:
       O: T_eff >= 30000
       B: 10000-30000
       A: 7500-10000
       F: 6000-7500
       G: 5200-6000
       K: 3700-5200
       M: T_eff < 3700

Expected layout:
    {outdir}/
        allStar-dr17-synspec.fits           (catalog with labels)
        stars/<telescope>/<field>/apStar-*.fits

Preprocess resamples R=22500 spectra to the common 4096-px grid:
    python scripts/preprocess.py --raw {outdir} --out data/processed --survey apogee
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/raw/apogee"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(INSTRUCTIONS.format(outdir=args.out))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
