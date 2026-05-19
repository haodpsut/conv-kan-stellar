"""Download LAMOST DR8 low-resolution spectra.

STUB: instructions only. The full DR8 LRS catalog has ~10M spectra (~500 GB).
"""
from __future__ import annotations

import argparse
from pathlib import Path


INSTRUCTIONS = """
================================================================
LAMOST DR8 low-res download — manual steps
================================================================

1. Create LAMOST data account: https://www.lamost.org/lmusers/

2. Use LAMOST Bulk Download interface:
   https://dr8.lamost.org/v2.0/data/

3. Filter Catalog:
   Catalog: LRS Stellar Parameter
   Select columns: obsid, ra, dec, snrg, subclass, teff, logg, feh

4. Download catalog as FITS  -> {outdir}/dr8_v2_stellar_catalog.fits.gz

5. Download spectra in batches via the per-obsid URL pattern:
       https://dr8.lamost.org/v2.0/spectrum/fits/<obsid>?token=<your_token>

   Use the provided shell script (parallel-safe, resumes on failure):
       cat obsid_list.txt | xargs -P 8 -I OBS \\
           wget -c -nc -P {outdir}/spectra/ \\
           "https://dr8.lamost.org/v2.0/spectrum/fits/OBS?token=$LAMOST_TOKEN"

Expected layout:
    {outdir}/
        dr8_v2_stellar_catalog.fits.gz
        spectra/
            <obsid>.fits                              (~10M files, ~500 GB raw)

After preprocessing (resampled to 4096 px on [3800, 8000] Å) the on-disk size
drops to ~160 GB.

Next step:
    python scripts/preprocess.py --raw {outdir} --out data/processed --survey lamost
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/raw/lamost"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(INSTRUCTIONS.format(outdir=args.out))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
