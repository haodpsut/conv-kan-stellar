"""Download SDSS DR19 BOSS spectra catalog + spectrum files.

STUB: writes step-by-step instructions to stdout. Real download is
~200 GB and requires either:
  (a) astroquery + SDSS DR19 SAS access (slow, single-thread)
  (b) Globus transfer (recommended; needs SAS account)
  (c) Pre-staged copy from a local mirror

This script intentionally does NOT download automatically — it tells the
user exactly what to run so they can monitor progress and re-resume on
network failure.

After download, run `scripts/preprocess.py` to resample to the common
4096-px grid and split into train/val/test.
"""
from __future__ import annotations

import argparse
from pathlib import Path


INSTRUCTIONS = """
================================================================
SDSS DR19 BOSS download — manual steps (no auto-download)
================================================================

OPTION A: astroquery (small subsets, debugging)
------------------------------------------------
    conda activate csnet
    pip install astroquery
    python -c "
    from astroquery.sdss import SDSS
    res = SDSS.query_sql('SELECT TOP 100 specObjID, plate, mjd, fiberID FROM SpecObj')
    print(res)
    "

OPTION B: Globus (recommended for the full ~200 GB)
---------------------------------------------------
1. Create an SDSS SAS account: https://www.sdss.org/dr19/data_access/
2. Install Globus CLI:  conda install -c conda-forge globus-cli
3. Set up endpoint on the 4090 server (one-time):
       globus endpoint create --personal csnet-4090
4. Initiate transfer:
       globus transfer --recursive \\
           "sdss#sas/dr19/spectro/boss/redux/v6_1_3/spectra/lite" \\
           "csnet-4090:{outdir}/dr19_boss_lite"
5. Resume on failure:  globus task resume <task-id>

OPTION C: rsync from a local mirror (if your institution has one)
-----------------------------------------------------------------
    rsync -avP --partial \\
        rsync://<mirror>/sdss/dr19/spectro/boss/redux/v6_1_3/spectra/lite/ \\
        {outdir}/dr19_boss_lite/

Expected layout after download:
    {outdir}/
        dr19_boss_lite/
            <plate>/spec-<plate>-<mjd>-<fiberID>.fits   (~2M files)
        specObj-dr19.fits                                (catalog with labels)

Next step:
    python scripts/preprocess.py --raw {outdir} --out data/processed --survey sdss
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/raw/sdss"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(INSTRUCTIONS.format(outdir=args.out))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
