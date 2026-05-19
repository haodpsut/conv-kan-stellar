"""Aggregate multi-seed runs into a summary CSV + console table.

Usage:
    python scripts/aggregate_results.py results/sdss_only/
    python scripts/aggregate_results.py results/smoke/        --out smoke_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def find_metrics(root: Path) -> list[dict]:
    rows = []
    for mfile in root.rglob("metrics.json"):
        try:
            with mfile.open(encoding="utf-8") as f:
                m = json.load(f)
        except Exception as e:
            print(f"WARN: failed to read {mfile}: {e}")
            continue
        m["_path"] = str(mfile)
        rows.append(m)
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    out = []
    for model, runs in sorted(by_model.items()):
        test_accs = [r["test_acc"] for r in runs if r.get("test_acc") is not None]
        f1s = [r["test_macro_f1"] for r in runs if r.get("test_macro_f1") is not None]
        cflib = [r["cflib_acc"] for r in runs if r.get("cflib_acc") is not None]
        walls = [r["wall_seconds"] for r in runs if r.get("wall_seconds") is not None]
        out.append({
            "model": model,
            "n_seeds": len(runs),
            "test_acc_mean": statistics.mean(test_accs) if test_accs else None,
            "test_acc_std": statistics.stdev(test_accs) if len(test_accs) > 1 else 0.0,
            "test_macro_f1_mean": statistics.mean(f1s) if f1s else None,
            "test_macro_f1_std": statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
            "cflib_acc_mean": statistics.mean(cflib) if cflib else None,
            "cflib_acc_std": statistics.stdev(cflib) if len(cflib) > 1 else 0.0,
            "wall_mean_s": statistics.mean(walls) if walls else None,
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path, help="Directory containing per-seed metrics.json files (recursive)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV (default: <root>/summary.csv)")
    args = p.parse_args()

    rows = find_metrics(args.root)
    if not rows:
        print(f"No metrics.json found under {args.root}")
        return 1
    summary = aggregate(rows)

    out = args.out or args.root / "summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"{'model':>16s}  {'n':>3s}  {'acc_mean':>9s}  {'acc_std':>8s}  "
          f"{'f1_mean':>9s}  {'cflib_mean':>10s}  {'wall_s':>8s}")
    for r in summary:
        print(f"{r['model']:>16s}  {r['n_seeds']:>3d}  "
              f"{(r['test_acc_mean'] or 0):>9.4f}  {(r['test_acc_std'] or 0):>8.4f}  "
              f"{(r['test_macro_f1_mean'] or 0):>9.4f}  "
              f"{(r['cflib_acc_mean'] or 0):>10.4f}  "
              f"{(r['wall_mean_s'] or 0):>8.1f}")
    print(f"\n-> wrote {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
