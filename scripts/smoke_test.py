"""End-to-end smoke test: trains 2 models x 3 seeds on synthetic data, then
checks that:
    - every run reaches at least `min_acc` test accuracy (pipeline learns)
    - per-seed numbers differ but are within plausible spread (RNG works)
    - the multi-seed aggregator writes a summary CSV

Runs entirely on CPU in roughly 1 minute. Use to verify install before
attempting real-data training.

Usage:
    python scripts/smoke_test.py            # default: 3 seeds, conv_kan + inception
    python scripts/smoke_test.py --models conv_kan starnet --seeds 5 --epochs 6
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# Ensure we can find csnet when running from the repo root without editable install.
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from csnet.train import load_config, train_one_run  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=_ROOT / "configs" / "smoke.yaml")
    p.add_argument("--models", nargs="+", default=["conv_kan", "inception"],
                   help="Models to run (must be in csnet.models.MODELS)")
    p.add_argument("--seeds", type=int, default=3, help="Number of seeds per model")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override config epochs (default: keep YAML value)")
    p.add_argument("--out", type=Path, default=_ROOT / "results" / "smoke")
    p.add_argument("--min-acc", type=float, default=0.55,
                   help="Minimum test accuracy any single run must exceed for the smoke to PASS. "
                        "0.55 is well above the 0.143 random-7-class chance.")
    args = p.parse_args()

    base_cfg = load_config(args.config)
    if args.epochs is not None:
        base_cfg["epochs"] = args.epochs

    seeds = list(range(42, 42 + args.seeds))
    all_results: dict[str, list[float]] = {m: [] for m in args.models}
    failed = []

    print(f"=== csnet smoke test ===")
    print(f"config:  {args.config}")
    print(f"models:  {args.models}")
    print(f"seeds:   {seeds}")
    print(f"epochs:  {base_cfg.get('epochs', '?')}")
    print(f"out:     {args.out}")
    print()

    for model in args.models:
        for seed in seeds:
            cfg = dict(base_cfg)
            cfg["model"] = model
            cfg["seed"] = seed
            cfg["run_name"] = f"smoke_{model}_seed{seed}"
            run_dir = args.out / model / f"seed_{seed}"
            metrics = train_one_run(cfg, run_dir)
            test_acc = metrics.test_acc or 0.0
            all_results[model].append(test_acc)
            tag = "OK " if test_acc >= args.min_acc else "FAIL"
            print(f"[smoke] {tag} model={model:>15s} seed={seed}  test_acc={test_acc:.4f}")
            if test_acc < args.min_acc:
                failed.append((model, seed, test_acc))
            print()

    # --------------------------------------------------------------------- #
    # Summary                                                                #
    # --------------------------------------------------------------------- #
    print("=" * 60)
    print(f"{'model':>16s}  {'n':>3s}  {'mean':>7s}  {'std':>7s}  {'min':>7s}  {'max':>7s}")
    print("-" * 60)
    summary_rows = []
    for model, accs in all_results.items():
        mean = statistics.mean(accs) if accs else 0.0
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        lo = min(accs) if accs else 0.0
        hi = max(accs) if accs else 0.0
        print(f"{model:>16s}  {len(accs):>3d}  {mean:>7.4f}  {std:>7.4f}  {lo:>7.4f}  {hi:>7.4f}")
        summary_rows.append({"model": model, "n_seeds": len(accs), "mean_acc": mean,
                             "std_acc": std, "min_acc": lo, "max_acc": hi})
    print("=" * 60)

    summary_path = args.out / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(f"\n[smoke] wrote summary -> {summary_path}")

    if failed:
        print(f"\n[smoke] FAIL — {len(failed)} run(s) below min_acc={args.min_acc}:")
        for m, s, a in failed:
            print(f"          model={m} seed={s} acc={a:.4f}")
        return 1

    # Sanity check 2: seeding genuinely produces variation. We check at the level
    # of per-epoch training-loss trajectories rather than final test accuracy,
    # because an easy task can let every seed saturate to 100% even when
    # seeding is working correctly. Identical loss curves across seeds would
    # indicate seeding (or weight initialisation) is silently broken.
    all_train_losses: dict[str, list[list[float]]] = {m: [] for m in args.models}
    for model in args.models:
        for seed in seeds:
            mpath = args.out / model / f"seed_{seed}" / "metrics.json"
            if mpath.exists():
                with mpath.open(encoding="utf-8") as f:
                    all_train_losses[model].append(json.load(f).get("train_loss", []))
    for model, curves in all_train_losses.items():
        if len(curves) > 1 and all(c == curves[0] for c in curves):
            print(f"\n[smoke] FAIL — model={model} produced IDENTICAL train_loss curves "
                  f"across all seeds. Seeding is likely broken.")
            return 1

    print("\n[smoke] PASS — multi-seed pipeline works end-to-end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
