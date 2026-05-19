#!/usr/bin/env bash
# Multi-seed sweep. Usage:
#     bash scripts/run_seed_sweep.sh <model|all> <config_stem> <n_seeds>
#
# Examples:
#     bash scripts/run_seed_sweep.sh conv_kan sdss_only 30
#     bash scripts/run_seed_sweep.sh all      multi_survey 30
#
# Writes results/<config_stem>/<model>/seed_<N>/metrics.json for each run.
set -euo pipefail

MODEL="${1:-conv_kan}"
CFG_STEM="${2:-smoke}"
N_SEEDS="${3:-30}"

CFG_PATH="configs/${CFG_STEM}.yaml"
if [[ ! -f "$CFG_PATH" ]]; then
  echo "Config not found: $CFG_PATH"
  exit 1
fi

if [[ "$MODEL" == "all" ]]; then
  MODELS=(conv_kan inception se_resnet cnn_transformer starnet mamba1d)
else
  MODELS=("$MODEL")
fi

LOG_DIR="logs/${CFG_STEM}"
mkdir -p "$LOG_DIR"

for m in "${MODELS[@]}"; do
  for ((s=0; s<N_SEEDS; s++)); do
    seed=$((42 + s))
    log="${LOG_DIR}/${m}_seed${seed}.log"
    echo "[sweep] model=${m} seed=${seed}  ->  $log"
    python -m csnet.train --config "$CFG_PATH" --model "$m" --seed "$seed" 2>&1 | tee "$log" \
      || echo "[sweep] WARN — model=${m} seed=${seed} exited non-zero (continuing)"
  done
done

echo "[sweep] done."
echo "[sweep] aggregate with:  python scripts/aggregate_results.py results/${CFG_STEM}/"
