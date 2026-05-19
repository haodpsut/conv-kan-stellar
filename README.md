# Conv-KAN Stellar — Multi-Survey Spectral Classification on NVIDIA H200

> **Team Agentra-Astro** — Vietnam AI Open Hackathon 2026 (NVIDIA Open Hackathons, DSAC Đà Nẵng, 09–12 June 2026)

Scaling **Convolutional Kolmogorov-Arnold Networks (Conv-KAN)** for stellar spectral classification (Harvard types O/B/A/F/G/K/M) on a combined ~12M-spectrum corpus from SDSS DR19 BOSS, LAMOST DR8, and SDSS APOGEE DR17, with cross-survey generalization tested on the CFLIB (Indo-US) library.

This work extends our prior paper currently under review at **IEEE Transactions on Artificial Intelligence** (Do, Pham, Pham, Nguyen — 2026).

---

## Quick start (5 minutes, CPU-only smoke test)

The smoke test trains 2 small models × 3 random seeds on a synthetic 7-class dataset, end-to-end in ~1 minute on a laptop CPU. Use it to verify your install before downloading real data.

```bash
# 1. Clone
git clone https://github.com/haodpsut/conv-kan-stellar.git
cd conv-kan-stellar

# 2. Create conda env (NO sudo required, NO system packages)
conda env create -f env.yml
conda activate csnet

# 3. Install this package in editable mode
pip install -e .

# 4. Run the multi-seed smoke test
python scripts/smoke_test.py

# Expected output (last lines):
#   [smoke] seed=42 model=conv_kan  acc=0.7XX
#   [smoke] seed=43 model=conv_kan  acc=0.7XX
#   [smoke] seed=44 model=conv_kan  acc=0.7XX
#   [smoke] seed=42 model=inception acc=0.6XX
#   ...
#   [smoke] PASS — multi-seed pipeline works end-to-end.
```

If the smoke test passes, your environment is ready for real-data training.

---

## Repository layout

```
conv-kan-stellar/
├── README.md              ← you are here
├── LICENSE                ← Apache 2.0
├── env.yml                ← conda env (no sudo, conda-only)
├── pyproject.toml         ← editable install
├── src/csnet/
│   ├── utils.py           ← seeding, logging, metrics
│   ├── data.py            ← synthetic generator + real-survey loaders
│   ├── models.py          ← Conv-KAN (proposed) + 5 baselines
│   ├── losses.py          ← ENS-Focal loss (Cui+2019)
│   └── train.py           ← training/eval CLI
├── configs/
│   ├── smoke.yaml         ← synthetic data, tiny model, 3 seeds (CPU OK)
│   ├── sdss_only.yaml     ← real SDSS DR19 BOSS, single survey
│   └── multi_survey.yaml  ← full corpus: SDSS + LAMOST + APOGEE
├── scripts/
│   ├── smoke_test.py      ← end-to-end multi-seed smoke (synthetic)
│   ├── download_sdss.py   ← STUB: documented download instructions
│   ├── download_lamost.py ← STUB
│   ├── download_apogee.py ← STUB
│   ├── preprocess.py      ← common 4096-px resampling + air→vacuum
│   └── run_seed_sweep.sh  ← bash loop: 30 seeds × N models on H200
└── tests/
    └── test_models.py     ← shape/forward-pass tests for every model
```

---

## Pre-hackathon dev workflow (RTX 4090, Ubuntu, conda-only)

```bash
# On the 4090 server (Ubuntu, no sudo)
git clone https://github.com/haodpsut/conv-kan-stellar.git
cd conv-kan-stellar
conda env create -f env.yml
conda activate csnet
pip install -e .

# Verify GPU is visible to PyTorch
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce RTX 4090

# Download data when disk is available (see scripts/download_*.py for instructions)
python scripts/download_sdss.py   --out data/raw/sdss
python scripts/download_lamost.py --out data/raw/lamost
python scripts/download_apogee.py --out data/raw/apogee

python scripts/preprocess.py --raw data/raw --out data/processed

# Train 1 model, 1 seed (sanity check on real data)
python -m csnet.train --config configs/sdss_only.yaml --model conv_kan --seed 42

# Full single-survey sweep, 30 seeds (will take ~5–7 days on 4090)
bash scripts/run_seed_sweep.sh conv_kan sdss_only 30
```

---

## Deploying on the NVIDIA H200 (hackathon environment)

The H200 instance provided by NVIDIA Open Hackathons typically gives shell access with a CUDA driver installed but **no sudo** and **conda-only** Python (similar constraints to our 4090 server). The same `env.yml` works, with one extra step for H200-specific optimizations.

### Step 1 — Create the base environment

```bash
ssh <your-h200-host>
cd ~
git clone https://github.com/haodpsut/conv-kan-stellar.git
cd conv-kan-stellar

# If conda is not installed yet, install Miniforge (no sudo needed):
#   wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
#   bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
#   eval "$($HOME/miniforge3/bin/conda shell.bash hook)"

conda env create -f env.yml
conda activate csnet
pip install -e .
```

### Step 2 — Install H200-specific FP8 + DALI (after env active)

These are pip-only and not pinned in env.yml because they require an H100/H200 GPU:

```bash
# FP8 Transformer Engine (needs Hopper or newer)
pip install transformer-engine

# Optional: NVIDIA DALI for GPU-side data loading
pip install nvidia-dali-cuda120 --extra-index-url https://developer.download.nvidia.com/compute/redist

# Optional: 1D Mamba baseline (needs CUDA toolkit headers from conda)
conda install -c conda-forge cudatoolkit-dev
pip install mamba-ssm causal-conv1d
```

### Step 3 — Verify FP8 path

```bash
python -c "
import torch
import transformer_engine.pytorch as te
assert torch.cuda.get_device_capability(0)[0] >= 9, 'Need Hopper (SM 90+)'
print('FP8 OK on', torch.cuda.get_device_name(0))
"
# Expected: FP8 OK on NVIDIA H200
```

### Step 4 — Transfer preprocessed data

If you preprocessed the corpus on your dev server, sync to H200:

```bash
# From dev server (4090):
rsync -avP --partial data/processed/ <user>@<h200-host>:~/conv-kan-stellar/data/processed/

# Estimated sizes after preprocessing (float32, 4096 px):
#   SDSS DR19 BOSS  ~32 GB
#   LAMOST DR8     ~160 GB
#   APOGEE DR17     ~12 GB (resampled from R~22500 to 4096 px)
#   Total          ~205 GB
```

### Step 5 — Profile + run full experiment matrix

```bash
# Day 1 of hackathon: profile baseline
nsys profile -o profile_baseline python -m csnet.train \
    --config configs/multi_survey.yaml --model conv_kan --seed 42 --max-steps 100

# Day 2-3: 30 seeds × 6 models × 3 dataset configs (~540 runs)
bash scripts/run_seed_sweep.sh all multi_survey 30

# Results land in results/<config>/<model>/seed_<N>/metrics.json
# Aggregate:
python -m csnet.train --aggregate results/ --out results/summary.csv
```

---

## What `csnet.train` does

A single invocation:

```bash
python -m csnet.train --config <cfg.yaml> --model <name> --seed <int>
```

1. Loads the YAML config (overridable via CLI flags).
2. Seeds all RNGs (Python, NumPy, PyTorch CPU + CUDA, deterministic algorithms).
3. Builds the requested model from `csnet.models.MODELS[<name>]`.
4. Loads train/val/test splits (synthetic for smoke, real for production configs).
5. Trains with ENS-Focal loss (Cui+2019, β=0.9999) for `--epochs` epochs.
6. Logs per-epoch metrics to MLflow (or stdout-only if `--no-mlflow`).
7. Evaluates on test set + cross-survey CFLIB benchmark.
8. Writes `results/<run_name>/metrics.json` + checkpoint.

Six models registered in `csnet.models.MODELS`:

| Key | Architecture | Status |
|---|---|---|
| `conv_kan` | Conv1D backbone (5 blocks) + KAN classifier, B-spline grid 5, order 3 | **proposed** |
| `inception` | InceptionTime 1D (Fawaz+2020), 6 inception modules with bottleneck | baseline |
| `se_resnet` | 1D ResNet-18 with Squeeze-and-Excitation blocks | baseline |
| `cnn_transformer` | CNN feature extractor + 2-layer Transformer encoder | baseline |
| `starnet` | StarNet (Fabbro+2018), 2 conv + 2 dense — designed for stellar spectra | baseline |
| `mamba1d` | 1D Mamba (Gu+Dao 2023), selective state-space model | baseline (H200 only) |

---

## Reproducing the prior IEEE TAI results

Once SDSS DR19 BOSS data is downloaded:

```bash
python -m csnet.train --config configs/sdss_only.yaml --model conv_kan --seed 42
```

Expected single-seed numbers (matching the under-review paper):

| Metric | Conv-KAN | InceptionTime | SE-ResNet | CNN-Transformer |
|---|---|---|---|---|
| SDSS test accuracy | 87.7% | 80.5% | 84.0% | 84.5% |
| CFLIB cross-survey | 73.2% | 61.2% | 60.2% | 60.6% |
| Macro F1 (SDSS) | 0.806 | 0.677 | 0.724 | 0.753 |
| Per-epoch time (RTX 4090) | 25 s | 945 s | 722 s | 631 s |

(Conv-KAN is fast because the KAN classifier head replaces a large MLP, while the conv backbone is shallow.)

---

## Citing

If you use this code, please cite the prior work it builds on:

```bibtex
@article{do2026convkan,
  title   = {Stellar Spectral Classification via Convolutional Kolmogorov-Arnold Networks},
  author  = {Do, Phuc Hao and Pham, Minh Tuan and Pham, Nhat Khanh and Nguyen, Nang Hung Van},
  journal = {IEEE Transactions on Artificial Intelligence},
  year    = {2026},
  note    = {under review}
}
```

---

## Authors and acknowledgements

**Hackathon team (Agentra-Astro):**
- Phuc Hao Do — Da Nang Architecture University (DAU)
- Phu Huu Le
- Hoang Ngoc Nguyen

**Prior work coauthors** (IEEE TAI submission this repo extends):
- Phuc Hao Do (DAU, corresponding)
- Minh Tuan Pham, Nhat Khanh Pham, Nang Hung Van Nguyen (DUT / UDN)

This work is supported by the NVIDIA Open Hackathons program and hosted by the Da Nang Semiconductor and AI Center (DSAC), Department of Science and Technology, Da Nang City.

Licensed under the Apache License 2.0 (see `LICENSE`).
