# Permutation-symmetric neural networks for multi-asset portfolio volatility forecasting

Code and data pipeline for the article/thesis. Everything is reproducible from public Yahoo Finance data.

## Setup
```
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python -m pip install -r requirements.txt
```

## Pipeline
| step | script | output |
|---|---|---|
| 1 | `scripts/01_download.py` | `data/raw/ohlcv_adjusted.parquet` (54 ETFs, daily adjusted OHLCV) |
| 2 | `scripts/02_build_panel.py` | `data/processed/panel.npz` (returns, asset features, correlations, EWMA covariance) |
| 3 | `scripts/03_tune.py` | `results/tuning/` (4 configurations per network, tuning fold 2008–2016 / 2017–2018) |
| 4 | `scripts/04_baselines.py`, `scripts/04b_harx.py` | classical forecasts in `results/<exp>/h<h>/preds/` |
| 5 | `scripts/05_train.py --exp main` / `--exp heldout` | neural forecasts (per seed) + permutation-consistency forecasts |
| 6 | `scripts/06_evaluate.py main` | tables in `results/<exp>/tables/` |
| 7 | `scripts/07_descriptives.py` | descriptive statistics of the universe |

Tests: `.venv\Scripts\python -m pytest tests` (permutation symmetry of the models, target alignment, absence of look-ahead).

## Layout
- `src/pvol/universe.py` – ETF universe and asset classes
- `src/pvol/data.py` – variance proxies, asset features, rolling/EWMA correlations
- `src/pvol/dataset.py` – GPU panel, random portfolio sampler, batch construction, time splits
- `src/pvol/models.py` – PortMLP, MLP, MLP-sorted, GRU, DeepSets, PENet, SetTransformer, EQCov
- `src/pvol/baselines.py` – HIST, EWMA, HAR, HAR-X, GARCH, GJR-GARCH, DCC-GARCH
- `src/pvol/train.py` – QLIKE training loop with EMA weights and early stopping
- `src/pvol/evaluate.py` – QLIKE/MSE, Diebold–Mariano (HLN), Model Confidence Set, Mincer–Zarnowitz, VaR backtests
- `paper/` – manuscript (LaTeX) and bibliography
