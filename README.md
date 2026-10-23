# Permutation-symmetric neural networks for forecasting the volatility of multi-asset portfolios

Code for the paper. It downloads the data, builds the panels, estimates every model and writes every table and figure
of the paper and its supplementary material. All data are public daily prices from Yahoo Finance.

## Requirements
Python 3.10 and an NVIDIA GPU with CUDA 12.4 for the neural networks (the classical benchmarks run on the CPU). The
results in the paper were computed on Windows 11 with one GPU.
```
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python -m pip install -r requirements.txt
```
On Linux, use `python3.10` and `.venv/bin/python` (set `PY=.venv/bin/python` for `run_all.sh`).

## Reproducing the paper
```
bash scripts/run_all.sh
```
runs the whole pipeline in the order used for the paper. Training all networks takes several days on one GPU. The
training scripts are resumable: forecasts are saved per (model, seed) and finished pairs are skipped, so an interrupted
run can be restarted with the same command.

The universe is chosen by the environment variable `PVOL_UNIVERSE`:

| value | universe | data | results |
|---|---|---|---|
| `etf` (default) | 54 exchange-traded funds | `data/raw`, `data/processed` | `results/` |
| `stocks` | 122 large US stocks | `data/raw_stocks`, `data/processed_stocks` | `results_stocks/` |
| `stocks54` | stratified subsample of 54 of the stocks | `data/processed_stocks54` | `results_stocks54/` |

The sample runs from 2007-12-19 to 2026-09-18 (`src/pvol/config.py`); the test years are 2020-2026. Yahoo Finance
revises adjusted prices after dividends and splits, so a later download can differ slightly from the one used in the
paper. The price files are not redistributed because of the provider's terms of use.

## Scripts

Data, tuning and benchmarks

| script | purpose |
|---|---|
| `01_download.py` | daily adjusted OHLCV from Yahoo Finance |
| `02_build_panel.py` | returns, variance proxies, asset features, correlations, EWMA covariance (`panel.npz`) |
| `03_tune.py` | hyper-parameter selection on a fold before the test period (four configurations per network) |
| `04_baselines.py`, `04b_harx.py`, `04c_shrinkage.py` | targets and classical benchmarks: HIST, EWMA, HAR, HAR-X, GARCH, GJR-GARCH, DCC, Ledoit-Wolf, nonlinear shrinkage, DCC-NL |
| `26_benchmarks_ext.py` | HAR-WLS, LogHAR-X, HEAVY and a PCA factor correlation model |
| `27_gbm.py` | gradient-boosting benchmark on the features of the linear set model |
| `07_descriptives.py` | descriptive statistics of the universe |

Experiments (forecasts in `results*/<experiment>/h<h>/preds/<tag>__s<seed>.npy`)

| script | experiment | tags |
|---|---|---|
| `05_train.py` | main design (`--exp main`) and assets never seen in training (`--exp heldout`) | model names |
| `08_ablation_datasize.py` | shortened training histories | `-K2`, `-K4` |
| `14_robust_protocol.py` | alternative training protocol (ETF) | `-P2` |
| `18_extra_experiments.py` | second series: raw inputs, fixed portfolio pools, portfolios of up to 54 assets | `-RAW`, `-P50`, `-P500`, `-N54` |
| `23_third_series.py` | third series: fixed pools with random orderings, sorted inputs, averaging over orderings, Deep Sets with pair summaries, 54-stock subsample | `-P50s`, `-P500s`, `MLP-sorted-*`, `-TTA`, `DeepSetsP*` |
| `25_fourth_series.py` | fourth series: linear set model, seeds 3-9 at h = 1 | `Linear`, seeds 3-9 |
| `13_minvar.py` | minimum-variance portfolios built with each model | |
| `11_eqcov_diagnostics.py` | correlation weights and volatility adjustments of EQ-Cov | |
| `16_validation_diagnostic_h22.py` | diagnostic on the tuning fold at h = 22 | |

`25_fourth_series.py S3` (a refit protocol) is implemented but not reported in the paper.

Evaluation and tests (tables in `results*/<experiment>/tables/`)

| script | output |
|---|---|
| `06_evaluate.py` | losses by portfolio block, Diebold-Mariano tests, model confidence set, Mincer-Zarnowitz, permutation consistency, value-at-risk, losses per year and per seed |
| `12_vol_targeting.py`, `17_economic_tests.py` | volatility targeting and the tests of the economic evaluation |
| `15_hypotheses.py` | hypotheses H1-H4 (`--ext` adds the benchmarks of `26_benchmarks_ext.py`, `--suffix=-P2` the alternative protocol) |
| `19_extra_hypotheses.py`, `24_third_hypotheses.py`, `29_fourth_hypotheses.py` | hypotheses of the second, third and fourth series |
| `28_fourth_analyses.py` | robustness: Patton loss, test sample without 2020, Giacomini-White test, agreement of seeds, value-at-risk with Student-t and filtered historical simulation |
| `22_compute_cost.py` | parameters, training and forecast time |

Every hypothesis is decided by the same rule: model a is more accurate than model b if its mean QLIKE is lower and the
Diebold-Mariano test with the Harvey-Leybourne-Newbold correction rejects at 5% after a Holm correction within the
family of comparisons. The hypotheses and decision rules of each series were fixed before the series was run.

Tables and figures (read both universes, run once)

| script | output |
|---|---|
| `10_tables_tex.py`, `20_scoreboard.py`, `21_hypothesis_tables.py`, `30_fourth_tables.py` | LaTeX tables in `paper/tables/` |
| `09_figures.py` | PDF figures in `paper/figures/` (needs a LaTeX installation, the figures use `usetex`) |

No number in the tables is typed by hand: every table is generated from the CSV files above.

## Code
- `src/pvol/config.py`: paths, sample, horizons, test years
- `src/pvol/universe.py`: ETF and stock universes with asset classes and sectors
- `src/pvol/data.py`: variance proxies, asset features, rolling and EWMA correlations
- `src/pvol/dataset.py`: GPU panel, random portfolio sampler, features, time splits
- `src/pvol/portfolios.py`: fixed evaluation portfolios and portfolio blocks
- `src/pvol/models.py`: portfolio-level MLP, MLP, GRU, sorted MLP, Deep Sets, PE-Net, Set Transformer, Set-GRU, EQ-Cov, linear set model
- `src/pvol/train.py`: QLIKE training with EMA weights and early stopping
- `src/pvol/experiment.py`: expanding-window protocol, one refit per test year
- `src/pvol/baselines.py`, `src/pvol/baselines_ext.py`: classical benchmarks
- `src/pvol/evaluate.py`: losses, Diebold-Mariano, model confidence set, Mincer-Zarnowitz, value-at-risk backtests

Tests: `.venv\Scripts\python -m pytest tests` (permutation symmetry of the models, alignment of targets, no look-ahead).
