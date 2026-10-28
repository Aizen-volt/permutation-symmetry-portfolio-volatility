# Permutation-symmetric neural networks for forecasting the volatility of multi-asset portfolios

[![tests](https://github.com/mateusz-kaszubowski/permutation-symmetry-portfolio-volatility/actions/workflows/tests.yml/badge.svg)](https://github.com/mateusz-kaszubowski/permutation-symmetry-portfolio-volatility/actions/workflows/tests.yml)

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
paper. The price files are not redistributed because of the provider's terms of use. The snapshot used in the paper
was downloaded on 2026-09-21; its SHA-256 checksums are:

```
8c8af11acaa49e9c10ab35155d9d9c0e366676031ff62590e966e70e76491aca  data/raw/ohlcv_adjusted.parquet
2e40d6c8bc2ca99dfda49ce9455493ffafe41832cdb7b61f566591d89c866d53  data/raw_stocks/ohlcv_adjusted.parquet
c1016bac854a0eb7810a76155564b837efeb0b56eeefb94c900c255fe4cd48b2  data/processed/panel.npz
e4d50d89cdf83ec38ce90de37617b2322e4555371907c550b0c5ffa98d625634  data/processed_stocks/panel.npz
b20fc3c3445ca0d51accb63dec8b30a33b1f4dcad4aa521a5ea8e0b5bf9bee52  data/processed_stocks54/panel.npz
```

The result tables in `replication/`, from which the tables and figures of the paper are built, do not depend on a new
download.

## Scripts

Data, tuning and benchmarks

| script | purpose |
|---|---|
| `01_download.py` | daily adjusted OHLCV from Yahoo Finance |
| `02_build_panel.py` | returns, variance proxies, asset features, correlations, EWMA covariance (`panel.npz`) |
| `03_tune.py` | hyper-parameter selection on a fold before the test period (four configurations per network; `--h`, `--tag` repeat the search at another horizon) |
| `04_baselines.py`, `04b_harx.py`, `04c_shrinkage.py` | targets and classical benchmarks: HIST, EWMA, HAR, HAR-X, GARCH, GJR-GARCH, DCC, Ledoit-Wolf, nonlinear shrinkage, DCC-NL |
| `26_benchmarks_ext.py` | HAR-WLS, LogHAR-X, HEAVY and a PCA factor correlation model |
| `27_gbm.py` | gradient-boosting benchmark on the features of the linear set model |
| `07_descriptives.py` | descriptive statistics of the universe |

Experiments (forecasts in `results*/<experiment>/h<h>/preds/<tag>__s<seed>.npy`)

| script | experiment | tags |
|---|---|---|
| `05_train.py` | main design (`--exp main`) and assets never seen in training (`--exp heldout`) | model names |
| `08_ablation_datasize.py` | shortened training histories | `-K2`, `-K4` |
| `14_robust_protocol.py` | alternative training protocol (ETF at h = 5, stocks at h = 1) | `-P2` |
| `18_extra_experiments.py` | second series: raw inputs, fixed portfolio pools, portfolios of up to 54 assets | `-RAW`, `-P50`, `-P500`, `-N54` |
| `23_third_series.py` | third series: fixed pools with random orderings, sorted inputs, averaging over orderings, Deep Sets with pair summaries, 54-stock subsample | `-P50s`, `-P500s`, `MLP-sorted-*`, `-TTA`, `DeepSetsP*` |
| `25_fourth_series.py` | fourth series: linear set model, seeds 3-9 at h = 1; `S3`: refit on training + validation data | `Linear`, seeds 3-9, `-RF` |
| `13_minvar.py` | minimum-variance portfolios built with each model | |
| `11_eqcov_diagnostics.py` | correlation weights and volatility adjustments of EQ-Cov | |
| `16_validation_diagnostic_h22.py` | diagnostic on the tuning fold at h = 22 | |

Evaluation and tests (tables in `results*/<experiment>/tables/`)

| script | output |
|---|---|
| `06_evaluate.py` | losses by portfolio block, Diebold-Mariano tests, model confidence set, Mincer-Zarnowitz, permutation consistency, value-at-risk, losses per year and per seed |
| `12_vol_targeting.py`, `17_economic_tests.py` | volatility targeting and the tests of the economic evaluation |
| `15_hypotheses.py` | hypotheses H1-H4 (`--ext` adds the benchmarks of `26_benchmarks_ext.py`, `--suffix=-P2` / `-RF` the alternative protocols) |
| `19_extra_hypotheses.py`, `24_third_hypotheses.py`, `29_fourth_hypotheses.py` | hypotheses of the second, third and fourth series |
| `28_fourth_analyses.py` | robustness: Patton loss, test samples without 2020 and without 2026, Giacomini-White test, agreement of seeds, value-at-risk with Student-t and filtered historical simulation |
| `22_compute_cost.py` | parameters, training and forecast time |
| `31_determinism.py` | retrains three networks twice and compares with the stored forecasts (`determinism.csv`) |
| `32_precision_multiplicity.py` | confidence intervals and detectable effects of every comparison; Holm and Benjamini-Hochberg over all comparisons |
| `33_data_crosscheck.py` | prices against a second source (Nasdaq, unadjusted) |
| `34_tuning_check.py` | hyper-parameters in use against the search repeated at other horizons and on the stocks |
| `35_refit_summary.py` | accuracy of the networks under the main, B and refit protocols; Linear against the symmetric networks under each |
| `36_tune_wide.py` | fifth series T: random search (21 configurations per learned model) with validation over 2017-2019 |
| `37_fifth_series.py` | fifth series: U (MLP with slots ordered by asset index), T (main design with the settings of the search) |
| `38_fifth_hypotheses.py` | tests of the fifth series (T1-T4, U1-U2), best validation epochs |
| `39_equivalence.py` | confidence intervals and equivalence tests (TOST, margins of 1 and 2 points; Bonferroni over the five networks) for S1b and H1 |
| `49_mcs_tmax.py` | Model Confidence Set with the T_max statistic for the in-range portfolios (the main tables use the range statistic) |
| `40_anchor_distance.py` | distance of the learned forecasts from the EWMA anchor and from the Linear correction |
| `41_sorted_ties.py` | permutation consistency of MLP-sorted (or of the model given as argument, e.g. `MLP-tsort`) on equally weighted portfolios (retrains it, h=5) |
| `42_sixth_series.py` | sixth series: V1 (MLP sorted by weight, ties broken by asset index), V2 (Linear without one feature group), V3 (networks of matched size) |
| `43_yearly.py` | V4: losses and Diebold-Mariano tests per test year |
| `44_minvar_check.py` | V5: minimum-variance weights from the exact solution (covariance models) and from several starts (networks) against the stored weights |
| `45_sixth_hypotheses.py` | tests of the sixth series (V1-V3) |
| `47_dropout_control.py` | V6: MLP-sorted retrained with the dropout of the MLP under the conditions of R2 |
| `48_v6_hypotheses.py` | tests of V6 |
| `50_seventh_series.py` | seventh series: W1 (Linear with raw inputs, h=5) |
| `51_seventh_analyses.py` | tests of the seventh series: W1, SPA test (W2), fixed-smoothing Diebold-Mariano tests (W3), forecast-to-realized ratios |
| `52_minvar_eqcov_exact.py` | W4: minimum-variance weights of EQ-Cov from the exact solution of its convex problem |
| `46_supplement_tables.py` | tables of the supplementary material (all models, secondary families, equivalence, series T, U, V, economic evaluation, cost and hyper-parameters) in `paper/supplement/tables/` |

Every hypothesis is decided by the same rule: model a is more accurate than model b if its mean QLIKE is lower and the
Diebold-Mariano test with the Harvey-Leybourne-Newbold correction rejects at 5% after a Holm correction within the
family of comparisons. The hypotheses and decision rules of each series were fixed before the series was run.

Tables and figures (read both universes, run once)

| script | output |
|---|---|
| `10_tables_tex.py`, `20_scoreboard.py`, `21_hypothesis_tables.py`, `30_fourth_tables.py` | LaTeX tables in `paper/tables/` |
| `09_figures.py` | PDF figures in `paper/figures/` (needs a LaTeX installation, the figures use `usetex`) |
| `53_forest_plot.py`, `54_learning_curves.py` | forest plot of H1, S1b and H2 and learning curves from the stored training histories |

No number in the tables is typed by hand: every table is generated from the CSV files above.

`scripts/collect_results.sh` copies the result tables, tuning logs and descriptive statistics (everything except the
forecasts) to `replication/`, which is under version control.

Training is deterministic on a given GPU and software stack: retraining with the same seed reproduces the stored
forecasts to within 1e-6 (`31_determinism.py`); other hardware or library versions can give slightly different
forecasts.

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

Tests: `.venv\Scripts\python -m pytest tests --cov=src/pvol` (every line of `src/pvol` is covered; about 30 s on
one GPU, the CPU works too). They check the permutation symmetry of the models, the alignment of the targets and the
absence of look-ahead, every benchmark and statistical test against a direct computation or a reference
implementation, short training runs of the networks and of the experiment runner, the Holm and Benjamini-Hochberg
corrections of the scripts, and that the verdicts and corrected p-values stored in `replication/` follow the
decision rule. They use `data/processed/panel.npz` (`scripts/02_build_panel.py`) when it exists; without the data they
build a panel of the same shape from simulated prices, so they also run on a fresh clone and on the CPU, as in the
continuous integration (`.github/workflows/tests.yml`, Ubuntu, CPU build of torch).

## License
MIT (see `LICENSE`), for the code and the result tables in `replication/`. The price data are not covered: they
are downloaded from Yahoo Finance under its terms of use.
