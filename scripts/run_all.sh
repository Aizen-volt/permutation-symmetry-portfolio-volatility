#!/usr/bin/env bash
# Full pipeline in the order used for the paper. The neural steps need a CUDA GPU and take several days in total;
# every training script is resumable (finished (tag, seed) pairs are skipped), so the script can be re-run after an
# interruption. Tables and figures are written to paper/tables and paper/figures.
set -eo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}
mkdir -p logs

universe() {  # $1 = etf | stocks, $2 = horizons of the held-out-assets experiment
    export PVOL_UNIVERSE=$1
    $PY scripts/01_download.py
    $PY scripts/02_build_panel.py
    $PY scripts/07_descriptives.py
    $PY scripts/03_tune.py
    $PY scripts/04_baselines.py
    $PY scripts/04b_harx.py
    $PY scripts/04c_shrinkage.py
    $PY scripts/26_benchmarks_ext.py
    $PY scripts/27_gbm.py
    $PY scripts/05_train.py --exp main
    $PY scripts/05_train.py --exp heldout --h $2
    $PY scripts/08_ablation_datasize.py
    $PY scripts/13_minvar.py
    $PY scripts/18_extra_experiments.py E2 E1 E3
    $PY scripts/23_third_series.py R1 R2 R3 R4
    $PY scripts/25_fourth_series.py LIN S2
    $PY scripts/06_evaluate.py main
    $PY scripts/06_evaluate.py heldout
    $PY scripts/12_vol_targeting.py main
    $PY scripts/15_hypotheses.py main
    $PY scripts/15_hypotheses.py main --ext
    $PY scripts/15_hypotheses.py heldout
    $PY scripts/17_economic_tests.py
    $PY scripts/19_extra_hypotheses.py
    $PY scripts/24_third_hypotheses.py
    $PY scripts/29_fourth_hypotheses.py
    $PY scripts/28_fourth_analyses.py
    $PY scripts/22_compute_cost.py
}

universe etf "5 1 22" 2>&1 | tee logs/etf.log
# alternative training protocol and refit protocol (ETF: protocol B at h = 5; stocks: at h = 1), retuning checks
PVOL_UNIVERSE=etf $PY scripts/14_robust_protocol.py 5
PVOL_UNIVERSE=etf $PY scripts/25_fourth_series.py S3
PVOL_UNIVERSE=etf $PY scripts/15_hypotheses.py main --ext --suffix=-P2
PVOL_UNIVERSE=etf $PY scripts/15_hypotheses.py main --ext --suffix=-RF
PVOL_UNIVERSE=etf $PY scripts/03_tune.py --h 1 --tag h1
PVOL_UNIVERSE=etf $PY scripts/03_tune.py --h 22 --tag h22
PVOL_UNIVERSE=etf $PY scripts/31_determinism.py
# ETF universe only: EQ-Cov diagnostics, validation diagnostic at h = 22
PVOL_UNIVERSE=etf $PY scripts/11_eqcov_diagnostics.py
PVOL_UNIVERSE=etf $PY scripts/16_validation_diagnostic_h22.py

universe stocks 5 2>&1 | tee logs/stocks.log
PVOL_UNIVERSE=stocks $PY scripts/14_robust_protocol.py 1
PVOL_UNIVERSE=stocks $PY scripts/25_fourth_series.py S3
PVOL_UNIVERSE=stocks $PY scripts/15_hypotheses.py main --ext --suffix=-P2
PVOL_UNIVERSE=stocks $PY scripts/15_hypotheses.py main --ext --suffix=-RF
PVOL_UNIVERSE=stocks $PY scripts/03_tune.py --h 5 --tag h5
PVOL_UNIVERSE=stocks $PY scripts/03_tune.py --h 1 --tag h1

# 54-stock subsample (R5 and its seeds 3-9); prices are read from the stock universe
export PVOL_UNIVERSE=stocks54
$PY scripts/02_build_panel.py
$PY scripts/03_tune.py
$PY scripts/04_baselines.py main
$PY scripts/23_third_series.py R5
$PY scripts/25_fourth_series.py S2
$PY scripts/24_third_hypotheses.py
$PY scripts/29_fourth_hypotheses.py
unset PVOL_UNIVERSE

# checks over both universes: second data source, retuning
PVOL_UNIVERSE=etf $PY scripts/33_data_crosscheck.py
PVOL_UNIVERSE=stocks $PY scripts/33_data_crosscheck.py
$PY scripts/34_tuning_check.py
PVOL_UNIVERSE=etf $PY scripts/35_refit_summary.py
PVOL_UNIVERSE=stocks $PY scripts/35_refit_summary.py
# fifth series: sort key (U) and broader search with validation over three years (T)
for u in etf stocks; do
    PVOL_UNIVERSE=$u $PY scripts/37_fifth_series.py U
    PVOL_UNIVERSE=$u $PY scripts/36_tune_wide.py
    PVOL_UNIVERSE=$u $PY scripts/37_fifth_series.py T
    PVOL_UNIVERSE=$u $PY scripts/38_fifth_hypotheses.py
    PVOL_UNIVERSE=$u $PY scripts/39_equivalence.py
    PVOL_UNIVERSE=$u $PY scripts/49_mcs_tmax.py
    PVOL_UNIVERSE=$u $PY scripts/40_anchor_distance.py
    PVOL_UNIVERSE=$u $PY scripts/41_sorted_ties.py
done
# sixth series: V1 (ties of the sort), V2 (feature ablation of Linear), V3 (matched size), V5 (min-var optimizer),
# V6 (dropout of MLP-sorted); V4 (results by test year) reads both universes
for u in etf stocks; do
    PVOL_UNIVERSE=$u $PY scripts/42_sixth_series.py V1 V3 V2
    PVOL_UNIVERSE=$u $PY scripts/41_sorted_ties.py MLP-tsort
    PVOL_UNIVERSE=$u $PY scripts/44_minvar_check.py
    PVOL_UNIVERSE=$u $PY scripts/45_sixth_hypotheses.py
    PVOL_UNIVERSE=$u $PY scripts/47_dropout_control.py
    PVOL_UNIVERSE=$u $PY scripts/48_v6_hypotheses.py
done
$PY scripts/43_yearly.py
# seventh series: W1 (Linear with raw inputs), W2-W3 (SPA, fixed-smoothing tests), W4 (EQ-Cov min-var exact)
for u in etf stocks; do
    PVOL_UNIVERSE=$u $PY scripts/50_seventh_series.py
    PVOL_UNIVERSE=$u $PY scripts/51_seventh_analyses.py
    PVOL_UNIVERSE=$u $PY scripts/52_minvar_eqcov_exact.py
done
# precision of every comparison and global multiplicity over all series
$PY scripts/32_precision_multiplicity.py

# tables and figures for both universes
$PY scripts/10_tables_tex.py
$PY scripts/20_scoreboard.py
$PY scripts/21_hypothesis_tables.py
$PY scripts/30_fourth_tables.py
$PY scripts/09_figures.py
$PY scripts/53_forest_plot.py
$PY scripts/54_learning_curves.py
$PY scripts/46_supplement_tables.py
bash scripts/collect_results.sh
