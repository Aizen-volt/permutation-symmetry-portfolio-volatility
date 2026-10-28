#!/usr/bin/env bash
# Copy the small result files (tables of the evaluation, tuning choices, evaluation portfolios) from the results*
# directories, which are not under version control, to replication/, which is. The forecasts (results*/*/h*/preds,
# about 9 GB) and the per-subset minimum-variance records (minvar_h5.csv, minvar_check*.csv
# except the summary) are archived separately.
set -eo pipefail
cd "$(dirname "$0")/.."
rm -rf replication
for R in results results_stocks results_stocks54; do
    [ -d "$R" ] || continue
    for E in main heldout; do
        [ -d "$R/$E" ] || continue
        mkdir -p "replication/$R/$E"
        cp "$R/$E"/*.json "replication/$R/$E/"
        if [ -d "$R/$E/tables" ]; then
            mkdir -p "replication/$R/$E/tables"
            find "$R/$E/tables" -maxdepth 1 -name '*.csv' ! -name 'minvar_h5.csv' ! -name 'minvar_check.csv' ! -name 'minvar_check_20*.csv' -exec cp {} "replication/$R/$E/tables/" \;
        fi
    done
    for D in tuning descriptives; do
        [ -d "$R/$D" ] || continue
        mkdir -p "replication/$R/$D"
        find "$R/$D" -maxdepth 1 \( -name '*.csv' -o -name '*.json' -o -name '*.jsonl' -o -name '*.txt' \) \
            -exec cp {} "replication/$R/$D/" \;
    done
done
du -sh replication
