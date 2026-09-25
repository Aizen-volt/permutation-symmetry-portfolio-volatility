#!/usr/bin/env bash
# Third series (R1-R5, docs/JOURNAL.md section 4b). GPU queue; the classical benchmarks of stocks54 run on the CPU in
# parallel (run_review_baselines). Log: logs/review_*.log
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python
$PY scripts/23_review_experiments.py R1 R2 R3 R4 > logs/review_etf.log 2>&1
PVOL_UNIVERSE=stocks $PY scripts/23_review_experiments.py R1 R2 R3 R4 > logs/review_stocks.log 2>&1
PVOL_UNIVERSE=stocks54 $PY scripts/23_review_experiments.py R5 > logs/review_stocks54.log 2>&1
echo "queue finished" >> logs/review_stocks54.log
