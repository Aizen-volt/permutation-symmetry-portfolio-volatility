#!/usr/bin/env bash
# sixth series, second part: tie consistency of MLP-tsort (V1) and the min-var optimizer check (V5), both universes
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
for u in etf stocks; do
  PVOL_UNIVERSE=$u $PY scripts/41_sorted_ties.py MLP-tsort > logs/sixth_ties_$u.log 2>&1
  PVOL_UNIVERSE=$u $PY scripts/44_minvar_check.py > logs/sixth_minvar_$u.log 2>&1
done
