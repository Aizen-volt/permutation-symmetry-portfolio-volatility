#!/usr/bin/env bash
# sixth series (docs/JOURNAL.md, section 4g): training runs V1-V3 for both universes, then the min-var check V5
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
for u in etf stocks; do
  PVOL_UNIVERSE=$u $PY scripts/42_sixth_series.py V1 V3 V2 > logs/sixth_$u.log 2>&1
done
