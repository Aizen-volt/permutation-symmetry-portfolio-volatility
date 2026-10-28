#!/usr/bin/env bash
# sixth series V6: dropout control of MLP-sorted, both universes, then the tests
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
for u in etf stocks; do
  PVOL_UNIVERSE=$u $PY scripts/47_dropout_control.py > logs/v6_$u.log 2>&1
done
