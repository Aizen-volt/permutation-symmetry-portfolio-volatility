#!/usr/bin/env bash
# Runs after the main ETF training chain: min-variance test, protocol-B robustness, stock-universe training.
cd /c/Code/magisterka
until [ -f results/main/h5/preds/SetGRU-K4__s1.npy ]; do sleep 60; done
.venv/Scripts/python.exe scripts/13_minvar.py > logs/minvar.log 2>&1
.venv/Scripts/python.exe scripts/14_robust_protocol.py > logs/robust_protocol.log 2>&1
PVOL_UNIVERSE=stocks .venv/Scripts/python.exe scripts/05_train.py --exp main --h 5 --seeds 0 1 2 > logs/train_stocks.log 2>&1
echo "queue finished"
