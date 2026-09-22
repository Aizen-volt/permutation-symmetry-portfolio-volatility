#!/usr/bin/env bash
# Replication of the full design on the stock universe (GPU jobs run sequentially).
cd /c/Code/magisterka
export PVOL_UNIVERSE=stocks
.venv/Scripts/python.exe scripts/18_extra_experiments.py E1 E3 > logs/stocks_e1e3.log 2>&1
.venv/Scripts/python.exe scripts/05_train.py --exp heldout --h 5 > logs/stocks_heldout.log 2>&1
.venv/Scripts/python.exe scripts/08_ablation_datasize.py > logs/stocks_ablation.log 2>&1
.venv/Scripts/python.exe scripts/13_minvar.py > logs/stocks_minvar.log 2>&1
.venv/Scripts/python.exe scripts/05_train.py --exp main --h 1 22 > logs/stocks_main_h1h22.log 2>&1
echo "stocks queue finished"
