"""Sixth series V6: dropout control for the MLP-sorted / MLP pair (decision rule fixed before running).

MLP-sorted selected dropout 0.1 and the MLP dropout 0.3 from the same grid; V6 retrains MLP-sorted with dropout 0.3 (the
MLP's setting), all else as tuned, under the conditions of R2 (h = 5): main design and raw inputs with seeds 0-2, fixed
pools of 50 and 500 portfolios with seeds 0-1. Tags: MLP-sorted-d3, MLP-sorted-RAW-d3, MLP-sorted-P50-d3, -P500-d3.

Usage: [PVOL_UNIVERSE=stocks] python scripts/47_dropout_control.py
Resumable: finished (tag, seed) pairs are skipped.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvol.data import load_panel  # noqa: E402
from pvol.experiment import Experiment, load_tuned_hp  # noqa: E402
from pvol.train import TrainCfg  # noqa: E402

if __name__ == "__main__":
    hp = {**load_tuned_hp()["MLP-sorted"], "dropout": 0.3}
    ex = Experiment("main", load_panel())
    for seed in (0, 1, 2):
        ex.run_nn("MLP-sorted", 5, [seed], hp=hp, tag="MLP-sorted-d3")
        ex.run_nn("MLP-sorted", 5, [seed], hp=hp, cfg=TrainCfg(raw_inputs=True), tag="MLP-sorted-RAW-d3")
    for P in (50, 500):
        for seed in (0, 1):
            ex.run_nn("MLP-sorted", 5, [seed], hp=hp, cfg=TrainCfg(portfolio_pool=P), tag=f"MLP-sorted-P{P}-d3")
    print("V6 finished", flush=True)
