"""Robustness: alternative training protocol applied identically to all networks (h = 5, 3 seeds).

Protocol B trains for at least 5 epochs, uses patience 15 and averages the EMA weights of the three best
validation epochs, which makes the selected model less sensitive to noise in the validation loss. Forecasts are
stored under the tag '<model>-P2'.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvol.data import load_panel  # noqa: E402
from pvol.experiment import Experiment, load_tuned_hp  # noqa: E402
from pvol.models import MODELS  # noqa: E402
from pvol.train import TrainCfg  # noqa: E402

if __name__ == "__main__":
    hp = load_tuned_hp()
    cfg = TrainCfg(min_epochs=5, patience=15, top_k_avg=3)
    ex = Experiment("main", load_panel())
    for seed in (0, 1, 2):
        for m in MODELS:
            ex.run_nn(m, 5, [seed], hp=hp[m], cfg=cfg, tag=f"{m}-P2")
