"""Robustness: alternative training protocol applied identically to all networks (3 seeds).

Protocol B trains for at least 5 epochs, uses patience 15 and averages the EMA weights of the three best
validation epochs, which makes the selected model less sensitive to noise in the validation loss. Forecasts are
stored under the tag '<model>-P2'.

Usage: [PVOL_UNIVERSE=stocks] python scripts/14_robust_protocol.py [H ...]   (default: h = 5)
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
    hp["Linear"] = {}
    for h in [int(a) for a in sys.argv[1:]] or [5]:
        for seed in (0, 1, 2):
            for m in (m for m in MODELS if m in hp):  # DeepSetsP is an E1 variant without tuned settings
                ex.run_nn(m, h, [seed], hp=hp[m], cfg=cfg, tag=f"{m}-P2")
