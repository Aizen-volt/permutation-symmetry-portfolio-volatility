"""Ablation: does architectural symmetry matter more when the training history is short?

Networks are retrained with only the last K calendar years of training data (K = 2, 4) before the validation year;
forecasts are stored under the tag '<model>-K<K>' next to the full-history runs (h = 5).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvol.data import load_panel  # noqa: E402
from pvol.experiment import Experiment, load_tuned_hp  # noqa: E402

MODELS = ["MLP", "DeepSets", "SetTransformer", "EQCov", "GRU", "SetGRU"]

if __name__ == "__main__":
    hp = load_tuned_hp()
    ex = Experiment("main", load_panel())
    for K in (2, 4):
        for seed in (0, 1):
            for m in MODELS:
                ex.run_nn(m, 5, [seed], hp=hp[m], tag=f"{m}-K{K}", train_years=K)
