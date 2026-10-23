"""Fourth series of experiments (S1-S3), hypotheses and decision rules fixed before running.

LIN (S1): the linear benchmark on the information of the networks (model Linear), main design, h = 1, 5, 22, 3 seeds.
S2:       seeds 3-9 of the main networks at h = 1 (stock universe: also the 8-ordering averaging tags MLP-TTA, GRU-TTA;
          PVOL_UNIVERSE=stocks54: the R5 networks).
S3:       refit protocol (tag <model>-RF): the validation year chooses the number of epochs, then the network is retrained
          on training + validation data for max(best epoch, 5) epochs; h = 1, 22, 3 seeds.

Usage: [PVOL_UNIVERSE=stocks|stocks54] python scripts/25_fourth_series.py LIN S2 S3
Resumable: finished (tag, seed) pairs are skipped.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvol.config import UNIVERSE_NAME  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.experiment import Experiment, load_tuned_hp  # noqa: E402
from pvol.train import TrainCfg  # noqa: E402

NETS = ["MLP", "GRU", "DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov"]

if __name__ == "__main__":
    todo = sys.argv[1:] or ["LIN", "S2", "S3"]
    hp = load_tuned_hp()
    hp["Linear"] = {}
    ex = Experiment("main", load_panel())
    for e in todo:
        if e == "LIN":
            for h in (1, 5, 22):
                ex.run_nn("Linear", h, [0, 1, 2], hp={})
        elif e == "S2":
            for seed in range(3, 10):
                for m in NETS:
                    ex.run_nn(m, 1, [seed], hp=hp[m])
                if UNIVERSE_NAME == "stocks":
                    for m in ("MLP", "GRU"):
                        ex.run_nn(m, 1, [seed], hp=hp[m], tag=f"{m}-TTA", tta_k=8)
        elif e == "S3":
            for h in (1, 22):
                for seed in (0, 1, 2):
                    for m in NETS:
                        ex.run_nn(m, h, [seed], hp=hp[m], cfg=TrainCfg(refit=True), tag=f"{m}-RF")
        else:
            raise SystemExit(f"unknown experiment {e}")
        print(f"{e} finished", flush=True)
