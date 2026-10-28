"""Fifth series (T, U), hypotheses and decision rules fixed before running.

T: main design with the settings of the broader search (scripts/36_tune_wide.py, best_hp_T.json), tag '<model>-T';
   networks and Linear, h = 1, 5, 22, seeds 0-2.
U: MLP with slots ordered by asset index (MLP-idsort, settings of MLP-sorted), h = 5: main design (tag MLP-idsort,
   seeds 0-2), raw inputs (MLP-idsort-RAW, seeds 0-2), fixed pool of 50 portfolios (MLP-idsort-P50, seeds 0-1).

Usage: [PVOL_UNIVERSE=stocks] python scripts/37_fifth_series.py U T
Resumable: finished (tag, seed) pairs are skipped.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvol.config import RESULTS  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.experiment import Experiment, load_tuned_hp  # noqa: E402
from pvol.train import TrainCfg  # noqa: E402

T_MODELS = ["MLP", "GRU", "DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov", "Linear"]

if __name__ == "__main__":
    todo = sys.argv[1:] or ["U", "T"]
    ex = Experiment("main", load_panel())
    for e in todo:
        if e == "U":
            hp = load_tuned_hp()["MLP-sorted"]
            for seed in (0, 1, 2):
                ex.run_nn("MLP-idsort", 5, [seed], hp=hp, tag="MLP-idsort")
                ex.run_nn("MLP-idsort", 5, [seed], hp=hp, cfg=TrainCfg(raw_inputs=True), tag="MLP-idsort-RAW")
            for seed in (0, 1):
                ex.run_nn("MLP-idsort", 5, [seed], hp=hp, cfg=TrainCfg(portfolio_pool=50), tag="MLP-idsort-P50")
        elif e == "T":
            best = json.loads((RESULTS / "tuning" / "best_hp_T.json").read_text())
            for h in (5, 1, 22):
                for seed in (0, 1, 2):
                    for m in T_MODELS:
                        ex.run_nn(m, h, [seed], hp=best[m]["hp"], cfg=TrainCfg(**best[m]["cfg"]), tag=f"{m}-T")
        else:
            raise SystemExit(f"unknown experiment {e}")
        print(f"{e} finished", flush=True)
