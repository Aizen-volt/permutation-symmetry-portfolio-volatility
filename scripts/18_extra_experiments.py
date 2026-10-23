"""Second series of experiments (hypotheses and decision rules fixed before running), h = 5.

E1 (tag -RAW):   no engineered portfolio-level summaries (constant anchor, no portfolio HAR, no Euler contributions).
E2 (tag -P50/-P500): training and validation on a fixed pool of 50 / 500 portfolios with a fixed asset order.
E3 (tag -N54):   training on portfolios with up to 54 assets; the MLP gets 54 slots.

Usage: python scripts/18_extra_experiments.py E2 E1 E3      (PVOL_UNIVERSE=stocks ... E2 for the stock universe)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvol.data import load_panel  # noqa: E402
from pvol.experiment import Experiment, load_tuned_hp  # noqa: E402
from pvol.train import TrainCfg  # noqa: E402

H = 5
E1_MODELS = ["MLP", "DeepSets", "PENet", "SetTransformer", "GRU", "SetGRU"]
E2_MODELS = ["MLP", "DeepSets", "PENet", "SetTransformer", "GRU", "SetGRU", "EQCov"]
E3_MODELS = ["MLP", "DeepSets", "PENet", "SetTransformer", "EQCov"]

if __name__ == "__main__":
    todo = sys.argv[1:] or ["E2", "E1", "E3"]
    hp = load_tuned_hp()
    ex = Experiment("main", load_panel())
    for e in todo:
        if e == "E2":
            for P in (50, 500):
                for seed in (0, 1):
                    for m in E2_MODELS:
                        ex.run_nn(m, H, [seed], hp=hp[m], cfg=TrainCfg(portfolio_pool=P), tag=f"{m}-P{P}")
        elif e == "E1":
            for seed in (0, 1, 2):
                for m in E1_MODELS:
                    ex.run_nn(m, H, [seed], hp=hp[m], cfg=TrainCfg(raw_inputs=True), tag=f"{m}-RAW")
        elif e == "E3":
            for seed in (0, 1):
                for m in E3_MODELS:
                    hp_m = {**hp[m], "n_slots": 54} if m == "MLP" else hp[m]
                    ex.run_nn(m, H, [seed], hp=hp_m, cfg=TrainCfg(n_max=54), tag=f"{m}-N54")
        else:
            raise SystemExit(f"unknown experiment {e}")
        print(f"{e} finished", flush=True)
