"""Train all neural networks with the tuned hyper-parameters (expanding window, one refit per test year).

Usage: python scripts/05_train.py --exp main --h 5 1 22 --seeds 0 1 2 [--models DeepSets MLP ...]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvol.data import load_panel  # noqa: E402
from pvol.experiment import Experiment, load_tuned_hp  # noqa: E402
from pvol.models import MODELS  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="main")
    ap.add_argument("--h", type=int, nargs="+", default=[5, 1, 22])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    a = ap.parse_args()
    hp = load_tuned_hp()
    missing = [m for m in a.models if m not in hp]
    if missing:
        raise SystemExit(f"no tuned hyper-parameters for {missing}; run scripts/03_tune.py first")
    ex = Experiment(a.exp, load_panel())
    t0 = time.time()
    for h in a.h:
        for seed in a.seeds:  # seed-major order: a first full set of models is available early
            for m in a.models:
                ex.run_nn(m, h, [seed], hp=hp[m])
    print(f"done in {(time.time() - t0) / 3600:.2f} h")
