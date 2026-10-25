"""Reproducibility check: retrain selected networks (seed 0, test year 2020, h = 5) twice and compare the forecasts with
each other and with the stored forecasts of the main experiment.

GPU kernels (scatter/atomic additions, cuDNN GRU) are not bit-wise deterministic, so identical seeds can give slightly
different weights; the check reports how large the differences are in the log-variance forecasts and in QLIKE.
Output: results/main/tables/determinism.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.data import load_panel  # noqa: E402
from pvol.dataset import make_split, origins_in_year  # noqa: E402
from pvol.experiment import Experiment, load_tuned_hp  # noqa: E402
from pvol.train import predict_portfolios, train_model  # noqa: E402


def qlike_np(s, y):
    return np.nanmean(y * np.exp(-s) + s - np.log(y) - 1.0)


if __name__ == "__main__":
    h, year, seed = 5, 2020, 0
    hp = load_tuned_hp()
    ex = Experiment("main", load_panel())
    d = ex.hdir(h)
    t_idx = origins_in_year(ex.panel.dates, year, h)
    y = np.load(d / "targets.npy")[: len(t_idx)]
    rows = []
    for m in sys.argv[1:] or ["MLP", "DeepSets", "SetGRU"]:
        stored = np.load(d / "preds" / f"{m}__s{seed}.npy")[: len(t_idx)]
        runs = []
        for _ in range(2):
            split = make_split(ex.panel.dates, ex.panel.first_valid, year, h)
            model, info = train_model(m, ex.panel, split, h, seed, hp=hp[m])
            runs.append(predict_portfolios(model, m, ex.panel, ex.portfolios, t_idx))
            print(f"{m}: val {info['best_val']:.5f} epochs {info['epochs']}", flush=True)
        ok = ~np.isnan(stored)
        rows.append(dict(model=m, max_abs_rerun=float(np.max(np.abs(runs[0] - runs[1])[ok])),
                         max_abs_stored=float(np.max(np.abs(runs[0] - stored)[ok])),
                         mean_abs_stored=float(np.mean(np.abs(runs[0] - stored)[ok])),
                         qlike_stored=qlike_np(stored[ok], y[ok]), qlike_rerun=qlike_np(runs[0][ok], y[ok])))
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(d.parent / "tables" / "determinism.csv", index=False)
    print(out.to_string())
