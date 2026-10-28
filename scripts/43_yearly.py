"""Sixth series V4: accuracy by test year, from the stored forecasts (no training).

For each universe and horizon: in-range QLIKE relative to EWMA in every test year for the networks, Linear, HEAVY and
the best classical model of the horizon (yearly_levels.csv), and for the pairs PE-Net - MLP, Set-GRU - GRU and
Linear - HEAVY the yearly mean of the daily loss differential with a DM test per year (yearly_pairs.csv, descriptive,
no correction). Rule: the sign of the yearly mean is the same in at least 5 of the 7 years -> stable.

Usage: python scripts/43_yearly.py   (both universes)
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pvol.data as pdata  # noqa: E402
import pvol.evaluate as ev  # noqa: E402
from pvol.config import HORIZONS  # noqa: E402

_spec = importlib.util.spec_from_file_location("hyp", Path(__file__).with_name("15_hypotheses.py"))
hyp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hyp)

ROOT = Path(__file__).resolve().parents[1]
NETS = ["MLP", "MLP-sorted", "GRU", "DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov"]
PAIRS = [("PENet", "MLP"), ("SetGRU", "GRU"), ("Linear", "HEAVY")]
UNIVERSES = [("ETF", "", "processed"), ("Stocks", "_stocks", "processed_stocks")]

if __name__ == "__main__":
    for u, sfx, proc in UNIVERSES:
        ev.RESULTS = ROOT / f"results{sfx}"
        pdata.PROCESSED = ROOT / "data" / proc
        dates = pd.to_datetime(pdata.load_panel()["dates"])
        levels, pairs = [], []
        for h in HORIZONS:
            R = ev.load_run("main", h)
            ci = hyp.cols_of(ev.group_index(R["ports"]), "in-range")
            L = {m: ev.daily_loss(ev.qlike(R["y"], f), ci) for m, f in R["F"].items()}
            best = min((m for m in ev.CLASSICAL if m in L), key=lambda m: L[m].mean())
            year = dates[R["origins"]].year.to_numpy()
            for y in np.unique(year):
                k = year == y
                e = L["EWMA"][k].mean()
                for m in NETS + ["Linear", "HEAVY", best]:
                    levels.append(dict(h=h, year=y, model=m, best_classical=m == best, qlike_rel_ewma=L[m][k].mean() / e))
                for a, b in PAIRS:
                    d = L[a][k] - L[b][k]
                    stat, p = ev.dm_test(d, h)
                    pairs.append(dict(h=h, year=y, model=a, benchmark=b, mean_diff_pp=100 * d.mean() / e, dm=stat, p=p))
        out = ev.RESULTS / "main" / "tables"
        lv = pd.DataFrame(levels).drop_duplicates(["h", "year", "model"])
        lv.to_csv(out / "yearly_levels.csv", index=False)
        P = pd.DataFrame(pairs)
        P.to_csv(out / "yearly_pairs.csv", index=False)
        print(f"== {u}")
        print(P.round(3).to_string(index=False))
        S = (P.assign(neg=P.mean_diff_pp < 0).groupby(["h", "model", "benchmark"])
             .agg(years_model_better=("neg", "sum"), years=("neg", "size"), min_pp=("mean_diff_pp", "min"),
                  max_pp=("mean_diff_pp", "max")).reset_index())
        S["verdict"] = np.where(np.maximum(S.years_model_better, S.years - S.years_model_better) >= 5, "stable",
                                "episodic")
        S.to_csv(out / "yearly_summary.csv", index=False)
        print(S.round(2).to_string(index=False))
