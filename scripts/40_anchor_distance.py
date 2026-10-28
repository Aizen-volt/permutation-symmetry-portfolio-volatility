"""How far the learned models move away from their EWMA anchor: for every learned model and horizon,
the standard deviation and the mean absolute value of log(forecast / EWMA forecast) over the in-range portfolios and
the test origins, for the seed ensemble and as the average over individual seeds. Zero means the model reproduces
RiskMetrics. For EQ-Cov, which is not a residual model, the same quantity measures its distance from EWMA.
Writes results*/main/tables/anchor_distance.csv for the current universe.

Usage: [PVOL_UNIVERSE=stocks] python scripts/40_anchor_distance.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import RESULTS  # noqa: E402
from pvol.evaluate import group_index, load_run  # noqa: E402

MODELS = ["Linear", "MLP", "MLP-sorted", "GRU", "DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov"]
IN_RANGE = ["N5", "N10", "N20", "N10-conc", "N10-EW"]

rows = []
for h in (1, 5, 22):
    run = load_run("main", h)
    gi = group_index(run["ports"])
    cols = np.concatenate([gi[g] for g in IN_RANGE])
    base = np.log(run["F"]["EWMA"][:, cols])
    for m in MODELS:
        if m not in run["F"]:
            continue
        g = np.log(run["F"][m][:, cols]) - base
        gs = [np.log(f[:, cols]) - base for f in run["F_seed"][m].values()]
        gl = np.log(run["F"]["Linear"][:, cols]) - base
        ok = np.isfinite(g) & np.isfinite(gl)
        rows.append(dict(h=h, model=m, mean_ensemble=np.nanmean(g), sd_ensemble=np.nanstd(g),
                         mad_ensemble=np.nanmean(np.abs(g)), sd_seed=np.mean([np.nanstd(x) for x in gs]),
                         mad_seed=np.mean([np.nanmean(np.abs(x)) for x in gs]),
                         corr_linear=np.corrcoef(g[ok], gl[ok])[0, 1], sd_minus_linear=np.std(g[ok] - gl[ok])))
out = pd.DataFrame(rows)
out.to_csv(RESULTS / "main" / "tables" / "anchor_distance.csv", index=False)
print(out.set_index(["h", "model"])[["sd_ensemble", "corr_linear", "sd_minus_linear"]].round(2))
