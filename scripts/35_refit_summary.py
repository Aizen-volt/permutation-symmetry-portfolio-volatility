"""Alternative training protocols: in-range QLIKE relative to EWMA of every network under the main protocol, protocol B
(tag suffix -P2) and the refit protocol (-RF), and S1b (symmetric network against Linear) under each protocol.

Writes results*/main/tables/protocols.csv. Usage: [PVOL_UNIVERSE=stocks] python scripts/35_refit_summary.py
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvol.config import RESULTS  # noqa: E402
from pvol.evaluate import daily_loss, dm_test, group_index, load_run, qlike  # noqa: E402

hyp = importlib.import_module("15_hypotheses")
NETS = ["Linear", "MLP", "GRU", "DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov"]
PROTOCOLS = {"main": "", "B": "-P2", "refit": "-RF"}

if __name__ == "__main__":
    rows, tests = [], []
    for h in (1, 5, 22):
        run = load_run("main", h)
        cols = np.concatenate([group_index(run["ports"])[g] for g in hyp.BLOCKS["in-range"]])
        base = daily_loss(qlike(run["y"], run["F"]["EWMA"]), cols).mean()
        for prot, suf in PROTOCOLS.items():
            if not all(m + suf in run["F"] for m in NETS):
                continue
            L = {m: daily_loss(qlike(run["y"], run["F"][m + suf]), cols) for m in NETS}
            rows += [dict(h=h, protocol=prot, model=m, rel_ewma=L[m].mean() / base) for m in NETS]
            fam = []
            for m in hyp.SYMMETRIC:
                d = L[m] - L["Linear"]
                stat, p = dm_test(d, h)
                fam.append(dict(h=h, protocol=prot, model=m, benchmark="Linear", mean_diff=d.mean() / base, p=p))
            tests += hyp.family(fam)
    out = RESULTS / "main/tables/protocols.csv"
    df = pd.DataFrame(rows).pivot_table(index=["h", "model"], columns="protocol", values="rel_ewma").reset_index()
    df.to_csv(out, index=False)
    pd.DataFrame(tests).to_csv(RESULTS / "main/tables/protocols_s1b.csv", index=False)
    print(df.round(3).to_string(index=False))
    print(pd.DataFrame(tests).groupby(["h", "protocol"]).verdict.value_counts().to_string())
