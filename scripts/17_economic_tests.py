"""Economic counterparts of H2 (and H1): minimum-variance allocation and volatility targeting.

(b) Minimum variance: for every rebalancing origin, average realised variance over the 100 subsets; DM test (h=5) of the
    difference between two models. Lower realised variance = better risk model.
(a) Volatility targeting: monthly squared log deviation of realised volatility from the target, averaged over the
    in-range portfolios; DM test (lag 1) on monthly differences. Lower = better.
Families and Holm correction as in scripts/15_hypotheses.py.
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import RESULTS  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.evaluate import dm_test, group_index, load_run  # noqa: E402

_spec = importlib.util.spec_from_file_location("hyp", Path(__file__).with_name("15_hypotheses.py"))
hyp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hyp)
_spec = importlib.util.spec_from_file_location("vt", Path(__file__).with_name("12_vol_targeting.py"))
vt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vt)

COV_MODELS = ["EWMA", "DCC", "DCCNL", "LW", "EWMANLS"]


def minvar_tests():
    df = pd.read_csv(RESULTS / "main" / "tables" / "minvar_h5.csv")
    by = df.groupby(["model", "origin"]).realised_var.mean().unstack(0)  # origins x models
    bench = min((m for m in COV_MODELS if m in by), key=lambda m: by[m].mean())
    rows = []
    for fam, pairs in (("H2-minvar", [(a, bench) for a in hyp.SYMMETRIC]),
                       ("H1-minvar", [(a, hyp.COUNTERPART[a]) for a in hyp.SYMMETRIC])):
        fr = []
        for a, b in pairs:
            d = (by[a] - by[b]).to_numpy()
            stat, p = dm_test(d, 5)
            fr.append(dict(hypothesis=fam, model=a, benchmark=b, loss_a=np.sqrt(by[a].mean() * 252),
                           loss_b=np.sqrt(by[b].mean() * 252), mean_diff=d.mean(), dm=stat, p=p))
        rows += hyp.family(fr)
    return rows


def voltarget_tests(h):
    p = load_panel()
    R = p["R"].astype(np.float64)
    run = load_run("main", h)
    gidx = group_index(run["ports"])
    cols = np.concatenate([gidx[g] for g in hyp.BLOCKS["in-range"] if g in gidx])
    r = np.column_stack([R[:, q["ids"]] @ np.asarray(q["weights"]) for q in (run["ports"][j] for j in cols)])
    month = 21
    sq = {}
    for m, f in run["F"].items():
        if m.endswith(("-K2", "-K4", "-P2")) or not np.isfinite(f[:, cols]).all():
            continue
        mr = vt.managed_returns(r, f[:, cols] * 252, run["origins"], h)
        n = (len(mr) // month) * month
        mv = np.sqrt((mr[:n].reshape(-1, month, mr.shape[1]) ** 2).mean(1) * 252)
        sq[m] = (np.log(mv) ** 2).mean(1)  # monthly, averaged over portfolios
    classical = [m for m in hyp.CLASSICAL if m in sq]
    bench = min(classical, key=lambda m: sq[m].mean())
    rows = []
    for fam, pairs in (("H2-voltarget", [(a, bench) for a in hyp.SYMMETRIC]),
                       ("H1-voltarget", [(a, hyp.COUNTERPART[a]) for a in hyp.SYMMETRIC])):
        fr = []
        for a, b in pairs:
            d = sq[a] - sq[b]
            stat, pv = dm_test(d, 1)
            fr.append(dict(hypothesis=fam, h=h, model=a, benchmark=b, loss_a=np.sqrt(sq[a].mean()),
                           loss_b=np.sqrt(sq[b].mean()), mean_diff=d.mean(), dm=stat, p=pv))
        rows += hyp.family(fr)
    return rows


if __name__ == "__main__":
    rows = minvar_tests()
    for h in (1, 5, 22):
        rows += voltarget_tests(h)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "main" / "tables" / "economic_tests.csv", index=False)
    pd.set_option("display.width", 220)
    print(df[["hypothesis", "h", "model", "benchmark", "loss_a", "loss_b", "dm", "p", "p_holm", "verdict"]].round(4).to_string(index=False))
