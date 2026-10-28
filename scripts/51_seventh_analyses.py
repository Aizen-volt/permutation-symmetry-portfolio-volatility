"""Tests and descriptive analyses of the seventh series (W), decision rules fixed before running.

W1: DeepSets-RAW, DeepSetsP-RAW, PENet-RAW, SetTransformer-RAW, SetGRU-RAW vs Linear-RAW (h = 5, in-range), DM-HLN,
    Holm over the five; MLP-RAW and GRU-RAW vs Linear-RAW descriptively.
W2: SPA test of Hansen (2005) for every h: each of the 14 classical models as the benchmark, the five symmetric networks
    as the alternatives (in-range daily losses, studentized, stationary bootstrap).
W3: H1, S1b and H2 of the main design with the fixed-smoothing (EWC) Diebold-Mariano test instead of Newey-West.
W5: quantiles of the ratio of forecast to realized variance (the source of large squared errors, h = 22); the median of
    the yearly differences of V4 is added in scripts/46_supplement_tables.py.
Writes results*/main/tables/hypotheses_seventh.csv, spa.csv, fy_quantiles.csv, ewc_size.csv.

Usage: [PVOL_UNIVERSE=stocks] python scripts/51_seventh_analyses.py
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from arch.bootstrap import SPA  # noqa: E402

from pvol.config import HORIZONS, RESULTS  # noqa: E402
from pvol.evaluate import CLASSICAL, daily_loss, dm_test, dm_test_ewc, group_index, load_run, qlike  # noqa: E402

_spec = importlib.util.spec_from_file_location("hyp", Path(__file__).with_name("15_hypotheses.py"))
hyp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hyp)

T = RESULTS / "main" / "tables"
W1_NETS = ["DeepSets", "DeepSetsP", "PENet", "SetTransformer", "SetGRU"]
SPA_REPS = 5000


def ewc_compare(La, Lb, cols):
    da, db = daily_loss(La, cols), daily_loss(Lb, cols)
    stat, p = dm_test_ewc(da - db)
    return dict(loss_a=da.mean(), loss_b=db.mean(), mean_diff=(da - db).mean(), dm=stat, p=p)


if __name__ == "__main__":
    rows, spa_rows = [], []
    for h in HORIZONS:
        R = load_run("main", h)
        y, F = R["y"], R["F"]
        cols = hyp.cols_of(group_index(R["ports"]), "in-range")
        L = {m: qlike(y, F[m]) for m in F}
        if h == 5:  # W1
            fam = []
            for a in W1_NETS:
                res, _ = hyp.compare(L[a + "-RAW"], L["Linear-RAW"], cols, h)
                fam.append(dict(hypothesis="W1", h=h, block="in-range", model=a + "-RAW", benchmark="Linear-RAW", **res))
            rows += hyp.family(fam)
            for a in ("MLP", "GRU"):
                res, _ = hyp.compare(L[a + "-RAW"], L["Linear-RAW"], cols, h)
                rows.append(dict(hypothesis="W1 (descriptive)", h=h, block="in-range", model=a + "-RAW",
                                 benchmark="Linear-RAW", **res, p_holm=np.nan, verdict="--"))
        # W3: EWC versions of H1, S1b and H2
        classical = [m for m in CLASSICAL if m in L]
        bench = min(classical, key=lambda m: np.nanmean(L[m][:, cols]))
        for name, pairs in (("H1", [(a, hyp.COUNTERPART[a]) for a in hyp.SYMMETRIC]),
                            ("S1b", [(a, "Linear") for a in hyp.SYMMETRIC]),
                            ("H2", [(a, bench) for a in hyp.SYMMETRIC])):
            fam = [dict(hypothesis=f"W3-{name}", h=h, block="in-range", model=a, benchmark=b,
                        **ewc_compare(L[a], L[b], cols)) for a, b in pairs]
            rows += hyp.family(fam)
        # W2: SPA, every classical model as the benchmark against the five symmetric networks
        nets = np.column_stack([daily_loss(L[a], cols) for a in hyp.SYMMETRIC])
        block = max(5, int(round(len(nets) ** (1 / 3))))
        for b in classical:
            spa = SPA(daily_loss(L[b], cols), nets, reps=SPA_REPS, block_size=block, bootstrap="stationary",
                      studentize=True, seed=0)
            spa.compute()
            spa_rows.append(dict(h=h, benchmark=b, benchmark_loss=daily_loss(L[b], cols).mean(),
                                 p_consistent=spa.pvalues["consistent"], p_lower=spa.pvalues["lower"],
                                 p_upper=spa.pvalues["upper"]))
        if h == 22:  # W5: ratio of forecast to realized variance, the source of large squared errors
            fy = []
            for m in ["EWMA", "HAR", "DCC", "Linear", "MLP", "DeepSets", "PENet", "SetTransformer", "EQCov"]:
                r = (F[m][:, cols] / np.maximum(y[:, cols], 1e-12)).ravel()
                fy.append(dict(h=h, model=m, **{f"q{q:g}".replace(".", "_"): np.quantile(r, q / 100) for q in (1, 50, 99, 99.9)},
                               max=r.max(), share_above_10=np.mean(r > 10)))
            pd.DataFrame(fy).to_csv(T / "fy_quantiles.csv", index=False)
    # size of the two tests at 5% for MA(h-1) differentials of the length of the test period (300 replications)
    g = np.random.default_rng(1)
    size = []
    for h in HORIZONS:
        n = len(np.load(RESULTS / "main" / f"h{h}" / "origins.npy"))
        rej = np.array([[dm_test(d, h)[1] < 0.05, dm_test_ewc(d)[1] < 0.05]
                        for d in (np.convolve(g.standard_normal(n + h - 1), np.ones(h), "valid") for _ in range(300))])
        size.append(dict(h=h, T=n, size_dm=rej[:, 0].mean(), size_ewc=rej[:, 1].mean()))
    pd.DataFrame(size).to_csv(T / "ewc_size.csv", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(T / "hypotheses_seventh.csv", index=False)
    spa = pd.DataFrame(spa_rows)
    spa.to_csv(T / "spa.csv", index=False)
    for (hy, h), g in out.groupby(["hypothesis", "h"]):
        print(hy, h, (g.verdict == "more accurate").sum(), "for /", (g.verdict == "less accurate").sum(), "against /",
              len(g))
    print(spa.groupby("h").agg(max_p=("p_consistent", "max"), rejected=("p_consistent", lambda p: int((p < 0.05).sum())),
                               n=("p_consistent", "size")))
