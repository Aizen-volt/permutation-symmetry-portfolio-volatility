"""Precision of the comparisons and multiplicity across all families (from the hypothesis tables, no new forecasts).

1. Precision: for every comparison of the scoreboard families, the 95% confidence interval of the mean daily QLIKE
   differential (HAC standard error of the Diebold-Mariano test, SE = mean / DM statistic, Student t with T-1 degrees
   of freedom) and the smallest difference the test detects with 80% power at the 5% level without the Holm correction,
   (t_0.975 + z_0.8) SE; both in percentage points of the in-range EWMA loss, as the effects of the scoreboard.
2. Multiplicity: the Holm correction over all comparisons of all families of both universes at once (instead of within
   each family) and the Benjamini-Hochberg false discovery rate over the same set; which significant verdicts survive.

Output: results/main/tables/precision.csv, results/main/tables/global_multiplicity.csv
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_spec = importlib.util.spec_from_file_location("sb", Path(__file__).with_name("20_scoreboard.py"))
sb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sb)
_spec = importlib.util.spec_from_file_location("hyp", Path(__file__).with_name("15_hypotheses.py"))
hyp = importlib.util.module_from_spec(_spec)
sys.argv = sys.argv[:1]
_spec.loader.exec_module(hyp)

ECONOMIC = sb.E


def family_rows(root, fname, flt):
    f = root / "main" / "tables" / fname
    if not f.exists():
        return None
    d = pd.read_csv(f)
    for k, v in flt.items():
        if k not in d.columns:
            return None
        d = d[d[k] == v]
    return d if len(d) else None


def n_days(root, h):
    return len(np.load(root / "main" / f"h{h}" / "origins.npy"))


def bh(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, p[i] * m / (rank + 1))
        adj[i] = running
    return adj


if __name__ == "__main__":
    prec, allrows = [], []
    for part, rows in (("main", sb.ROWS), ("secondary", sb.ROWS_SUPP)):
        for lbl, fname, flt, *alt in rows:
            if fname is None:
                continue
            roots = [(u, r) for u, r in sb.UNIVERSES]
            if alt:
                roots = [("Stocks", Path(str(sb.BASE) + alt[0]))]
            for u, root in roots:
                d = family_rows(root, fname, flt)
                if d is None:
                    continue
                for r in d.itertuples():
                    allrows.append(dict(part=part, family=lbl, universe=u, model=r.model, benchmark=r.benchmark,
                                        mean_diff=getattr(r, "mean_diff", np.nan), p=r.p, p_holm=r.p_holm,
                                        verdict=r.verdict))
                if fname == ECONOMIC or "dm" not in d.columns:
                    continue
                h = int(d.h.iloc[0]) if "h" in d.columns and d.h.notna().all() else 5
                e = sb.ewma_loss(root, h)
                T = n_days(root, h)
                tq = stats.t.ppf(0.975, T - 1)
                se = np.abs(d.mean_diff.to_numpy() / d.dm.to_numpy())
                eff = 100 * d.mean_diff.to_numpy() / e
                for m, b, ef, s in zip(d.model, d.benchmark, eff, 100 * se / e):
                    prec.append(dict(part=part, family=lbl, universe=u, h=h, model=m, benchmark=b, effect=ef,
                                     ci_lo=ef - tq * s, ci_hi=ef + tq * s, halfwidth=tq * s,
                                     mde80=(tq + stats.norm.ppf(0.8)) * s))
    P = pd.DataFrame(prec)
    out = sb.BASE / "main" / "tables"
    P.to_csv(out / "precision.csv", index=False)
    A = pd.DataFrame(allrows)
    A["p_global_holm"] = hyp.holm(A.p.to_numpy())
    A["p_global_bh"] = bh(A.p.to_numpy())
    A.to_csv(out / "global_multiplicity.csv", index=False)

    pd.set_option("display.width", 250)
    print(P.groupby(["part", "family", "universe"], sort=False)
          .agg(effect_lo=("effect", "min"), effect_hi=("effect", "max"), halfwidth=("halfwidth", "max"),
               mde80=("mde80", "max")).round(2).to_string())
    sig = A[A.verdict != "indistinguishable"]
    print(f"\n{len(A)} comparisons in {A.groupby(['family', 'universe']).ngroups} families; "
          f"{len(sig)} significant within families")
    print(sig.assign(global_holm=sig.p_global_holm < 0.05, global_bh=sig.p_global_bh < 0.05)
          [["part", "family", "universe", "model", "benchmark", "verdict", "p", "p_holm", "p_global_holm",
            "p_global_bh", "global_holm", "global_bh"]].round(4).to_string())
