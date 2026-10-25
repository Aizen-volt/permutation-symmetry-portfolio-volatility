"""Fourth series: robustness analyses from saved forecasts (descriptive, no decision rules).

- losses_variants.csv: in-range loss relative to EWMA under QLIKE, Patton's b = -1 loss and MSE, and QLIKE excluding 2020;
- hypotheses_excl_year.csv: H1 and H2 (decision rule of the paper) on the test origins outside 2020 and outside 2026;
- gw.csv: conditional Giacomini-White test (instruments 1 and the lagged differential) for H1 and H2, Holm per family;
- seed_signs.csv: share of (seed of the symmetric network, seed of the counterpart) pairs in which the symmetric network
  has the lower in-range mean QLIKE (stock universe at h = 1: seeds 0-9, seeds 3-9 from the fourth series S2);
- var_t_fhs.csv: one-day value-at-risk with Student-t quantiles (degrees of freedom from the kurtosis of the last 250
  standardised returns) and filtered historical simulation (empirical quantile of the last 250 standardised returns);
  evaluated from the 251st test origin on.

Usage: [PVOL_UNIVERSE=stocks] python scripts/28_fourth_analyses.py
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from pvol.config import HORIZONS, RESULTS  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.evaluate import CLASSICAL, daily_loss, dm_test, group_index, load_run, mse, patton_bm1, qlike  # noqa: E402

_spec = importlib.util.spec_from_file_location("hyp", Path(__file__).with_name("15_hypotheses.py"))
hyp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hyp)
SYM, CP = hyp.SYMMETRIC, hyp.COUNTERPART
EXT = ["HAR-WLS", "LogHAR-X", "HEAVY", "PCA-F"]
NETS = ["MLP", "GRU"] + SYM
OUT = RESULTS / "main" / "tables"


def gw_test(d, h):
    """Conditional Giacomini-White (2006) test of E[d_t | d_{t-1}] = 0: Wald statistic on Z_t = (d_t, d_{t-1} d_t) with a
    Newey-West covariance (same lag rule as the DM test); chi-square with 2 degrees of freedom."""
    d = np.asarray(d, dtype=np.float64)
    Z = np.column_stack([d[1:], d[:-1] * d[1:]])
    T = len(Z)
    Zc = Z - Z.mean(0)
    lag = max(h, int(np.floor(4 * (T / 100) ** (2 / 9))))
    S = Zc.T @ Zc / T
    for k in range(1, lag + 1):
        G = Zc[k:].T @ Zc[:-k] / T
        S += (1 - k / (lag + 1)) * (G + G.T)
    zb = Z.mean(0)
    stat = T * zb @ np.linalg.solve(S, zb)
    return stat, stats.chi2.sf(stat, 2)


def ensemble(F_seed, tags, seeds=None):
    runs = [f for t in tags for s, f in F_seed.get(t, {}).items() if seeds is None or s in seeds]
    return np.mean(np.stack(runs), 0) if runs else None


if __name__ == "__main__":
    dates = pd.to_datetime(load_panel()["dates"])
    lv, h12, gw, signs, var = [], [], [], [], []
    for h in HORIZONS:
        R = load_run("main", h, seeds=None)
        y, Fs = R["y"], R["F_seed"]
        # the main analysis uses seeds 0-2; seeds 3-9 (fourth series S2) only enter seed_signs
        F = {m: f for m, f in R["F"].items() if not m.endswith(("-S10",))}
        for m in NETS:
            if m in Fs:
                F[m] = ensemble(Fs, [m], seeds=(0, 1, 2))
        ci = hyp.cols_of(group_index(R["ports"]), "in-range")
        ex = ~(dates[R["origins"]].year == 2020)
        models = [m for m in dict.fromkeys(CLASSICAL + EXT + ["Linear", "GBM"] + NETS) if m in F]
        for name, fn, rows in (("QLIKE", qlike, slice(None)), ("Patton b=-1", patton_bm1, slice(None)), ("MSE", mse, slice(None)),
                               ("QLIKE excl. 2020", qlike, ex)):
            e = np.nanmean(fn(y[rows], F["EWMA"][rows])[:, ci])
            for m in models:
                lv.append(dict(loss=name, h=h, model=m, rel_ewma=np.nanmean(fn(y[rows], F[m][rows])[:, ci]) / e))
        L = {m: qlike(y, F[m]) for m in models}
        # H1 / H2 outside 2020 (the crisis year) and outside 2026 (the incomplete year); best classical chosen ex post
        # on that sample
        for yr in (2020, 2026):
            keep = ~(dates[R["origins"]].year == yr)
            for fam, pairs in (("H1", [(a, CP[a]) for a in SYM]),
                               ("H2", [(a, min([m for m in CLASSICAL if m in L],
                                               key=lambda m: np.nanmean(L[m][keep][:, ci]))) for a in SYM])):
                rows = []
                for a, b in pairs:
                    d = daily_loss(L[a], ci)[keep] - daily_loss(L[b], ci)[keep]
                    st, p = dm_test(d, h)
                    rows.append(dict(hypothesis=fam, h=h, block=f"in-range, excl. {yr}", model=a, benchmark=b,
                                     mean_diff=d.mean(), dm=st, p=p))
                h12 += hyp.family(rows)
        for fam, pairs in (("H1", [(a, CP[a]) for a in SYM]), ("H2", None)):
            grow = []
            for a, b in pairs if fam == "H1" else [(a, min([m for m in CLASSICAL if m in L],
                                                          key=lambda m: np.nanmean(L[m][:, ci]))) for a in SYM]:
                d = daily_loss(L[a], ci) - daily_loss(L[b], ci)
                st, p = gw_test(d, h)
                grow.append(dict(hypothesis=fam, h=h, model=a, benchmark=b, mean_diff=d.mean(), gw=st, p=p))
            gw += hyp.family(grow)
        # seed-pair sign agreement
        for a in SYM:
            b = CP[a]
            sa = {**Fs.get(a, {}), **Fs.get(a + "-S10", {})}
            sb = {**Fs.get(b, {}), **Fs.get(b + "-S10", {})}
            la = {s: np.nanmean(qlike(y, f)[:, ci]) for s, f in sa.items()}
            lb = {s: np.nanmean(qlike(y, f)[:, ci]) for s, f in sb.items()}
            wins = [la[i] < lb[j] for i in la for j in lb]
            if wins:
                signs.append(dict(h=h, model=a, counterpart=b, seeds_a=len(la), seeds_b=len(lb), share_lower=np.mean(wins)))
        # value-at-risk at h = 1 with t and filtered historical simulation quantiles
        if h == 1:
            Rr = load_panel()["R"].astype(np.float64)
            rn = np.column_stack([Rr[R["origins"] + 1][:, q["ids"]] @ np.asarray(q["weights"]) for q in R["ports"]])[:, ci]
            for m in ["EWMA", "HAR", "GARCH", "GJR", "DCC"] + [x for x in EXT if x in F] + NETS:
                sd = np.sqrt(F[m][:, ci])
                z = rn / sd
                W = 250
                for alpha in (0.01, 0.05):
                    hit_t, hit_f = [], []
                    for t in range(W, len(z)):
                        past = z[t - W : t]
                        q_f = np.quantile(past, alpha, axis=0)
                        k = np.clip(stats.kurtosis(past, axis=0), 0.2, None)
                        nu = np.clip(4 + 6 / k, 4.1, 30)
                        q_t = stats.t.ppf(alpha, nu) * np.sqrt((nu - 2) / nu)
                        hit_t.append(z[t] < q_t)
                        hit_f.append(z[t] < q_f)
                    for kind, hits in (("t", np.array(hit_t)), ("FHS", np.array(hit_f))):
                        n, x = hits.shape[0], hits.sum(0)
                        pi = np.clip(x / n, 1e-9, 1 - 1e-9)
                        lr = -2 * (x * np.log(alpha) + (n - x) * np.log(1 - alpha) - x * np.log(pi) - (n - x) * np.log(1 - pi))
                        var.append(dict(model=m, alpha=alpha, kind=kind, hit_rate=hits.mean(),
                                        kupiec_reject_share=np.mean(stats.chi2.sf(lr, 1) < 0.05)))
        print(f"h={h} done", flush=True)
    for name, rows in (("losses_variants", lv), ("hypotheses_excl_year", h12), ("gw", gw), ("seed_signs", signs),
                       ("var_t_fhs", var)):
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    print("done")
