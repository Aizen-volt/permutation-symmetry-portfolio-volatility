"""Evaluation tables for an experiment: losses by portfolio group, DM tests, MCS, permutation consistency, VaR, per-year.

Usage: python scripts/06_evaluate.py main [seeds...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import HORIZONS, RESULTS  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.evaluate import (CLASSICAL, NEURAL, daily_loss, dm_test, group_index, load_run, mcs,  # noqa: E402
                           mincer_zarnowitz, mse, qlike, var_backtest)
from pvol.portfolios import load  # noqa: E402

GROUP_SETS = {
    "main": {
        "in-range": ["N5", "N10", "N20", "N10-conc", "N10-EW"],
        "larger-N": ["N30", "N40", "EW-all"],
        "long-short": ["N10-LS"],
        "structured": ["60/40", "AllWeather", "GlobalMulti"],
    },
    "heldout": {"unseen-assets": ["H5", "H10", "H10-EW"]},
}
PAIRS = [("DeepSets", "MLP"), ("PENet", "MLP"), ("SetTransformer", "MLP"), ("EQCov", "MLP"),
         ("SetGRU", "GRU"), ("DeepSets", "GRU"), ("MLP-sorted", "MLP"), ("DeepSets", "PortMLP"),
         ("SetTransformer", "PortMLP"), ("EQCov", "DCC")]


def order(models):
    return [m for m in CLASSICAL + NEURAL if m in models]


def evaluate_h(exp, h, seeds, out):
    run = load_run(exp, h, seeds)
    y, F, ports = run["y"], run["F"], run["ports"]
    models = order(F)
    gidx = group_index(ports)
    Lq = {m: qlike(y, F[m]) for m in models}
    Lm = {m: mse(y, F[m]) for m in models}
    rows = []
    # block sets present in this universe (e.g. no structured allocations for the stock universe)
    group_sets = {b: gs for b, gs in GROUP_SETS[exp].items() if any(g in gidx for g in gs)}
    # losses per individual group and per group set
    blocks = {**{g: [g] for g in gidx}, **group_sets}
    for b, groups in blocks.items():
        cols = np.concatenate([gidx[g] for g in groups if g in gidx])
        for m in models:
            q = Lq[m][:, cols]
            if np.isnan(q).all():
                continue
            rows.append(dict(h=h, block=b, model=m, qlike=np.nanmean(q), mse=np.nanmean(Lm[m][:, cols]),
                             coverage=np.isfinite(q).mean()))
    tab = pd.DataFrame(rows)
    ref = tab[tab.model == "EWMA"].set_index("block")
    tab["qlike_rel_ewma"] = tab.qlike / tab.block.map(ref.qlike)
    tab["mse_rel_ewma"] = tab.mse / tab.block.map(ref.mse)
    tab.to_csv(out / f"losses_h{h}.csv", index=False)

    # DM tests on daily cross-sectional average QLIKE, per group set
    dm_rows, pf_rows = [], []
    for b, groups in group_sets.items():
        cols = np.concatenate([gidx[g] for g in groups if g in gidx])
        daily = {m: daily_loss(Lq[m], cols) for m in models if np.isfinite(Lq[m][:, cols]).all()}
        best_classical = min((m for m in CLASSICAL if m in daily), key=lambda m: daily[m].mean())
        pairs = PAIRS + [(m, best_classical) for m in NEURAL if m in daily]
        for a, c in pairs:
            if a in daily and c in daily:
                stat, p = dm_test(daily[a] - daily[c], h)
                dm_rows.append(dict(h=h, block=b, model=a, benchmark=c, mean_diff=np.mean(daily[a] - daily[c]),
                                    dm_stat=stat, p_value=p))
        # portfolio-level comparisons for the matched pairs: win rate and share of significant DM tests
        for a, c in PAIRS:
            if a in daily and c in daily:
                D = Lq[a][:, cols] - Lq[c][:, cols]
                res = np.array([dm_test(D[:, j], h) for j in range(D.shape[1])])
                pf_rows.append(dict(h=h, block=b, model=a, benchmark=c, n_portfolios=D.shape[1],
                                    win_rate=np.mean(D.mean(0) < 0),
                                    share_sig_better=np.mean((res[:, 1] < 0.05) & (res[:, 0] < 0)),
                                    share_sig_worse=np.mean((res[:, 1] < 0.05) & (res[:, 0] > 0))))
        # Model Confidence Set
        ldf = pd.DataFrame(daily)
        pv = mcs(ldf, size=0.10)
        for m, v in pv.items():
            dm_rows.append(dict(h=h, block=b, model=m, benchmark="MCS", mean_diff=np.nan, dm_stat=np.nan, p_value=v))
    pd.DataFrame(dm_rows).to_csv(out / f"tests_h{h}.csv", index=False)
    pd.DataFrame(pf_rows).to_csv(out / f"portfolio_tests_h{h}.csv", index=False)

    # Mincer-Zarnowitz (log scale, in-range groups / first group set)
    b0, groups = next(iter(group_sets.items()))
    cols = np.concatenate([gidx[g] for g in groups if g in gidx])
    mz = []
    for m in models:
        f = np.log(F[m][:, cols]).ravel()
        yy = np.log(np.maximum(y[:, cols], 1e-12)).ravel()
        if np.isfinite(f).all():
            a, b, r2 = mincer_zarnowitz(yy, f)
            mz.append(dict(h=h, block=b0, model=m, alpha=a, beta=b, r2=r2))
    pd.DataFrame(mz).to_csv(out / f"mz_h{h}.csv", index=False)

    # per-year QLIKE (first group set)
    panel_dates = load_panel()["dates"]
    years = pd.Series(panel_dates[run["origins"]]).str[:4].to_numpy()
    yr = []
    for m in models:
        q = Lq[m][:, cols]
        if np.isfinite(q).all():
            for Y in np.unique(years):
                yr.append(dict(h=h, model=m, year=int(Y), qlike=q[years == Y].mean()))
    pd.DataFrame(yr).to_csv(out / f"yearly_h{h}.csv", index=False)

    # seed dispersion of NN losses (first group set)
    sd = []
    for m, runs in ((m, r) for m, r in run["F_seed"].items() if m in Lq):  # skip tagged variants (-K2, -P50, ...)
        vals = [np.nanmean(qlike(y[:, cols], f[:, cols])) for f in runs.values()]
        sd.append(dict(h=h, model=m, n_seeds=len(vals), mean=np.mean(vals), sd=np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
                       ensemble=np.nanmean(Lq[m][:, cols])))
    pd.DataFrame(sd).to_csv(out / f"seeds_h{h}.csv", index=False)

    # permutation consistency
    perm_path = RESULTS / exp / "perm_portfolios.json"
    pp = load(perm_path) if perm_path.exists() else []
    src = np.array([p["src"] for p in pp], dtype=int)
    perm_k = np.array([p["perm"] for p in pp], dtype=int)
    d = RESULTS / exp / f"h{h}" / "preds"
    pc = []
    for m in NEURAL:
        files = sorted(d.glob(f"{m}__s*__perm.npy"))
        if not files or not len(pp):
            continue
        stats_m = []
        for f in files:
            s = np.load(f).astype(np.float64)  # (T, n_ports * n_perm)
            n_perm = perm_k.max() + 1
            S = np.stack([s[:, perm_k == k] for k in range(n_perm)], 0)  # (n_perm, T, n_ports)
            spread = S.max(0) - S.min(0)  # range of log forecasts across orderings
            srcs = src[perm_k == 0]
            yq = y[:, srcs]
            ql = np.stack([qlike(yq, np.exp(S[k])).mean() for k in range(n_perm)])
            stats_m.append((np.mean(np.std(S, 0)), np.median(spread), np.percentile(spread, 99), ql.max() - ql.min()))
        a = np.array(stats_m).mean(0)
        pc.append(dict(h=h, model=m, mean_sd_logvar=a[0], median_range_logvar=a[1], p99_range_logvar=a[2],
                       qlike_range_over_orderings=a[3]))
    pd.DataFrame(pc).to_csv(out / f"perm_h{h}.csv", index=False)

    # VaR backtest (h = 1)
    if h == 1:
        p = load_panel()
        R = p["R"].astype(np.float64)
        r_next = np.column_stack([R[run["origins"] + 1][:, q["ids"]] @ np.asarray(q["weights"]) for q in ports])
        vr = []
        for alpha in (0.01, 0.05):
            for m in models:
                res = var_backtest(r_next[:, cols], F[m][:, cols], alpha)
                vr.append(dict(alpha=alpha, model=m, **res))
        pd.DataFrame(vr).to_csv(out / "var_h1.csv", index=False)
    return tab


if __name__ == "__main__":
    exp = sys.argv[1] if len(sys.argv) > 1 else "main"
    seeds = [int(s) for s in sys.argv[2:]] or [0, 1, 2]
    out = RESULTS / exp / "tables"
    out.mkdir(parents=True, exist_ok=True)
    for h in HORIZONS:
        if not (RESULTS / exp / f"h{h}" / "targets.npy").exists():
            continue
        tab = evaluate_h(exp, h, seeds, out)
        piv = tab[tab.block.isin(GROUP_SETS[exp])].pivot(index="model", columns="block", values="qlike_rel_ewma")
        print(f"\n=== {exp} h={h}: QLIKE relative to EWMA ===")
        print(piv.reindex(order(piv.index)).round(3).to_string())
    (out / "meta.json").write_text(json.dumps(dict(seeds=seeds)))
