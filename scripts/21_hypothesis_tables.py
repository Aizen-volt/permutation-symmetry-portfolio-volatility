"""Appendix tables with the test statistics behind every verdict of the scoreboard, and the level table of the second
series (E1-E3), for both universes; built from results*/main/tables/*.csv only.

hyp_accuracy.tex  H1 and H2 at every horizon (in-range block)
hyp_other.tex     H3a-H3c and H4
hyp_econ.tex      minimum variance and volatility targeting
hyp_review*.tex   third series (R1-R4)
hyp_extra.tex     HE1-HE3
hyp_fourth*.tex   fourth series (S1b, S2) and H2 against the ten classical models of the original design
extra_levels.tex  QLIKE relative to EWMA of the networks under the training conditions of E1-E3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import RESULTS  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "paper" / "tables"
BASE = Path(str(RESULTS).replace("_stocks", ""))
UNIVERSES = [("ETF", BASE), ("Stocks", Path(str(BASE) + "_stocks"))]
LAB = {"SetTransformer": "Set Transformer", "DeepSets": "Deep Sets", "EQCov": "EQ-Cov", "PENet": "PE-Net",
       "SetGRU": "Set-GRU", "HARX": "HAR-X", "GJR": "GJR-GARCH", "EWMANLS": "EWMA-NLS", "DCCNL": "DCC-NL"}
SYMMETRIC = ["DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov"]
TAGS = ("-RAW", "-P50", "-P500", "-N54", "-K2", "-K4")


def lab(m):
    return LAB.get(m, m)


def base(m):
    for t in TAGS:
        if m.endswith(t):
            return m[: -len(t)]
    return m


def counterpart(m):
    return "GRU" if base(m) == "SetGRU" else "MLP"


def load(root, fname):
    f = root / "main" / "tables" / fname
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def select(d, flt):
    if d.empty:
        return d
    for k, v in flt.items():
        if k not in d.columns:
            return d.iloc[0:0]
        d = d[d[k] == v]
    return d


def fmt_p(r):
    p = f"{r.p_holm:.3f}" if r.p_holm >= 0.001 else "$<$0.001"
    if r.verdict == "more accurate":
        return f"\\textbf{{{p}}}"
    if r.verdict == "less accurate":
        return p + "$^{-}$"
    return p


def family_rows(title, fname, flt, versus, roots=None):
    """One block of rows: pair label, then DM statistic and Holm p-value in each universe (roots overrides the
    results directory of a universe and leaves the others empty)."""
    cells, bench = {}, {}
    for u, root in UNIVERSES:
        if roots is not None:
            if u not in roots:
                continue
            root = roots[u]
        d = select(load(root, fname), flt)
        for _, r in d.iterrows():
            b = r.benchmark if versus == "benchmark" and isinstance(r.get("benchmark"), str) else counterpart(r.model)
            if versus == "benchmark":
                bench.setdefault(u, lab(base(b)))
                key = lab(base(r.model))
            else:
                key = f"{lab(base(r.model))} vs {lab(base(b))}"
            cells.setdefault(key, {})[u] = (f"${r.dm:.2f}$", fmt_p(r))
    if not cells:
        return []
    if bench and " vs " not in title:  # a title with "vs" already names the benchmark
        title += " (benchmark: " + " / ".join(bench.get(u, "--") for u, _ in UNIVERSES) + ")"
    L = [f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}}\\\\"]
    for key, v in cells.items():
        L.append(f"\\quad {key} & " + " & ".join(" & ".join(v.get(u, ("--", "--"))) for u, _ in UNIVERSES) + "\\\\")
    return L


def hyp_table(name, caption, label, families):
    L = ["\\begin{table}[p]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}",
         f"\\caption{{{caption}}}\\label{{{label}}}",
         "\\begin{tabular}{lrrrr}\\toprule",
         " & \\multicolumn{2}{c}{ETF} & \\multicolumn{2}{c}{Stocks}\\\\ \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
         "Comparison & DM & $p_{\\text{Holm}}$ & DM & $p_{\\text{Holm}}$\\\\ \\midrule"]
    first = True
    for fam in families:
        rows = family_rows(*fam)
        if rows:
            if not first:
                L.append("\\addlinespace")
            L += rows
            first = False
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / f"{name}.tex").write_text("\n".join(L), encoding="utf-8")


CAPTION = ("Negative DM statistics favor the symmetric network (for differences of differentials: favor the stated "
           "condition). $p_{\\text{Holm}}$: Holm-adjusted $p$-value within the family; bold: decided in favor of the "
           "symmetric network; $^{-}$: decided against it. `--': not run.")


def levels_table():
    conds = [("", "unlimited"), ("-P500", "pool 500"), ("-P50", "pool 50"), ("-RAW", "raw inputs"),
             ("-N54", "trained $N\\le54$")]
    models = ["MLP", "GRU", "DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov"]
    lev = {u: load(root, "extra_levels.csv").set_index("model") for u, root in UNIVERSES}
    L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}",
         "\\caption{Second series of experiments ($h=5$): QLIKE relative to EWMA of the networks, in-range portfolios, "
         "under the training conditions of Section~\\ref{sec:hypotheses}: unlimited random portfolios (main design), "
         "fixed pools of 500 and 50 portfolios, raw inputs without portfolio-level summaries, and training on portfolios "
         "of up to 54 assets (last column: portfolios with $N>20$; $N\\in\\{30,40,54\\}$ for ETFs, $N\\in\\{30,40\\}$ "
         "for stocks). Seed ensembles. EQ-Cov uses no portfolio-level summaries and has no raw-input variant; the GRU and "
         "Set-GRU were not trained on large portfolios.}\\label{tab:extra_levels}",
         "\\begin{tabular}{lrrrrrr}\\toprule",
         " & \\multicolumn{5}{c}{In-range portfolios} & \\\\ \\cmidrule(lr){2-6}",
         "Model & " + " & ".join(c for _, c in conds) + " & $N>20$\\\\"]
    for u, _ in UNIVERSES:  # one panel per universe
        L += ["\\midrule", f"\\multicolumn{{7}}{{l}}{{\\emph{{{u}}}}}\\\\"]
        d = lev[u]
        for m in models:
            vals = [f"{d.loc[m + t, 'qlike_rel_ewma_in_range']:.3f}" if m + t in d.index else "--" for t, _ in conds]
            vals.append(f"{d.loc[m + '-N54', 'qlike_rel_ewma_larger']:.3f}" if m + "-N54" in d.index else "--")
            L.append(f"\\quad {lab(m)} & " + " & ".join(vals) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / "extra_levels.tex").write_text("\n".join(L), encoding="utf-8")


def economic_table():
    """Minimum-variance allocation (h=5) and volatility targeting (in-range portfolios) for every model, both universes."""
    from pvol.evaluate import CLASSICAL, NEURAL

    cols = {}
    for u, root in UNIVERSES:
        t = root / "main" / "tables"
        mv = pd.read_csv(t / "minvar_h5.csv")
        by = mv.groupby(["model", "origin"]).realised_var.mean().unstack(0)
        vol = (100 * (by.mean() * 252) ** 0.5).to_dict()  # annualized realized volatility, as in 17_economic_tests
        hhi = mv.groupby("model").hhi.mean().to_dict()
        vt = pd.read_csv(t / "vol_targeting.csv")
        vt = vt[vt.block == "in-range"].set_index(["model", "h"]).rmse_log_monthly
        cols[u] = (vol, hhi, vt)
    skip = {"HAR-WLS", "GBM", "PortMLP", "MLP-sorted"}  # not discussed in the economic evaluation; kept in the CSV files
    models = [m for m in CLASSICAL + NEURAL if m not in skip
              and any(m in vt.index.get_level_values(0) for _, _, vt in cols.values())]

    def best(values):  # bold the lowest value of a column
        v = [x for x in values if x is not None]
        return min(v) if v else None

    table = {}
    for u, (vol, hhi, vt) in cols.items():
        for m in models:
            table[(u, m)] = [vol.get(m), hhi.get(m)] + [vt.get((m, h)) for h in (1, 5, 22)]
    fmts = ["{:.2f}", "{:.2f}", "{:.3f}", "{:.3f}", "{:.3f}"]
    L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}",
         "\\caption{Economic evaluation. Minimum variance ($h=5$): annualized realized volatility (\\%) of the long-only "
         "portfolios that minimize each model's forecast over 100 random ten-asset subsets at every fifth test origin, and "
         "their average Herfindahl index of weights (HHI). Volatility targeting: root mean squared log deviation of "
         "monthly realized volatility from the target, in-range portfolios, rebalanced every $h$ days. Lower is better "
         "everywhere except HHI; bold: best in the column. Models absent from the minimum-variance experiment (univariate "
         "per-portfolio models) are marked `--'; HAR-WLS, GBM, PortMLP and MLP-sorted are in the replication package.}"
         "\\label{tab:economic}",
         "\\begin{tabular}{lrrrrrrrrrr}\\toprule",
         " & " + " & ".join(f"\\multicolumn{{5}}{{c}}{{{u}}}" for u, _ in UNIVERSES) + "\\\\",
         "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}",
         " & " + " & ".join("\\multicolumn{2}{c}{Min.\\ variance} & \\multicolumn{3}{c}{Vol.\\ targeting}"
                           for _ in UNIVERSES) + "\\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-11}",
         "Model & " + " & ".join("vol. & HHI & $h=1$ & $h=5$ & $h=22$" for _ in UNIVERSES) + "\\\\ \\midrule"]
    bests = {(u, j): best([table[(u, m)][j] for m in models]) for u, _ in UNIVERSES for j in (0, 2, 3, 4)}
    for m in models:
        if m == NEURAL[0]:
            L.append("\\midrule")
        cells = []
        for u, _ in UNIVERSES:
            for j, (v, f) in enumerate(zip(table[(u, m)], fmts)):
                c = "--" if v is None or v != v else f.format(v)
                if v is not None and bests.get((u, j)) is not None and f.format(v) == f.format(bests[(u, j)]):
                    c = f"\\textbf{{{c}}}"
                cells.append(c)
        L.append(f"{lab(m)} & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / "economic.tex").write_text("\n".join(L), encoding="utf-8")


def cost_table():
    """Parameters, training time and forecast time of the networks (scripts/22_compute_cost.py)."""
    c = {u: load(root, "compute_cost.csv").set_index("model") for u, root in UNIVERSES}
    if any(d.empty for d in c.values()):
        return
    etf, st = c["ETF"], c["Stocks"]

    def ms(v):
        return "--" if v != v else f"{v:.1f}"

    L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}",
         "\\caption{Computational cost of the networks. Parameters of the selected configurations (the input dimension "
         "differs between the universes: seven asset classes against eleven sectors); average training time and number "
         "of epochs per (seed, test year) run at $h=5$ (21 runs, early stopping); forecast time in milliseconds per 1,000 "
         "portfolios of $N$ constituents (forward pass on precomputed inputs, median of 20 repetitions; large portfolios "
         "are scored in chunks of at most $819{,}200/N^2$ portfolios, as in the forecasting code). NVIDIA GeForce "
         "RTX 3060 Ti; `--': the slot-based networks cannot score portfolios with more than 20 constituents.}"
         "\\label{tab:cost}",
         "\\begin{tabular}{lrrrrrrrrr}\\toprule",
         " & \\multicolumn{2}{c}{Parameters} & \\multicolumn{2}{c}{Training (s)} & \\multicolumn{2}{c}{Epochs} & "
         "\\multicolumn{3}{c}{Forecast (ms)}\\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-10}",
         "Model & ETF & Stocks & ETF & Stocks & ETF & Stocks & $N=20$ & $N=54$ & $N=122$\\\\ \\midrule"]
    for m in etf.index:
        a, b = etf.loc[m], st.loc[m]
        L.append(f"{lab(m)} & {int(a.params):,} & {int(b.params):,} & {a.train_sec:.0f} & {b.train_sec:.0f} & "
                 f"{a.epochs:.0f} & {b.epochs:.0f} & {ms(a.ms_20)} & {ms(a.ms_all)} & {ms(b.ms_all)}\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / "compute_cost.tex").write_text("\n".join(L), encoding="utf-8")

# appendix: robustness and supplementary tables
NETS = ["PortMLP", "MLP", "MLP-sorted", "GRU", "DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov"]
HS = (1, 5, 22)


def _both_universes_table(name, label, caption, value, models, fmt="{:.3f}", note_rows=None):
    """Rows = models, columns = ETF h=1,5,22 | Stocks h=1,5,22; value(u, root, h, m) -> float or None."""
    L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}",
         f"\\caption{{{caption}}}\\label{{{label}}}",
         "\\begin{tabular}{lrrrrrr}\\toprule",
         " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{u}}}" for u, _ in UNIVERSES) + "\\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "Model & " + " & ".join(f"$h={h}$" for _ in UNIVERSES for h in HS) + "\\\\ \\midrule"]
    for m in models:
        if m == "PortMLP":
            L.append("\\midrule")
        cells = []
        for u, root in UNIVERSES:
            for h in HS:
                v = value(u, root, h, m)
                cells.append("--" if v is None or v != v else fmt.format(v))
        L.append(f"{lab(m)} & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / f"{name}.tex").write_text("\n".join(L), encoding="utf-8")


def _losses(root, h, exp="main"):
    f = root / exp / "tables" / f"losses_h{h}.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def supplementary_tables():
    from pvol.evaluate import CLASSICAL

    models = [m for m in CLASSICAL if m != "EWMA"] + NETS

    def mse(u, root, h, m):
        d = _losses(root, h)
        r = d[(d.block == "in-range") & (d.model == m)]
        return float(r.mse_rel_ewma.iloc[0]) if len(r) else None

    _both_universes_table("mse", "tab:mse",
                          "Out-of-sample MSE relative to EWMA, in-range portfolios, 2020--2026 (secondary loss). MSE is "
                          "dominated by the few days with the largest realized variance.", mse, models)

    def mz(u, root, h, m):
        f = root / "main" / "tables" / f"mz_h{h}.csv"
        if not f.exists():
            return None
        d = pd.read_csv(f)
        r = d[(d.block == "in-range") & (d.model == m)]
        return float(r.r2.iloc[0]) if len(r) else None

    _both_universes_table("mz", "tab:mz",
                          "Mincer--Zarnowitz regressions of log realized on log forecast variance, in-range portfolios, "
                          "pooled over portfolios and origins: $R^2$.", mz, ["EWMA"] + models)

    def seed_sd(u, root, h, m):
        f = root / "main" / "tables" / f"seeds_h{h}.csv"
        d, lo = pd.read_csv(f), _losses(root, h)
        e = lo[(lo.block == "in-range") & (lo.model == "EWMA")].qlike.iloc[0]
        r = d[d.model == m]
        return float(100 * r.sd.iloc[0] / e) if len(r) else None

    _both_universes_table("seeds", "tab:seeds",
                          "Dispersion across the three training seeds: standard deviation of the QLIKE relative to EWMA of "
                          "the individual seeds, in percentage points of the EWMA loss (in-range portfolios). The evaluated forecast "
                          "is the ensemble of the seeds.", seed_sd, NETS[1:], fmt="{:.2f}")

    tail_table(["EWMA"] + models)


def tail_table(models):
    """Volatility targeting, in-range portfolios, h=1 and h=5: share of months above 1.5 x target, below target / 1.5,
    and mean log deviation of monthly realized volatility from the target, both universes."""
    vt = {u: pd.read_csv(root / "main" / "tables" / "vol_targeting.csv") for u, root in UNIVERSES}
    cols = [("tail_share", 100, "{:.1f}"), ("low_share", 100, "{:.1f}"), ("log_bias", 100, "${:+.1f}$")]
    L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}",
         "\\caption{Volatility targeting, in-range portfolios: share of months (\\%) in which the realized volatility of "
         "the levered portfolio is above 1.5 times the target (over) or below the target divided by 1.5 (under), and mean "
         "log deviation of monthly realized volatility from the target ($\\times100$; negative: realized volatility "
         "below the target on average, i.e.\\ variance forecasts too high and leverage too low). Descriptive only; "
         "these measures are not tested.}\\label{tab:tail}",
         "\\begin{tabular}{l" + "r" * 12 + "}\\toprule",
         " & " + " & ".join(f"\\multicolumn{{6}}{{c}}{{{u}}}" for u, _ in UNIVERSES) + "\\\\",
         "\\cmidrule(lr){2-7}\\cmidrule(lr){8-13}",
         " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{$h={h}$}}" for _ in UNIVERSES for h in (1, 5)) + "\\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}",
         "Model & " + " & ".join("over & under & bias" for _ in range(4)) + "\\\\ \\midrule"]
    for m in models:
        if m == "PortMLP":
            L.append("\\midrule")
        cells = []
        for u, _ in UNIVERSES:
            d = vt[u]
            for h in (1, 5):
                r = d[(d.block == "in-range") & (d.model == m) & (d.h == h)]
                cells += [f.format(k * r[c].iloc[0] + 0.0).replace("$-0.0$", "$0.0$") if len(r) else "--"
                          for c, k, f in cols]
        L.append(f"{lab(m)} & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / "tail.tex").write_text("\n".join(L), encoding="utf-8")


def protocol_table():
    """Alternative training protocol (tag -P2, ETF, h=5): ensemble accuracy and seed dispersion under both protocols."""
    from pvol.evaluate import group_index, load_run, qlike
    import importlib.util

    spec = importlib.util.spec_from_file_location("hyp", Path(__file__).with_name("15_hypotheses.py"))
    hyp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hyp)
    R = load_run("main", 5)
    ci = hyp.cols_of(group_index(R["ports"]), "in-range")
    e = np.nanmean(qlike(R["y"], R["F"]["EWMA"])[:, ci])

    def rel(f):
        return np.nanmean(qlike(R["y"], f)[:, ci]) / e

    L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{5pt}",
         "\\caption{Alternative training protocol (at least five epochs, patience 15, average of the weights of the three "
         "best validation epochs), ETF universe, $h=5$, in-range portfolios: QLIKE relative to EWMA of the seed ensemble and "
         "standard deviation across seeds (percentage points of the EWMA loss). The verdicts of H1 and H2 under this protocol are "
         "unchanged except where noted in the text.}\\label{tab:protocol}",
         "\\begin{tabular}{lrrrr}\\toprule",
         " & \\multicolumn{2}{c}{Main protocol} & \\multicolumn{2}{c}{Alternative protocol}\\\\ "
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
         "Model & QLIKE & sd & QLIKE & sd\\\\ \\midrule"]
    for m in NETS[1:]:
        if m + "-P2" not in R["F"]:
            continue
        cells = []
        for tag in (m, m + "-P2"):
            seeds = R["F_seed"].get(tag, {})
            cells += [f"{rel(R['F'][tag]):.3f}", f"{100 * np.std([rel(f) for f in seeds.values()], ddof=1):.2f}"]
        L.append(f"{lab(m)} & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / "protocol.tex").write_text("\n".join(L), encoding="utf-8")


def epochs_table():
    """Best validation epoch per (network, seed, test year), main design, h=5: median and share of runs whose best epoch
    is one of the first two."""
    import json

    rows = {}
    for u, root in UNIVERSES:
        for f in (root / "main" / "h5" / "info").glob("*__s*__y*.json"):
            d = json.loads(f.read_text())
            if d.get("tag", d["model"]) not in NETS[1:] or not d.get("history"):
                continue
            best = 1 + int(np.argmin([r[2] for r in d["history"]]))
            rows.setdefault((u, d["tag"]), []).append(best)
    L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{5pt}",
         "\\caption{Epoch with the best validation loss (the weights kept by early stopping), main design, $h=5$, over "
         "seeds and test years (21 runs per cell): median epoch and share of runs in which it is the first or the second "
         "epoch, i.e.\\ in which the kept network has been trained for at most 250 optimization steps.}\\label{tab:epochs}",
         "\\begin{tabular}{lrrrr}\\toprule",
         " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{u}}}" for u, _ in UNIVERSES) + "\\\\ "
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
         "Model & " + " & ".join("median & $\\le2$" for _ in UNIVERSES) + "\\\\ \\midrule"]
    for m in NETS[1:]:
        cells = []
        for u, _ in UNIVERSES:
            b = rows.get((u, m), [])
            cells += [f"{np.median(b):.0f}", f"{100 * np.mean(np.array(b) <= 2):.0f}\\%"] if b else ["--", "--"]
        L.append(f"{lab(m)} & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / "epochs.tex").write_text("\n".join(L), encoding="utf-8")


def heldout_table():
    """Unseen assets (H3c): QLIKE relative to EWMA for portfolios composed only of assets excluded from training."""
    from pvol.evaluate import CLASSICAL

    cols = [("ETF", BASE, h) for h in HS] + [("Stocks", Path(str(BASE) + "_stocks"), 5)]
    vals = {}
    for u, root, h in cols:
        d = _losses(root, h, "heldout")
        if d.empty:
            continue
        d = d[d.block == "unseen-assets"].set_index("model").qlike_rel_ewma
        vals[(u, h)] = d
    models = [m for m in CLASSICAL + NETS if any(m in v.index for v in vals.values())]
    L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{5pt}",
         "\\caption{Unseen assets (H3c): QLIKE relative to EWMA for portfolios of 5 and 10 assets composed only of the "
         "assets excluded from training (12 funds, 28 stocks); the stock universe was run at $h=5$ only. Bold: best in the "
         "column.}\\label{tab:heldout}",
         "\\begin{tabular}{lrrrr}\\toprule",
         " & \\multicolumn{3}{c}{ETF} & Stocks\\\\ \\cmidrule(lr){2-4}\\cmidrule(lr){5-5}",
         "Model & $h=1$ & $h=5$ & $h=22$ & $h=5$\\\\ \\midrule"]
    for m in models:
        if m == "PortMLP":
            L.append("\\midrule")
        cells = []
        for u, _, h in cols:
            v = vals.get((u, h))
            if v is None or m not in v.index:
                cells.append("--")
                continue
            c = f"{v[m]:.3f}"
            cells.append(f"\\textbf{{{c}}}" if np.isclose(v[m], v.min()) else c)
        L.append(f"{lab(m)} & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / "heldout.tex").write_text("\n".join(L), encoding="utf-8")


def structured_table():
    from pvol.portfolios import STRUCTURED

    names = {"60/40": "60/40", "AllWeather": "All-weather type", "GlobalMulti": "Global multi-asset"}
    L = ["\\begin{table}[!htbp]\\centering\\footnotesize\\setlength{\\tabcolsep}{5pt}",
         "\\caption{Structured allocations of the ETF universe (constant weights, rebalanced daily). The long--short "
         "portfolios have ten constituents with weights $z_k/\\sum_l|z_l|$, $z_k$ independent standard normal, i.e.\\ unit "
         "gross exposure and a random net exposure with mean zero.}\\label{tab:structured}",
         "\\begin{tabular}{lp{11cm}}\\toprule", "Allocation & Weights\\\\ \\midrule"]
    for k, spec in STRUCTURED.items():
        L.append(f"{names.get(k, k)} & " + ", ".join(f"{t} {100 * w:g}\\%" for t, w in spec.items()) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / "structured.tex").write_text("\n".join(L), encoding="utf-8")


# third series (R1-R5)
RLAB = {"-P50s": " (pool 50, shuffled)", "-P500s": " (pool 500, shuffled)", "-P50": " (pool 50)", "-P500": " (pool 500)",
        "-RAW": " (raw)", "-TTA": " (TTA)"}


def rlab(m):
    for suf in ("-P500s", "-P50s", "-P500", "-P50", "-RAW", "-TTA"):
        if m.endswith(suf):
            b = m[: -len(suf)]
            return ("Deep Sets + pairs" if b == "DeepSetsP" else lab(b)) + RLAB[suf]
    return "Deep Sets + pairs" if m == "DeepSetsP" else lab(m)


def third_series_tables():
    import pandas as pd

    fams = [("R1a: symmetric (fixed pool of 50) vs counterpart (same pool, shuffled)", dict(hypothesis="R1a", block="in-range, pool 50, shuffled counterpart")),
            ("R1a: symmetric (fixed pool of 500) vs counterpart (same pool, shuffled)", dict(hypothesis="R1a", block="in-range, pool 500, shuffled counterpart")),
            ("R1b: shuffled vs fixed order, same pool", dict(hypothesis="R1b")),
            ("R2: symmetric vs MLP-sorted, raw inputs", dict(hypothesis="R2", block="in-range, raw inputs")),
            ("R2: symmetric vs MLP-sorted, pool of 50", dict(hypothesis="R2", block="in-range, pool 50")),
            ("R2: symmetric vs MLP-sorted, pool of 500", dict(hypothesis="R2", block="in-range, pool 500")),
            ("R3: symmetric vs counterpart with 8-ordering averaging, $h=1$", dict(hypothesis="R3", h=1)),
            ("R3: symmetric vs counterpart with 8-ordering averaging, $h=5$", dict(hypothesis="R3", h=5)),
            ("R3 (auxiliary): 8-ordering averaging vs a single ordering, $h=1$",
             dict(hypothesis="R3 (aux: TTA vs single ordering)", h=1)),
            ("R3 (auxiliary): 8-ordering averaging vs a single ordering, $h=5$",
             dict(hypothesis="R3 (aux: TTA vs single ordering)", h=5)),
            ("R4: Deep Sets with pair summaries", dict(hypothesis="R4")),
            ("R4 (auxiliary)", dict(hypothesis="R4 (aux)"))]
    d = {u: load(root, "hypotheses_review.csv") for u, root in UNIVERSES}
    parts = [("hyp_review", "tab:hyp_review", fams[:6], "Third series of experiments, R1 and R2 ($h=5$, in-range "
              "portfolios): Diebold--Mariano statistics and Holm-adjusted $p$-values."),
             ("hyp_review2", "tab:hyp_review2", fams[6:], "Third series of experiments, R3 and R4 ($h=5$ unless stated, "
              "in-range portfolios): Diebold--Mariano statistics and Holm-adjusted $p$-values.")]
    for name, label, part, caption in parts:
        L = ["\\begin{table}[p]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}",
             f"\\caption{{{caption} Negative statistics favor the first model. Bold: first model significantly more "
             f"accurate; $^{{-}}$: significantly less accurate.}}\\label{{{label}}}",
             "\\begin{tabular}{lrrrr}\\toprule",
             " & \\multicolumn{2}{c}{ETF} & \\multicolumn{2}{c}{Stocks}\\\\ \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "Comparison & DM & $p_{\\text{Holm}}$ & DM & $p_{\\text{Holm}}$\\\\ \\midrule"]
        first = True
        for title, flt in part:
            cells = {}
            for u, _ in UNIVERSES:
                for _, r in select(d[u], flt).iterrows():
                    cells.setdefault(f"{rlab(r.model)} vs {rlab(r.benchmark)}", {})[u] = (f"${r.dm:.2f}$", fmt_p(r))
            if not cells:
                continue
            if not first:
                L.append("\\addlinespace")
            first = False
            L.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}}\\\\")
            for k, v in cells.items():
                L.append(f"\\quad {k} & " + " & ".join(" & ".join(v.get(u, ("--", "--"))) for u, _ in UNIVERSES) + "\\\\")
        L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
        (OUT / f"{name}.tex").write_text("\n".join(L), encoding="utf-8")

    # levels: what each cheap fix does to the unconstrained network
    lev = {u: load(root, "review_levels.csv") for u, root in UNIVERSES}
    s54 = load(Path(str(BASE) + "_stocks54"), "review_levels.csv")
    rows = [("A. Fixed pool of 50 portfolios", None),
            ("MLP, fixed order", "MLP-P50"), ("MLP, shuffled order", "MLP-P50s"), ("MLP-sorted", "MLP-sorted-P50"),
            ("PE-Net", "PENet-P50"), ("Set Transformer", "SetTransformer-P50"),
            ("B. Raw inputs", None),
            ("MLP", "MLP-RAW"), ("MLP-sorted", "MLP-sorted-RAW"), ("Deep Sets", "DeepSets-RAW"),
            ("Deep Sets + pairs", "DeepSetsP-RAW"), ("PE-Net", "PENet-RAW"),
            ("C. Main design", None),
            ("MLP", "MLP"), ("MLP, 8-ordering averaging", "MLP-TTA"), ("MLP-sorted", "MLP-sorted"),
            ("Deep Sets + pairs", "DeepSetsP"), ("PE-Net", "PENet"),
            ("D. Main design, $h=1$ (R3)", None),
            ("MLP", "MLP"), ("MLP, 8-ordering averaging", "MLP-TTA"), ("GRU", "GRU"),
            ("GRU, 8-ordering averaging", "GRU-TTA"), ("Deep Sets", "DeepSets"), ("PE-Net", "PENet"),
            ("Set Transformer", "SetTransformer"), ("Set-GRU", "SetGRU"), ("EQ-Cov", "EQCov")]

    def val(frame, h, m):
        x = frame[(frame.h == h) & (frame.model == m)] if len(frame) else frame
        return f"{x.qlike_rel_ewma.iloc[0]:.3f}" if len(x) else "--"

    L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{5pt}",
         "\\caption{Third series: QLIKE relative to EWMA (in-range portfolios, seed ensembles; $h=5$ in panels A--C, "
         "$h=1$ in panel D) of the unconstrained networks with "
         "the cheap alternatives to architectural symmetry---a new random order in every draw from the pool, "
         "canonicalization by sorting, averaging over 8 orderings at prediction time---and of symmetric networks for "
         "reference. Deep Sets + pairs adds the weight-averaged correlation of each constituent with the others to its "
         "element features.}\\label{tab:review_levels}",
         "\\begin{tabular}{lrr}\\toprule",
         "Network & ETF & Stocks\\\\"]
    hz = 5
    for name, tag in rows:
        if tag is None:
            hz = 1 if name.startswith("D.") else 5
            L += ["\\midrule", f"\\multicolumn{{3}}{{l}}{{\\emph{{{name}}}}}\\\\"]
            continue
        cells = [val(lev[u], hz, tag) for u, _ in UNIVERSES]
        L.append(f"\\quad {name} & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / "review_levels.tex").write_text("\n".join(L), encoding="utf-8")

    # R5: the 54-stock subsample
    r5 = load(Path(str(BASE) + "_stocks54"), "hypotheses_review.csv")
    L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{5pt}",
         "\\caption{Subsample of 54 of the 122 stocks (stratified by sector): H1 in the in-range block. QLIKE relative to "
         "EWMA of the symmetric network and of its counterpart, DM statistic and Holm-adjusted $p$-value (bold: symmetric "
         "network significantly more accurate).}\\label{tab:stocks54}",
         "\\begin{tabular}{lrrrrrrrr}\\toprule",
         " & \\multicolumn{4}{c}{$h=1$} & \\multicolumn{4}{c}{$h=5$}\\\\ \\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
         "Pair & sym. & count. & DM & $p_{\\text{Holm}}$ & sym. & count. & DM & $p_{\\text{Holm}}$\\\\ \\midrule"]
    for a in ["DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov"]:
        cells = []
        for h in (1, 5):
            r = r5[(r5.h == h) & (r5.model == a)]
            if not len(r):
                cells += ["--"] * 4
                continue
            r = r.iloc[0]
            cells += [val(s54, h, a), val(s54, h, r.benchmark), f"${r.dm:.2f}$", fmt_p(r)]
        L.append(f"{lab(a)} vs {'GRU' if a == 'SetGRU' else 'MLP'} & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / "stocks54.tex").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    economic_table()
    cost_table()
    supplementary_tables()
    protocol_table()
    epochs_table()
    heldout_table()
    structured_table()
    third_series_tables()
    H = "hypotheses.csv"
    hyp_table("hyp_accuracy",
              "Tests behind the verdicts of H1 (symmetric network vs its counterpart) and H2 (symmetric network vs the "
              "best classical benchmark of the block), in-range portfolios. " + CAPTION, "tab:hyp_accuracy",
              [(f"H1, $h={h}$", H, dict(hypothesis="H1", h=h, block="in-range"), "pair") for h in (1, 5, 22)]
              + [(f"H2, $h={h}$", H, dict(hypothesis="H2", h=h, block="in-range"), "benchmark") for h in (1, 5, 22)])
    hyp_table("hyp_other",
              "Tests behind the verdicts of H3a (improvement over the best in-range classical benchmark larger for "
              "$N>20$ than in range), H3b (long--short portfolios), H3c (unseen assets), H4 (shortened training "
              "history; differential with $K$ years minus differential with the full history, positive = the symmetric "
              "network loses more), $h=5$. " + CAPTION, "tab:hyp_other",
              [("H3a, larger $N$ minus in-range", H, dict(hypothesis="H3a", h=5), "benchmark"),
               ("H3b, long--short", H, dict(hypothesis="H3b", h=5), "pair"),
               ("H3c, unseen assets", H, dict(hypothesis="H3c", h=5), "pair"),
               ("H4, $K=4$ years", H, dict(hypothesis="H4", block="in-range, K=4 minus full"), "pair"),
               ("H4, $K=2$ years", H, dict(hypothesis="H4", block="in-range, K=2 minus full"), "pair")])
    hyp_table("hyp_econ",
              "Tests behind the verdicts of the economic comparisons: minimum variance ($h=5$) and volatility targeting "
              "(root mean squared log deviation of monthly realized volatility from the target, every horizon). "
              + CAPTION, "tab:hyp_econ",
              [               ("H1, minimum variance", "economic_tests.csv", dict(hypothesis="H1-minvar"), "pair"),
               ("H2, minimum variance", "economic_tests.csv", dict(hypothesis="H2-minvar"), "benchmark"),
               *[(f"H1, volatility targeting, $h={h}$", "economic_tests.csv",
                  dict(hypothesis="H1-voltarget", h=float(h)), "pair") for h in (1, 5, 22)],
               *[(f"H2, volatility targeting, $h={h}$", "economic_tests.csv",
                  dict(hypothesis="H2-voltarget", h=float(h)), "benchmark") for h in (1, 5, 22)]])
    X = "hypotheses_extra.csv"
    hyp_table("hyp_extra",
              "Tests behind the verdicts of the second series ($h=5$, in-range portfolios unless stated). HE1d and HE2d "
              "test differences of the daily differentials (raw minus main design; pool of 50 minus pool of 500). "
              + CAPTION, "tab:hyp_extra",
              [("HE1, raw inputs", X, dict(hypothesis="HE1"), "pair"),
               ("HE1d, raw minus main design", X, dict(hypothesis="HE1d"), "pair"),
               ("HE2, pool of 500", X, dict(hypothesis="HE2", block="in-range, pool 500"), "pair"),
               ("HE2, pool of 50", X, dict(hypothesis="HE2", block="in-range, pool 50"), "pair"),
               ("HE2d, pool of 50 minus pool of 500", X, dict(hypothesis="HE2d"), "pair"),
               ("HE3, portfolios with $N>20$", X, dict(hypothesis="HE3"), "pair")])
    F, O = "hypotheses_fourth.csv", "hypotheses_h2orig.csv"
    hyp_table("hyp_fourth",
              "Tests behind the verdicts of S1b (symmetric network vs the invariant linear model and vs GBM), "
              "in-range portfolios. " + CAPTION, "tab:hyp_fourth",
              [(f"S1b vs Linear, $h={h}$", F, dict(hypothesis="S1b", h=h, benchmark="Linear"), "benchmark")
               for h in (1, 5, 22)]
              + [(f"S1b vs GBM, $h={h}$", F, dict(hypothesis="S1b", h=h, benchmark="GBM"), "benchmark")
                 for h in (1, 5, 22)])
    hyp_table("hyp_fourth2",
              "Tests behind the verdicts of S2 (H1, R3 with the counterpart averaged over eight orderings, and R5 on "
              "the 54-stock subsample, at $h=1$ in the stock universe with ensembles of ten seeds) and of H2 against "
              "the best of the ten classical models of the original design (in-range portfolios). " + CAPTION,
              "tab:hyp_fourth2",
              [("S2, H1 with ten seeds, $h=1$", F, dict(hypothesis="S2-H1"), "pair"),
                 ("S2, R3 with ten seeds, $h=1$", F, dict(hypothesis="S2-R3"), "pair"),
                 ("S2, R5 with ten seeds, $h=1$", F, dict(hypothesis="S2-R5"), "pair",
                  {"Stocks": Path(str(BASE) + "_stocks54")})]
              + [(f"H2, original ten, $h={h}$", O, dict(hypothesis="H2", h=h, block="in-range"),
                  "benchmark") for h in (1, 5, 22)])
    levels_table()
    print("tables written to", OUT)
