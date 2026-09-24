"""Appendix tables with the test statistics behind every verdict of the scoreboard, and the level table of the second
series (E1-E3), for both universes; built from results*/main/tables/*.csv only.

hyp_accuracy.tex  H1 and H2 at every horizon (in-range block)
hyp_other.tex     H3a-H3c, H4 and the minimum-variance comparisons
hyp_extra.tex     HE1-HE3
extra_levels.tex  QLIKE relative to EWMA of the networks under the training conditions of E1-E3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
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


def family_rows(title, fname, flt, versus):
    """One block of rows: pair label, then DM statistic and Holm p-value in each universe."""
    cells, bench = {}, {}
    for u, root in UNIVERSES:
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
    if bench:
        title += " (benchmark: " + " / ".join(bench.get(u, "--") for u, _ in UNIVERSES) + ")"
    L = [f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}}\\\\"]
    for key, v in cells.items():
        L.append(f"\\quad {key} & " + " & ".join(" & ".join(v.get(u, ("--", "--"))) for u, _ in UNIVERSES) + "\\\\")
    return L


def hyp_table(name, caption, label, families):
    L = ["\\begin{table}[p]\\centering\\footnotesize\\setlength{\\tabcolsep}{5pt}",
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


CAPTION = ("Negative DM statistics favour the symmetric network (for differences of differentials: favour the stated "
           "condition). $p_{\\text{Holm}}$: Holm-adjusted $p$-value within the family; bold: decided in favour of the "
           "symmetric network; $^{-}$: decided against it. `--': not run.")


def levels_table():
    conds = [("", "unlimited"), ("-P500", "pool 500"), ("-P50", "pool 50"), ("-RAW", "raw inputs"), ("-N54", "$N\\le54$")]
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
        vol = (100 * (by.mean() * 252) ** 0.5).to_dict()  # annualised realised volatility, as in 17_economic_tests
        hhi = mv.groupby("model").hhi.mean().to_dict()
        vt = pd.read_csv(t / "vol_targeting.csv")
        vt = vt[vt.block == "in-range"].set_index(["model", "h"]).rmse_log_monthly
        cols[u] = (vol, hhi, vt)
    models = [m for m in CLASSICAL + NEURAL if any(m in vt.index.get_level_values(0) for _, _, vt in cols.values())]

    def best(values):  # bold the lowest value of a column
        v = [x for x in values if x is not None]
        return min(v) if v else None

    table = {}
    for u, (vol, hhi, vt) in cols.items():
        for m in models:
            table[(u, m)] = [vol.get(m), hhi.get(m)] + [vt.get((m, h)) for h in (1, 5, 22)]
    fmts = ["{:.2f}", "{:.2f}", "{:.3f}", "{:.3f}", "{:.3f}"]
    L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}",
         "\\caption{Economic evaluation. Minimum variance ($h=5$): annualised realised volatility (\\%) of the long-only "
         "portfolios that minimise each model's forecast over 100 random ten-asset subsets at every fifth test origin, and "
         "their average Herfindahl index of weights (HHI). Volatility targeting: root mean squared log deviation of "
         "monthly realised volatility from the target, in-range portfolios, rebalanced every $h$ days. Lower is better "
         "everywhere except HHI; bold: best in the column. Models absent from the minimum-variance experiment (univariate "
         "per-portfolio models) are marked `--'.}\\label{tab:economic}",
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


if __name__ == "__main__":
    economic_table()
    cost_table()
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
              "network loses more) and the minimum-variance comparisons, $h=5$. " + CAPTION, "tab:hyp_other",
              [("H3a, larger $N$ minus in-range", H, dict(hypothesis="H3a", h=5), "benchmark"),
               ("H3b, long--short", H, dict(hypothesis="H3b", h=5), "pair"),
               ("H3c, unseen assets", H, dict(hypothesis="H3c", h=5), "pair"),
               ("H4, $K=4$ years", H, dict(hypothesis="H4", block="in-range, K=4 minus full"), "pair"),
               ("H4, $K=2$ years", H, dict(hypothesis="H4", block="in-range, K=2 minus full"), "pair"),
               ("H1, minimum variance", "economic_tests.csv", dict(hypothesis="H1-minvar"), "pair"),
               ("H2, minimum variance", "economic_tests.csv", dict(hypothesis="H2-minvar"), "benchmark")])
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
    levels_table()
    print("tables written to", OUT)
