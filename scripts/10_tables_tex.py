"""LaTeX tables for the paper, generated only from results/<exp>/tables/*.csv (no numbers typed by hand)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import HORIZONS, RESULTS, UNIVERSE_NAME  # noqa: E402
from pvol.evaluate import CLASSICAL, NEURAL  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
SUF = "" if UNIVERSE_NAME == "etf" else f"_{UNIVERSE_NAME}"  # keep the tables of both universes side by side
LAB = {"SetTransformer": "Set Transformer", "DeepSets": "Deep Sets", "EQCov": "EQ-Cov", "PENet": "PE-Net",
       "SetGRU": "Set-GRU", "HARX": "HAR-X", "GJR": "GJR-GARCH",
       "EWMANLS": "EWMA-NLS", "DCCNL": "DCC-NL"}
SYM = {"PortMLP": "agg.", "Linear": "pool.", "GBM": "pool.", "MLP": "--", "MLP-sorted": "canon.", "GRU": "--", "DeepSets": "arch.", "PENet": "arch.",
       "SetTransformer": "arch.", "SetGRU": "arch.", "EQCov": "arch."}
BLOCKS = [("in-range", "In-range"), ("larger-N", "Larger $N$"), ("long-short", "Long--short"), ("structured", "Structured")]


def lab(m):
    return LAB.get(m, m)


def stars(p):
    return "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.10 else ""


BASE = Path(str(RESULTS).replace("_stocks", ""))
UNIVERSES = [("ETF", BASE), ("Stocks", Path(str(BASE) + "_stocks"))]
PAIRS = [("MLP-sorted", "MLP"), ("DeepSets", "PortMLP"), ("EQCov", "DCC"), ("EQCov", "DCCNL")]


def qlike_cells(root, exp="main"):
    """{(h, block): {model: cell}} with QLIKE relative to EWMA; best in bold, 90% MCS members marked with a dagger."""
    out = {}
    for h in HORIZONS:
        f, t = root / exp / "tables" / f"losses_h{h}.csv", root / exp / "tables" / f"tests_h{h}.csv"
        if not f.exists():
            continue
        d, tests = pd.read_csv(f), pd.read_csv(t)
        for b, _ in BLOCKS:
            db = d[d.block == b].set_index("model")
            if db.empty:
                continue
            mcs = tests[(tests.block == b) & (tests.benchmark == "MCS")].set_index("model").p_value
            best = db.qlike_rel_ewma.min()
            for m, v in db.qlike_rel_ewma.items():
                cell = f"{v:.3f}"
                if np.isclose(v, best):
                    cell = f"\\textbf{{{cell}}}"
                if m in mcs.index and mcs[m] > 0.10:
                    cell += "$^\\dagger$"
                out.setdefault((h, b), {})[m] = cell
    return out


MAIN_NOTE = ("Bold: best model in the column. $^\\dagger$: member of the 90\\% Model Confidence Set of the column. "
             "Neural forecasts are seed ensembles.")


# main text: the strongest classical models, the invariant linear model and the networks; all models in main_all.tex
MAIN_MODELS = ["EWMA", "HAR", "HARX", "LogHAR-X", "HEAVY", "GJR", "DCC",
               "Linear", "MLP", "MLP-sorted", "GRU", "DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov"]


def main_table(exp="main", full=False):
    """In-range portfolios, three horizons, both universes (portrait): main text (selected models) or supplement (all)."""
    cells = {u: qlike_cells(root, exp) for u, root in UNIVERSES}
    models = [m for m in CLASSICAL + NEURAL if (full or m in MAIN_MODELS)
              and any(m in c.get((h, "in-range"), {}) for c in cells.values() for h in HORIZONS)]
    first_learned = next(m for m in models if m in NEURAL)
    L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{5pt}",
         "\\caption{Out-of-sample QLIKE relative to EWMA (values below 1 beat RiskMetrics) for the in-range portfolios, test "
         "years 2020--2026, averaged over portfolios and forecast origins" + ("" if full else ", for the strongest "
         "classical models, the invariant linear model and the networks") + ". Symmetry: agg.\\ = invariant because only "
         "portfolio aggregates are used; pool.\\ = invariant pooled features; canon.\\ = canonicalisation by sorting; "
         "arch.\\ = architectural; -- = none. Bold: best of all models in the column. $^\\dagger$: member of the 90\\% "
         "Model Confidence Set of the column. Learned forecasts are seed ensembles."
         + ("}\\label{tab:main_all}" if full else " All models are reported in the Supplementary Material.}\\label{tab:main}"),
         "\\begin{tabular}{llrrrrrr}\\toprule",
         " & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{u}}}" for u, _ in UNIVERSES) + "\\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
         "Model & Sym. & " + " & ".join(f"$h={h}$" for _ in UNIVERSES for h in HORIZONS) + "\\\\ \\midrule"]
    for m in models:
        if m in (first_learned, "MLP", "DeepSets"):
            L.append("\\midrule")
        vals = [cells[u].get((h, "in-range"), {}).get(m, "--") for u, _ in UNIVERSES for h in HORIZONS]
        L.append(f"{lab(m)} & {SYM.get(m, '')} & " + " & ".join(vals) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / ("main_all.tex" if full else "main_main.tex")).write_text("\n".join(L), encoding="utf-8")


def blocks_table(exp="main"):
    """Appendix tables: the out-of-distribution blocks, one portrait table per universe."""
    for u, root in UNIVERSES:
        cells = qlike_cells(root, exp)
        blocks = [(b, lbl) for b, lbl in BLOCKS[1:] if any((h, b) in cells for h in HORIZONS)]
        cols = [(h, b) for h in HORIZONS for b, _ in blocks]
        models = [m for m in CLASSICAL + NEURAL if any(m in cells.get(c, {}) for c in cols)]
        short = {"larger-N": "Larger $N$", "long-short": "Long--sh.", "structured": "Struct."}
        suf = "" if u == "ETF" else "_stocks"
        what = ("$N\\in\\{30,40\\}$ and the equally weighted portfolio of all funds; ten-asset long--short portfolios; "
                "three structured allocations" if u == "ETF" else
                "$N\\in\\{30,40\\}$ and the equally weighted portfolio of all stocks; ten-asset long--short portfolios")
        L = ["\\begin{table}[p]\\centering\\footnotesize\\setlength{\\tabcolsep}{3.5pt}",
             f"\\caption{{{u} universe: QLIKE relative to EWMA for the portfolio blocks never seen in training ({what}). "
             + MAIN_NOTE + " `--': the slot-based networks cannot score portfolios with more than 20 constituents.}"
             f"\\label{{tab:blocks{suf.replace('_', ':')}}}",
             "\\begin{tabular}{l" + "r" * len(cols) + "}\\toprule",
             " & " + " & ".join(f"\\multicolumn{{{len(blocks)}}}{{c}}{{$h={h}$}}" for h in HORIZONS) + "\\\\",
             " ".join(f"\\cmidrule(lr){{{2 + i * len(blocks)}-{1 + (i + 1) * len(blocks)}}}" for i in range(len(HORIZONS))),
             "Model & " + " & ".join(short[b] for _ in HORIZONS for b, _ in blocks) + "\\\\ \\midrule"]
        for m in models:
            if m in (NEURAL[0], "DeepSets"):
                L.append("\\midrule")
            L.append(f"{lab(m)} & " + " & ".join(cells.get(c, {}).get(m, "--") for c in cols) + "\\\\")
        L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
        (OUT / f"blocks_{exp}{suf}.tex").write_text("\n".join(L), encoding="utf-8")


def dm_table(exp="main", block="in-range"):
    """Diebold-Mariano statistics for the matched pairs, both universes (negative = first model more accurate)."""
    cells = {}
    for u, root in UNIVERSES:
        for h in HORIZONS:
            t = root / exp / "tables" / f"tests_h{h}.csv"
            if not t.exists():
                continue
            d = pd.read_csv(t)
            for a, c in PAIRS:
                r = d[(d.block == block) & (d.model == a) & (d.benchmark == c)]
                if len(r):
                    r = r.iloc[0]
                    cells[(u, a, c, h)] = f"${r.dm_stat:.2f}{stars(r.p_value)}$"
    what = "in-range portfolios" if block == "in-range" else "portfolios of assets excluded from training"
    L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{5pt}",
         "\\caption{Diebold--Mariano tests of equal QLIKE for comparisons outside the hypotheses, " + what + " "
         "(HLN-corrected statistics on daily cross-sectional average loss differentials, unadjusted significance): "
         "canonicalisation, reading the constituents, and a learned against a classical covariance model. Negative values "
         "favour the first model. $^{*}$, $^{**}$, $^{***}$: 10\\%, 5\\%, 1\\%. The matched comparisons of the "
         "hypotheses are reported in the appendix of the article.}\\label{tab:dm_" + exp + "}",
         "\\begin{tabular}{lrrrrrr}\\toprule",
         " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{u}}}" for u, _ in UNIVERSES) + "\\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "Pair & " + " & ".join(f"$h={h}$" for _ in UNIVERSES for h in HORIZONS) + "\\\\ \\midrule"]
    for a, c in PAIRS:
        vals = [cells.get((u, a, c, h), "--") for u, _ in UNIVERSES for h in HORIZONS]
        if all(v == "--" for v in vals):
            continue
        L.append(f"{lab(a)} vs {lab(c)} & " + " & ".join(vals) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / f"dm_{exp}.tex").write_text("\n".join(L), encoding="utf-8")


def perm_table_both(exp="main"):
    """Permutation consistency, both universes: median (panel A) and 99th percentile (panel B) of the range of the
    log-variance forecast across five orderings of the same portfolio."""
    vals = {}
    for u, root in UNIVERSES:
        for h in HORIZONS:
            f = root / exp / "tables" / f"perm_h{h}.csv"
            if f.exists() and f.stat().st_size > 10:
                for _, r in pd.read_csv(f).iterrows():
                    vals[(u, r.model, h)] = (r.median_range_logvar, r.p99_range_logvar)
    models = [m for m in NEURAL + ["MLP-TTA", "GRU-TTA"]
              if any((u, m, h) in vals for u, _ in UNIVERSES for h in HORIZONS)
              and any(vals.get((u, m, h), (0, 0))[1] > 1e-4 for u, _ in UNIVERSES for h in HORIZONS)]
    L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{5pt}",
         "\\caption{Permutation consistency: the same 100 ten-asset portfolios are forecast with their constituents listed "
         "in five different orders. Median (panel A) and 99th percentile (panel B), over portfolios and origins, of the "
         "range of the log-variance forecast across the orderings; averages over seeds. A range of 0.07 means that two "
         "orderings of the same portfolio receive variance forecasts that differ by about 7\\%. For PortMLP, MLP-sorted "
         "and all architecturally symmetric networks the range is zero up to floating-point error ($<10^{-6}$).}"
         "\\label{tab:perm}",
         "\\begin{tabular}{lrrrrrr}\\toprule",
         " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{u}}}" for u, _ in UNIVERSES) + "\\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "Model & " + " & ".join(f"$h={h}$" for _ in UNIVERSES for h in HORIZONS) + "\\\\"]
    for k, title in enumerate(["A. Median range", "B. 99th percentile of the range"]):
        L += ["\\midrule", f"\\multicolumn{{7}}{{l}}{{\\emph{{{title}}}}}\\\\"]
        for m in models:
            cells = [f"{vals[(u, m, h)][k]:.3f}" if (u, m, h) in vals else "--" for u, _ in UNIVERSES for h in HORIZONS]
            L.append(f"\\quad {lab(m)} & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / "perm_main.tex").write_text("\n".join(L), encoding="utf-8")


def var_table_both(exp="main"):
    d = {u: pd.read_csv(root / exp / "tables" / "var_h1.csv") for u, root in UNIVERSES}
    models = [m for m in CLASSICAL + NEURAL if m in set(d["ETF"].model) | set(d["Stocks"].model)]
    L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{4.5pt}",
         "\\caption{Backtest of one-day Gaussian value-at-risk for the in-range portfolios, 2020--2026: hit rates at the 1\\% "
         "and 5\\% levels and, at the 1\\% level, the share of portfolios for which the unconditional coverage "
         "\\citep{kupiec1995} and conditional coverage \\citep{christoffersen1998} tests reject at 5\\%. All entries in per "
         "cent.}\\label{tab:var}",
         "\\begin{tabular}{lrrrrrrrr}\\toprule",
         " & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{u}}}" for u, _ in UNIVERSES) + "\\\\",
         "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
         "Model & " + " & ".join("hit 1\\% & UC & CC & hit 5\\%" for _ in UNIVERSES) + "\\\\ \\midrule"]
    for m in models:
        if m in (NEURAL[0], "DeepSets"):
            L.append("\\midrule")
        cells = []
        for u, _ in UNIVERSES:
            x = d[u]
            a, b = x[(x.model == m) & (x.alpha == 0.01)], x[(x.model == m) & (x.alpha == 0.05)]
            if len(a) and len(b):
                a, b = a.iloc[0], b.iloc[0]
                cells += [f"{100 * a.hit_rate:.2f}", f"{100 * a.rej_uc:.0f}", f"{100 * a.rej_cc:.0f}", f"{100 * b.hit_rate:.2f}"]
            else:
                cells += ["--"] * 4
        L.append(f"{lab(m)} & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / "var_main.tex").write_text("\n".join(L), encoding="utf-8")


def universe_table(root, name, caption, label, place="t", size="small"):
    f = root / "descriptives" / "universe_stats.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    L = [f"\\begin{{table}}[{place}]\\centering\\{size}", f"\\caption{{{caption}}}\\label{{{label}}}",
         "\\begin{tabular}{p{2.8cm}rp{5.4cm}rrrr}\\toprule",
         "Class & $n$ & Tickers & Ret. & Vol. & $\\overline{\\text{corr}}_{\\text{in}}$ & "
         "$\\overline{\\text{corr}}_{\\text{out}}$\\\\ \\midrule"]
    for _, r in d.iterrows():
        L.append(f"{r.asset_class} & {r.n} & {r.tickers} & {100 * r.ann_ret:.1f}\\% & {100 * r.ann_vol:.1f}\\% & "
                 f"{r.corr_within:.2f} & {r.corr_other:.2f}\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / f"{name}.tex").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    universe_table(RESULTS, "universe",
                   "Investment universe: 54 US-listed ETFs, daily data from 19 December 2007 to 18 September 2026. Annualised arithmetic "
                   "mean return and volatility are averages over the funds in each class; "
                   "$\\overline{\\text{corr}}_{\\text{in}}$ ($\\overline{\\text{corr}}_{\\text{out}}$) is the average "
                   "pairwise correlation of daily returns within the class (with funds of other classes).", "tab:universe")
    universe_table(Path(str(RESULTS) + "_stocks"), "universe_stocks",
                   "Second universe: 122 large US stocks by GICS sector, same sample period and statistics as in "
                   "Table~\\ref{tab:universe}. The stocks are today's large capitalisations with a complete price history, "
                   "which explains the high average returns (survivorship bias). Sectors follow GICS with one inconsistency: ADP is "
                   "labelled Information Technology, its sector before the March 2023 GICS revision (now Industrials), "
                   "while Mastercard follows the revised classification (Financials); the sector label enters the "
                   "networks only as an input dummy.", "tab:universe_stocks", place="!htbp", size="footnotesize")
    if UNIVERSE_NAME == "etf":  # the combined tables read both universes
        main_table()
        main_table(full=True)
        blocks_table()
        dm_table()
        perm_table_both()
        var_table_both()
    print("tables written to", OUT)
