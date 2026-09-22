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
       "SetGRU": "Set-GRU", "HARX": "HAR-X", "GJR": "GJR-GARCH"}
SYM = {"PortMLP": "agg.", "MLP": "--", "MLP-sorted": "canon.", "GRU": "--", "DeepSets": "arch.", "PENet": "arch.",
       "SetTransformer": "arch.", "SetGRU": "arch.", "EQCov": "arch."}
BLOCKS = [("in-range", "In-range"), ("larger-N", "Larger $N$"), ("long-short", "Long--short"), ("structured", "Structured")]


def lab(m):
    return LAB.get(m, m)


def stars(p):
    return "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.10 else ""


def main_table(exp="main"):
    """QLIKE relative to EWMA by horizon and portfolio block; best per column in bold; MCS members marked with dagger."""
    rows = {}
    for h in HORIZONS:
        f = RESULTS / exp / "tables" / f"losses_h{h}.csv"
        t = RESULTS / exp / "tables" / f"tests_h{h}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        tests = pd.read_csv(t)
        for b, _ in BLOCKS:
            db = d[d.block == b].set_index("model")
            mcs = tests[(tests.block == b) & (tests.benchmark == "MCS")].set_index("model").p_value
            best = db.qlike_rel_ewma.min()
            for m, v in db.qlike_rel_ewma.items():
                cell = f"{v:.3f}"
                if np.isclose(v, best):
                    cell = f"\\textbf{{{cell}}}"
                if m in mcs.index and mcs[m] > 0.10:
                    cell += "$^\\dagger$"
                rows.setdefault(m, {})[(h, b)] = cell
    models = [m for m in CLASSICAL + NEURAL if m in rows]
    cols = [(h, b) for h in HORIZONS for b, _ in BLOCKS]
    L = ["\\begin{table}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}",
         "\\caption{Out-of-sample QLIKE relative to EWMA (values below 1 beat RiskMetrics), test years 2020--2026, "
         "averaged over portfolios and forecast origins. Bold: best model in the column. $^\\dagger$: model belongs to the "
         "90\\% Model Confidence Set of the column. Symmetry: agg.\\ = invariant because only portfolio aggregates are used; "
         "canon.\\ = canonicalisation by sorting; arch.\\ = architectural; -- = none. Neural forecasts are seed ensembles.}"
         "\label{tab:main" + SUF.replace("_", ":") + "}",
         "\\begin{tabular}{ll" + "r" * len(cols) + "}\\toprule",
         " & & " + " & ".join(f"\\multicolumn{{{len(BLOCKS)}}}{{c}}{{$h={h}$}}" for h in HORIZONS) + "\\\\",
         " ".join(f"\\cmidrule(lr){{{3 + i * len(BLOCKS)}-{2 + (i + 1) * len(BLOCKS)}}}" for i in range(len(HORIZONS))),
         "Model & Sym. & " + " & ".join(lbl for _ in HORIZONS for _, lbl in BLOCKS) + "\\\\ \\midrule"]
    for m in models:
        if m == NEURAL[0]:
            L.append("\\midrule")
        L.append(f"{lab(m)} & {SYM.get(m, '')} & " + " & ".join(rows[m].get(c, "n/a") for c in cols) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / f"main_{exp}{SUF}.tex").write_text("\n".join(L), encoding="utf-8")


def dm_table(exp="main", block="in-range"):
    """Diebold-Mariano statistics for the matched pairs (negative = first model more accurate)."""
    pairs = [("DeepSets", "MLP"), ("PENet", "MLP"), ("SetTransformer", "MLP"), ("EQCov", "MLP"), ("MLP-sorted", "MLP"),
             ("SetGRU", "GRU"), ("DeepSets", "PortMLP"), ("EQCov", "DCC")]
    cells = {}
    blocks = [b for b, _ in BLOCKS] if exp == "main" else [block]
    for h in HORIZONS:
        t = RESULTS / exp / "tables" / f"tests_h{h}.csv"
        if not t.exists():
            continue
        d = pd.read_csv(t)
        for b in blocks:
            for a, c in pairs:
                r = d[(d.block == b) & (d.model == a) & (d.benchmark == c)]
                if len(r):
                    r = r.iloc[0]
                    cells[(a, c, h, b)] = f"${r.dm_stat:.2f}{stars(r.p_value)}$"
    L = ["\\begin{table}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}",
         "\\caption{Diebold--Mariano tests of equal QLIKE for pairs of networks that differ only in whether they respect "
         "the permutation symmetry (HLN-corrected statistics on daily cross-sectional average loss differentials). "
         "Negative values favour the first model. $^{*}$, $^{**}$, $^{***}$: significance at 10\\%, 5\\%, 1\\%.}"
         "\label{tab:dm_" + exp + SUF + "}",
         "\\begin{tabular}{l" + "r" * (len(HORIZONS) * len(blocks)) + "}\\toprule",
         " & " + " & ".join(f"\\multicolumn{{{len(blocks)}}}{{c}}{{$h={h}$}}" for h in HORIZONS) + "\\\\"]
    if len(blocks) > 1:
        L.append("Pair & " + " & ".join(lbl for _ in HORIZONS for b, lbl in BLOCKS) + "\\\\ \\midrule")
    else:
        L.append("Pair & " + " & ".join("" for _ in HORIZONS) + "\\\\ \\midrule")
    for a, c in pairs:
        vals = [cells.get((a, c, h, b), "n/a") for h in HORIZONS for b in blocks]
        if all(v == "n/a" for v in vals):
            continue
        L.append(f"{lab(a)} vs {lab(c)} & " + " & ".join(vals) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / f"dm_{exp}{SUF}.tex").write_text("\n".join(L), encoding="utf-8")


def perm_table(exp="main"):
    rows = {}
    for h in HORIZONS:
        f = RESULTS / exp / "tables" / f"perm_h{h}.csv"
        if f.exists():
            for _, r in pd.read_csv(f).iterrows():
                rows.setdefault(r.model, {})[h] = (r.median_range_logvar, r.p99_range_logvar, r.qlike_range_over_orderings)
    models = [m for m in NEURAL if m in rows]
    L = ["\\begin{table}[t]\\centering\\small",
         "\\caption{Permutation consistency: the same 100 ten-asset portfolios are forecast with their constituents listed "
         "in five different orders. Median and 99th percentile (over portfolios and origins) of the range of the log-variance "
         "forecast across orderings, and range of the resulting average QLIKE. Averages over seeds.}\label{tab:perm_" + exp + SUF + "}",
         "\\begin{tabular}{l" + "rrr" * len(HORIZONS) + "}\\toprule",
         " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{$h={h}$}}" for h in HORIZONS) + "\\\\",
         "Model & " + " & ".join("med. & p99 & $\\Delta$QL" for _ in HORIZONS) + "\\\\ \\midrule"]
    for m in models:
        vals = []
        for h in HORIZONS:
            v = rows[m].get(h)
            vals += ["n/a"] * 3 if v is None else [f"{v[0]:.3f}", f"{v[1]:.3f}", f"{v[2]:.4f}"]
        L.append(f"{lab(m)} & " + " & ".join(vals) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / f"perm_{exp}{SUF}.tex").write_text("\n".join(L), encoding="utf-8")


def var_table(exp="main"):
    f = RESULTS / exp / "tables" / "var_h1.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    models = [m for m in CLASSICAL + NEURAL if m in set(d.model)]
    L = ["\\begin{table}[t]\\centering\\small",
         "\\caption{Backtest of one-day Gaussian value-at-risk for the in-range portfolios, 2020--2026. Hit rate and share "
         "of portfolios for which the unconditional coverage \\citep{kupiec1995} and conditional coverage "
         "\citep{christoffersen1998} tests reject at 5\%.}\label{tab:var" + SUF.replace("_", ":") + "}",
         "\\begin{tabular}{lrrrrrr}\\toprule",
         " & \\multicolumn{3}{c}{VaR 1\\%} & \\multicolumn{3}{c}{VaR 5\\%}\\\\ \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "Model & hit & rej.\\ UC & rej.\\ CC & hit & rej.\\ UC & rej.\\ CC\\\\ \\midrule"]
    for m in models:
        a = d[(d.model == m) & (d.alpha == 0.01)].iloc[0]
        b = d[(d.model == m) & (d.alpha == 0.05)].iloc[0]
        L.append(f"{lab(m)} & {100 * a.hit_rate:.2f}\\% & {100 * a.rej_uc:.0f}\\% & {100 * a.rej_cc:.0f}\\% & "
                 f"{100 * b.hit_rate:.2f}\\% & {100 * b.rej_uc:.0f}\\% & {100 * b.rej_cc:.0f}\\%\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / f"var_{exp}{SUF}.tex").write_text("\n".join(L), encoding="utf-8")


def universe_table(root, name, caption, label):
    f = root / "descriptives" / "universe_stats.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    L = ["\\begin{table}[t]\\centering\\small", f"\\caption{{{caption}}}\\label{{{label}}}",
         "\\begin{tabular}{p{2.8cm}rp{5.4cm}rrrr}\\toprule",
         "Class & $n$ & Tickers & Ret. & Vol. & $\\bar\\rho_{\\text{within}}$ & $\\bar\\rho_{\\text{other}}$\\\\ \\midrule"]
    for _, r in d.iterrows():
        L.append(f"{r.asset_class} & {r.n} & {r.tickers} & {100 * r.ann_ret:.1f}\\% & {100 * r.ann_vol:.1f}\\% & "
                 f"{r.corr_within:.2f} & {r.corr_other:.2f}\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / f"{name}.tex").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    universe_table(RESULTS, "universe",
                   "Investment universe: 54 US-listed ETFs, daily data 19 December 2007 -- 18 September 2026. Annualised mean "
                   "return and volatility are averages over the funds in each class; $\\bar\\rho_{\\text{within}}$ "
                   "($\\bar\\rho_{\\text{other}}$) is the average pairwise correlation of daily returns within the class "
                   "(with funds of other classes).", "tab:universe")
    universe_table(Path(str(RESULTS) + "_stocks"), "universe_stocks",
                   "Second universe: 122 large US stocks by GICS sector, same sample period and statistics as in "
                   "Table~\\ref{tab:universe}. The stocks are today's large capitalisations with a complete price history, "
                   "which explains the high average returns (survivorship bias).", "tab:universe_stocks")
    for exp in sys.argv[1:] or ["main", "heldout"]:
        if (RESULTS / exp / "tables").exists():
            if exp == "main":
                main_table(exp)
                var_table(exp)
            dm_table(exp, block="unseen-assets" if exp == "heldout" else "in-range")
            perm_table(exp)
    print("tables written to", OUT)
