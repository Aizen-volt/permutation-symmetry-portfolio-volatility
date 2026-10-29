"""Scoreboard of all hypotheses for both universes, built from results*/main/tables/hypotheses*.csv.

Each cell counts the pairs (or comparisons) of the family whose verdict is 'more accurate', out of the pairs tested;
a trailing '-k' marks k comparisons with the opposite significant verdict ('less accurate').
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import RESULTS  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "paper" / "tables"
BASE = Path(str(RESULTS).replace("_stocks", ""))
UNIVERSES = [("ETF", BASE), ("Stocks", Path(str(BASE) + "_stocks"))]
E, R = "economic_tests.csv", "hypotheses_third.csv"

F4 = "hypotheses_fourth.csv"
# (label, file, filter[, suffix of the results directory that replaces the stock universe]); file None = section title
# main text: the families of the three research questions
ROWS = [
    ("RQ1: architecture against learned symmetry", None, None),
    ("A1: raw inputs", "hypotheses_extra.csv", dict(hypothesis="HE1")),
    ("A2: pool of 50 portfolios", "hypotheses_extra.csv", dict(hypothesis="HE2", block="in-range, pool 50")),
    ("A2: pool of 500 portfolios", "hypotheses_extra.csv", dict(hypothesis="HE2", block="in-range, pool 500")),
    ("RQ2: simpler routes to invariance", None, None),
    ("B1: pool of 50 vs shuffled counterpart", R, dict(hypothesis="R1a", block="in-range, pool 50, shuffled counterpart")),
    ("B1: pool of 500 vs shuffled counterpart", R,
     dict(hypothesis="R1a", block="in-range, pool 500, shuffled counterpart")),
    ("B2: shuffling improves the counterpart$^{a}$", R, dict(hypothesis="R1b")),
    ("B3: vs MLP-sorted, raw inputs", R, dict(hypothesis="R2", block="in-range, raw inputs")),
    ("B3: vs MLP-sorted, pool of 50", R, dict(hypothesis="R2", block="in-range, pool 50")),
    ("B3: vs MLP-sorted, pool of 500", R, dict(hypothesis="R2", block="in-range, pool 500")),
    ("B4: vs counterpart averaged over orderings, $h=1$", R, dict(hypothesis="R3", h=1)),
    ("B4: vs counterpart averaged over orderings, $h=5$", R, dict(hypothesis="R3", h=5)),
    ("B6: H1 on 54 stocks, $h=1$", R, dict(hypothesis="R5", h=1), "_stocks54"),
    ("B6: H1 on 54 stocks, $h=5$", R, dict(hypothesis="R5", h=5), "_stocks54"),
    ("K1: H1 with ten seeds, $h=1$", F4, dict(hypothesis="S2-H1", h=1)),
    ("K1: B4 with ten seeds, $h=1$", F4, dict(hypothesis="S2-R3", h=1)),
    ("K1: B6 with ten seeds, $h=1$", F4, dict(hypothesis="S2-R5", h=1), "_stocks54"),
    ("RQ3: flexibility and classical benchmarks", None, None),
    ("C2: vs invariant linear, raw inputs", "hypotheses_seventh.csv", dict(hypothesis="W1")),
]
# the main comparisons, shown in the main text as a forest plot (scripts/53_forest_plot.py)
ROWS_MAIN = [
    *[(f"H1: symmetric vs counterpart, $h={h}$", "hypotheses.csv", dict(hypothesis="H1", h=h, block="in-range"))
      for h in (1, 5, 22)],
    *[(f"C1: symmetric vs invariant linear, $h={h}$", F4, dict(hypothesis="S1b", h=h, benchmark="Linear"))
      for h in (1, 5, 22)],
    *[(f"H2: symmetric vs best classical, $h={h}$", "hypotheses.csv", dict(hypothesis="H2", h=h, block="in-range"))
      for h in (1, 5, 22)],
]
# secondary families (robustness section of the article)
ROWS_SUPP = [
    *ROWS_MAIN,
    *[(f"H2 (without the four added classical models), $h={h}$", "hypotheses_h2orig.csv",
       dict(hypothesis="H2", h=h, block="in-range")) for h in (1, 5, 22)],
    ("H1: minimum-variance portfolios", E, dict(hypothesis="H1-minvar")),
    *[(f"H1: volatility targeting, $h={h}$", E, dict(hypothesis="H1-voltarget", h=h)) for h in (1, 5, 22)],
    ("H2: minimum-variance portfolios", E, dict(hypothesis="H2-minvar")),
    *[(f"H2: volatility targeting, $h={h}$", E, dict(hypothesis="H2-voltarget", h=h)) for h in (1, 5, 22)],
    ("G1: advantage grows with $N$", "hypotheses.csv", dict(hypothesis="H3a", h=5)),
    ("G2: long--short weights, $h=5$", "hypotheses.csv", dict(hypothesis="H3b", h=5)),
    ("G3: unseen assets, $h=5$", "hypotheses.csv", dict(hypothesis="H3c", h=5)),
    ("G4: short training history", "hypotheses.csv", dict(hypothesis="H4")),
    ("A4: larger advantage than with summaries", "hypotheses_extra.csv", dict(hypothesis="HE1d")),
    ("A5: larger advantage with 50 than 500", "hypotheses_extra.csv", dict(hypothesis="HE2d")),
    ("A3: portfolios with $N>20$", "hypotheses_extra.csv", dict(hypothesis="HE3")),
    ("B5: Deep Sets + pairs vs MLP, raw inputs", R, dict(hypothesis="R4")),
    *[(f"C3: symmetric vs GBM, $h={h}$", F4, dict(hypothesis="S1b", h=h, benchmark="GBM")) for h in (1, 5, 22)],
]


IN_RANGE = ["N5", "N10", "N20", "N10-conc", "N10-EW"]


def ewma_loss(root, h, _cache={}):
    """Mean in-range QLIKE of EWMA (the unit of the effect sizes: one percentage point of the EWMA loss = 0.01 of it)."""
    key = (str(root), h)
    if key not in _cache:
        d = root / "main" / f"h{h}"
        if not (d / "targets.npy").exists():
            _cache[key] = None
            return None
        y = np.maximum(np.load(d / "targets.npy"), 1e-12)
        f = np.exp(np.load(d / "preds" / "EWMA.npy").astype(np.float64))
        ports = json.loads((root / "main" / "portfolios.json").read_text())
        cols = [j for j, p in enumerate(ports) if p["group"] in IN_RANGE]
        r = y[:, cols] / f[:, cols]
        _cache[key] = float(np.mean(r - np.log(r) - 1))
    return _cache[key]


def cell(root, fname, flt):
    """(verdict count, effect range, range of Holm-adjusted p-values): effect = mean daily loss differential (first model
    minus second) in percentage points of the in-range EWMA loss; negative favors the symmetric network. Economic
    comparisons: no effect size."""
    f = root / "main" / "tables" / fname
    if not f.exists():
        return "--", "", ""
    d = pd.read_csv(f)
    for k, v in flt.items():
        if k not in d.columns:
            return "--", "", ""
        d = d[d[k] == v]
    if not len(d):
        return "--", "", ""
    good = (d.verdict == "more accurate").sum()
    bad = (d.verdict == "less accurate").sum()
    count = f"{good}/{len(d)}" + (f" ($-${bad})" if bad else "")
    eff = ""
    if fname != E and "mean_diff" in d.columns:
        h = int(d.h.iloc[0]) if "h" in d.columns and d.h.notna().all() else 5
        e = ewma_loss(root, h)
        if e:
            v = 100 * d.mean_diff.to_numpy() / e
            lo, hi = v.min(), v.max()
            fmt = lambda x: f"${x:.1f}$".replace("$-", "$-")
            eff = fmt(lo) if abs(hi - lo) < 0.05 else f"{fmt(lo)} to {fmt(hi)}"
    pv = ""
    if "p_holm" in d.columns:
        fp = lambda x: "$<$0.001" if x < 0.001 else f"{x:.3f}"
        lo, hi = d.p_holm.min(), d.p_holm.max()
        pv = fp(lo) if fp(lo) == fp(hi) else f"{fp(lo)}--{fp(hi)}"
    return count, eff, pv


def write(rows, name, label, caption):
    L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}",
         "\\caption{" + caption + " Verdict: comparisons decided in favor of the symmetric network (Section~"
         "\\ref{sec:hypotheses}) out of those tested; ($-k$): $k$ decided against it; `--': not run. Effect: range of the "
         "mean QLIKE differential (first model minus second) in points of the EWMA loss; negative favors the symmetric "
         "network" + ("" if name == "scoreboard" else " (G1, G4: the stated condition); no effect size for the economic "
                      "comparisons") + ". $p_{\\text{Holm}}$: range of the Holm-adjusted $p$-values."
         + ("" if name != "scoreboard" else " $^{a}$Shuffled counterpart vs the same network in fixed order; here the "
            "verdict counts comparisons decided in favor of the shuffled counterpart, and negative effects favor it.")
         + "}\\label{" + label + "}",
         "\\begin{tabular}{lcrrcrr}\\toprule",
         " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{u}}}" for u, _ in UNIVERSES) + "\\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "Hypothesis & " + " & ".join("verdict & effect (pp) & $p_{\\text{Holm}}$" for _ in UNIVERSES)
         + "\\\\ \\midrule"]
    for i, (lbl, fname, flt, *alt) in enumerate(rows):
        if fname is None:
            L += ([] if i == 0 else ["\\midrule"]) + [f"\\multicolumn{{7}}{{l}}{{\\emph{{{lbl}}}}}\\\\"]
            continue
        roots = [root for _, root in UNIVERSES]
        if alt:  # run on a sub-universe only (R5): no ETF cell, stock cell from that directory
            roots = [None, Path(str(BASE) + alt[0])]
        cells = [cell(r, fname, flt) if r else ("--", "", "") for r in roots]
        L.append(f"{lbl} & " + " & ".join(f"{c} & {e} & {p}" for c, e, p in cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / f"{name}.tex").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[4:-2]))


if __name__ == "__main__":
    write(ROWS, "scoreboard", "tab:scoreboard",
          "Verdicts of the exploratory hypotheses of the three research questions (the confirmatory H1 and H2 and the "
          "comparison C1 with Linear are shown in Figure~\\ref{fig:forest}). B6 and its K1 repetition use 54 stocks.")
    write(ROWS_SUPP, "scoreboard_supp", "tab:scoreboard_supp",
          "Verdicts of the secondary families, both universes.")
