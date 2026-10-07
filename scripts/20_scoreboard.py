"""Scoreboard of all hypotheses for both universes, built from results*/main/tables/hypotheses*.csv.

Each cell counts the pairs (or comparisons) of the family whose verdict is 'more accurate', out of the pairs tested;
a trailing '-k' marks k comparisons with the opposite significant verdict ('less accurate').
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402

from pvol.config import RESULTS  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "paper" / "tables"
BASE = Path(str(RESULTS).replace("_stocks", ""))
UNIVERSES = [("ETF", BASE), ("Stocks", Path(str(BASE) + "_stocks"))]
E, R = "economic_tests.csv", "hypotheses_review.csv"

# (label, file, filter[, suffix of the results directory that replaces the stock universe]); file None = section title
ROWS = [
    ("H1: symmetric vs counterpart, $h=1$", "hypotheses.csv", dict(hypothesis="H1", h=1, block="in-range")),
    ("H1: symmetric vs counterpart, $h=5$", "hypotheses.csv", dict(hypothesis="H1", h=5, block="in-range")),
    ("H1: symmetric vs counterpart, $h=22$", "hypotheses.csv", dict(hypothesis="H1", h=22, block="in-range")),
    ("H1: minimum-variance portfolios", E, dict(hypothesis="H1-minvar")),
    *[(f"H1: volatility targeting, $h={h}$", E, dict(hypothesis="H1-voltarget", h=h)) for h in (1, 5, 22)],
    ("H2: symmetric vs best classical, $h=1$", "hypotheses.csv", dict(hypothesis="H2", h=1, block="in-range")),
    ("H2: symmetric vs best classical, $h=5$", "hypotheses.csv", dict(hypothesis="H2", h=5, block="in-range")),
    ("H2: symmetric vs best classical, $h=22$", "hypotheses.csv", dict(hypothesis="H2", h=22, block="in-range")),
    ("H2: minimum-variance portfolios", E, dict(hypothesis="H2-minvar")),
    *[(f"H2: volatility targeting, $h={h}$", E, dict(hypothesis="H2-voltarget", h=h)) for h in (1, 5, 22)],
    ("H3a: advantage grows with $N$", "hypotheses.csv", dict(hypothesis="H3a", h=5)),
    ("H3b: long--short weights, $h=5$", "hypotheses.csv", dict(hypothesis="H3b", h=5)),
    ("H3c: unseen assets, $h=5$", "hypotheses.csv", dict(hypothesis="H3c", h=5)),
    ("H4: short training history", "hypotheses.csv", dict(hypothesis="H4")),
    ("Second series", None, None),
    ("HE1: raw inputs", "hypotheses_extra.csv", dict(hypothesis="HE1")),
    ("HE1d: larger advantage than with summaries", "hypotheses_extra.csv", dict(hypothesis="HE1d")),
    ("HE2: pool of 50 portfolios", "hypotheses_extra.csv", dict(hypothesis="HE2", block="in-range, pool 50")),
    ("HE2: pool of 500 portfolios", "hypotheses_extra.csv", dict(hypothesis="HE2", block="in-range, pool 500")),
    ("HE2d: larger advantage with 50 than 500", "hypotheses_extra.csv", dict(hypothesis="HE2d")),
    ("HE3: portfolios with $N>20$", "hypotheses_extra.csv", dict(hypothesis="HE3")),
    ("Third series (exploratory)", None, None),
    ("R1a: pool of 50 vs shuffled counterpart", R, dict(hypothesis="R1a", block="in-range, pool 50, shuffled counterpart")),
    ("R1a: pool of 500 vs shuffled counterpart", R,
     dict(hypothesis="R1a", block="in-range, pool 500, shuffled counterpart")),
    ("R1b: shuffling improves the counterpart$^{a}$", R, dict(hypothesis="R1b")),
    ("R2: vs MLP-sorted, raw inputs", R, dict(hypothesis="R2", block="in-range, raw inputs")),
    ("R2: vs MLP-sorted, pool of 50", R, dict(hypothesis="R2", block="in-range, pool 50")),
    ("R2: vs MLP-sorted, pool of 500", R, dict(hypothesis="R2", block="in-range, pool 500")),
    ("R3: vs counterpart averaged over orderings, $h=1$", R, dict(hypothesis="R3", h=1)),
    ("R3: vs counterpart averaged over orderings, $h=5$", R, dict(hypothesis="R3", h=5)),
    ("R4: Deep Sets + pairs vs MLP, raw inputs", R, dict(hypothesis="R4")),
    ("R5: H1 on 54 stocks, $h=1$", R, dict(hypothesis="R5", h=1), "_stocks54"),
    ("R5: H1 on 54 stocks, $h=5$", R, dict(hypothesis="R5", h=5), "_stocks54"),
]


def cell(root, fname, flt):
    f = root / "main" / "tables" / fname
    if not f.exists():
        return "--"
    d = pd.read_csv(f)
    for k, v in flt.items():
        if k not in d.columns:
            return "--"
        d = d[d[k] == v]
    if not len(d):
        return "--"
    good = (d.verdict == "more accurate").sum()
    bad = (d.verdict == "less accurate").sum()
    return f"{good}/{len(d)}" + (f" ($-${bad})" if bad else "")


if __name__ == "__main__":
    L = ["\\begin{table}[t]\\centering\\small",
         "\\caption{Verdicts for every hypothesis and both universes. Each cell reports how many of the tested "
         "comparisons of the family are decided in favour of the symmetry-respecting network (QLIKE lower and the "
         "Diebold--Mariano test significant at 5\\% after the Holm correction), out of the comparisons tested; "
         "$-k$ marks $k$ comparisons decided significantly against it. `--' means the family was not run for that "
         "universe. The third series was designed after all other results had been seen; R5 is run on a subsample of "
         "54 stocks and reported in the stock column. $^{a}$Comparisons in which the counterpart trained on the pool in "
         "random orders is more accurate than the same network in a fixed order (MLP and GRU, pools of 50 and 500)."
         "}\\label{tab:scoreboard}",
         "\\begin{tabular}{lcc}\\toprule",
         "Hypothesis & " + " & ".join(u for u, _ in UNIVERSES) + "\\\\ \\midrule"]
    for label, fname, flt, *alt in ROWS:
        if fname is None:
            L += ["\\midrule", f"\\multicolumn{{3}}{{l}}{{\\emph{{{label}}}}}\\\\"]
            continue
        roots = [root for _, root in UNIVERSES]
        if alt:  # run on a sub-universe only (R5): no ETF cell, stock cell from that directory
            roots = [None, Path(str(BASE) + alt[0])]
        L.append(f"{label} & " + " & ".join(cell(r, fname, flt) if r else "--" for r in roots) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / "scoreboard.tex").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[4:-2]))
