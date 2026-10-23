"""LaTeX tables of the fourth-series robustness analyses (scripts/28_fourth_analyses.py), both universes.

- paper/tables/fourth_losses.tex: in-range loss relative to EWMA under Patton's b = -1 loss and under QLIKE without 2020;
- paper/tables/fourth_tests.tex: verdict counts of H1 and H2 under the Diebold-Mariano test on the full test sample, on
  the test origins outside 2020 and under the conditional Giacomini-White test; share of seed pairs with a lower loss of
  the symmetric network.
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
       "SetGRU": "Set-GRU", "GJR": "GJR-GARCH"}
MODELS = ["EWMA", "HAR", "LogHAR-X", "HEAVY", "GJR", "DCC", "Linear", "GBM", "MLP", "GRU", "DeepSets", "PENet",
          "SetTransformer", "SetGRU", "EQCov"]
H = (1, 5, 22)


def tab(root, name):
    return pd.read_csv(root / "main" / "tables" / f"{name}.csv")


def losses():
    L = ["\\begin{table}[t]\\centering\\small",
         "\\caption{In-range loss relative to EWMA under Patton's loss with $b=-1$ (Panel A) and under QLIKE on the test "
         "origins outside 2020 (Panel B). Patton's loss with $b=-1$ is, like QLIKE, robust to noise in the volatility "
         "proxy but penalises over- and under-prediction differently. Bold: lowest loss among the models shown.}\\label{tab:fourth_losses}",
         "\\begin{tabular}{l" + "rrr" * len(UNIVERSES) + "}\\toprule",
         " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{u}}}" for u, _ in UNIVERSES) + "\\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "Model & " + " & ".join(f"$h={h}$" for _ in UNIVERSES for h in H) + "\\\\"]
    D = {u: tab(root, "losses_variants") for u, root in UNIVERSES}
    for panel, loss in (("A: Patton, $b=-1$", "Patton b=-1"), ("B: QLIKE, excluding 2020", "QLIKE excl. 2020")):
        L += ["\\midrule", f"\\multicolumn{{7}}{{l}}{{\\emph{{Panel {panel}}}}}\\\\"]
        for m in MODELS:
            cells = []
            for u, _ in UNIVERSES:
                d = D[u][D[u].loss == loss].drop_duplicates(["model", "h"]).set_index(["model", "h"]).rel_ewma.sort_index()
                col = [d.get((m, h)) for h in H]
                best = [d.xs(h, level="h").reindex(MODELS[1:]).min() for h in H]
                cells += ["--" if v is None else (f"\\textbf{{{v:.3f}}}" if abs(v - b) < 5e-4 else f"{v:.3f}")
                          for v, b in zip(col, best)]
            L.append(f"{LAB.get(m, m)} & " + " & ".join(cells) + "\\\\")
            if m in ("DCC", "GBM"):
                L.append("\\addlinespace")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / "fourth_losses.tex").write_text("\n".join(L), encoding="utf-8")


def count(d):
    good, bad = (d.verdict == "more accurate").sum(), (d.verdict == "less accurate").sum()
    return f"{good}/{len(d)}" + (f" ($-${bad})" if bad else "") if len(d) else "--"


def tests():
    L = ["\\begin{table}[t]\\centering\\small",
         "\\caption{H1 and H2 (in-range, best of the fourteen classical models) under alternative tests. Counts of the "
         "comparisons decided in favour of the symmetric network out of five; $-k$ marks $k$ comparisons decided "
         "against it (Holm correction within each family). DM: Diebold--Mariano test of the main text; excl.\\ 2020: "
         "the same test on the test origins outside 2020, with the best classical model chosen on that sample; GW: "
         "conditional Giacomini--White test with instruments 1 and the lagged loss differential, with the sign of the "
         "mean differential deciding the direction. Seed pairs: share of the pairs (seed of the symmetric network, "
         "seed of the counterpart) in which the symmetric network has the lower in-range QLIKE, range over the five "
         "pairs; in the stock universe at $h=1$ ten seeds each (100 pairs), otherwise three (9 pairs).}"
         "\\label{tab:fourth_tests}",
         "\\begin{tabular}{ll" + "c" * 2 * len(UNIVERSES) + "}\\toprule",
         " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{u}}}" for u, _ in UNIVERSES) + "\\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
         "Test & $h$ & " + " & ".join("H1 & H2" for _ in UNIVERSES) + "\\\\ \\midrule"]
    T = {u: (tab(r, "hypotheses"), tab(r, "hypotheses_ex2020"), tab(r, "gw"), tab(r, "seed_signs")) for u, r in UNIVERSES}
    for label, k in (("DM", 0), ("DM, excl.\\ 2020", 1), ("GW", 2)):
        for i, h in enumerate(H):
            cells = []
            for u, _ in UNIVERSES:
                d = T[u][k]
                d = d[d.h == h]
                if k == 0:
                    d = d[d.block == "in-range"]
                cells += [count(d[d.hypothesis == f]) for f in ("H1", "H2")]
            L.append(f"{label if i == 0 else ''} & {h} & " + " & ".join(cells) + "\\\\")
        L.append("\\addlinespace")
    for i, h in enumerate(H):
        cells = []
        for u, _ in UNIVERSES:
            s = T[u][3]
            s = s[s.h == h].share_lower
            cells += [f"{s.min():.2f}--{s.max():.2f}" if s.max() - s.min() > 5e-3 else f"{s.min():.2f}", ""]
        L.append(f"{'Seed pairs' if i == 0 else ''} & {h} & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}", "\\end{table}"]
    (OUT / "fourth_tests.tex").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    losses()
    tests()
    print((OUT / "fourth_tests.tex").read_text(encoding="utf-8"))
