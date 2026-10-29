"""Forest plot of the main comparisons: H1 (symmetric vs counterpart), C1 (symmetric vs Linear) and H2 (symmetric vs
best classical model), in-range portfolios, every horizon and both universes.

Effect: mean daily QLIKE differential (first model minus second) in points of the in-range EWMA loss, with the 95%
interval of the HLN-corrected Diebold-Mariano test (standard error = mean / DM statistic); filled markers: decided at
5% after the Holm correction within the family. Read from results*/main/tables (no new computation).
Writes paper/figures/forest.pdf.

Usage: python scripts/53_forest_plot.py   (both universes)
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("figs", Path(__file__).with_name("09_figures.py"))
figs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(figs)
plt = figs.plt

SYM = ["DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov"]
FAMILIES = [("H1", "hypotheses.csv", dict(hypothesis="H1", block="in-range"), "vs counterpart"),
            ("C1", "hypotheses_fourth.csv", dict(hypothesis="S1b", benchmark="Linear", block="in-range"), "vs Linear"),
            ("H2", "hypotheses.csv", dict(hypothesis="H2", block="in-range"), "vs best classical")]
UNI = [("ETF", figs.UNIVERSES[0][1], "o", "#0072B2"), ("Stocks", figs.UNIVERSES[1][1], "s", "#D55E00")]


def rows(root, fname, flt, h):
    d = pd.read_csv(root / "main" / "tables" / fname)
    for k, v in flt.items():
        d = d[d[k] == v]
    d = d[d.h == h].set_index("model").reindex(SYM)
    e = pd.read_csv(root / "main" / "tables" / f"losses_h{h}.csv").query("block == 'in-range' and model == 'EWMA'").qlike.iloc[0]
    md = 100 * d.mean_diff / e
    se = (md / d.dm).abs()
    return md, 1.96 * se, d.verdict == "more accurate", d.verdict == "less accurate"


if __name__ == "__main__":
    fig, axes = plt.subplots(3, 3, figsize=(figs.TEXTWIDTH, 5.6), sharey=True, sharex="row")
    ypos = np.arange(len(SYM))[::-1]
    for j, (lab, fname, flt, title) in enumerate(FAMILIES):
        for i, h in enumerate((1, 5, 22)):
            ax = axes[i, j]
            ax.axvline(0, color=figs.INK2, lw=0.8)
            for k, (u, root, mk, col) in enumerate(UNI):
                md, hw, good, bad = rows(root, fname, flt, h)
                y = ypos + (0.15 if k == 0 else -0.15)
                ax.errorbar(md, y, xerr=hw, fmt="none", ecolor=col, elinewidth=1.0, capsize=0)
                dec = (good | bad).to_numpy()
                ax.scatter(md[dec], y[dec], marker=mk, color=col, s=22, zorder=3, label=u if (i, j) == (0, 0) else None)
                ax.scatter(md[~dec], y[~dec], marker=mk, facecolor="white", edgecolor=col, s=22, zorder=3)
            ax.grid(axis="y", visible=False)
            if i == 0:
                ax.set_title(f"{lab}: {title}", loc="left")
            if j == 0:
                ax.set_ylabel(f"$h={h}$")
    axes[0, 0].set_yticks(ypos, [figs.lab(m) for m in SYM])
    fig.legend(loc="upper right", ncol=2, handletextpad=0.2, columnspacing=1.0, bbox_to_anchor=(1.0, 1.02))
    fig.supxlabel("difference (points of the EWMA loss; negative: the symmetric network is more accurate)", fontsize=9)
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    fig.savefig(figs.FIG / "forest.pdf")
    fig.savefig(figs.PREVIEW / "forest.png", dpi=150)
    print("written", figs.FIG / "forest.pdf")
