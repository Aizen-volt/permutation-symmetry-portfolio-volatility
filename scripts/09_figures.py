"""Figures for the paper (vector PDF). Palette: validated categorical slots (blue, orange, aqua, yellow) for at most four
highlighted series per panel, markers as secondary encoding, remaining models in recessive gray; one y-axis per panel."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import HORIZONS, RESULTS, UNIVERSE_NAME  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
PREVIEW = Path(__file__).resolve().parents[1] / "results" / "figure_previews"
PREVIEW.mkdir(parents=True, exist_ok=True)


SUF = "" if UNIVERSE_NAME == "etf" else f"_{UNIVERSE_NAME}"


def save(fig, name):
    name = name + SUF
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(PREVIEW / f"{name}.png", dpi=130)
    plt.close(fig)

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
GRAY = "#b9b8b2"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})

LABEL = {"SetTransformer": "Set Transformer", "DeepSets": "Deep Sets", "EQCov": "EQ-Cov", "PENet": "PE-Net",
         "SetGRU": "Set-GRU", "HARX": "HAR-X", "MLP-sorted": "MLP-sorted"}


def lab(m):
    return LABEL.get(m, m)


OMIT = {"LW", "HIST", "EWMANLS", "EWMA"}  # static / far-off benchmarks stay in the tables, not in the figures


def highlighted_lines(ax, df, x, y, highlight, others, xlabels=None):
    others = [m for m in others if m not in OMIT]
    for m in others:
        d = df[df.model == m].sort_values(x)
        if len(d):
            ax.plot(d[x], d[y], color=GRAY, lw=0.9, zorder=1)
    for k, m in enumerate(highlight):
        d = df[df.model == m].sort_values(x)
        if not len(d):
            continue
        ax.plot(d[x], d[y], color=SLOTS[k], marker=MARKERS[k], label=lab(m), zorder=3,
                markeredgecolor="white", markeredgewidth=0.8)
    ax.axhline(1.0, color=AXIS, lw=0.8, zorder=0)
    shown = df[df.model.isin(list(highlight) + others)][y]
    lo, hi = shown.min(), np.nanpercentile(shown, 99)
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.08 * (hi - lo))


def fig_yearly(exp="main", highlight=("SetGRU", "GRU", "DeepSets", "MLP"), bench=("HARX", "DCC")):
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(7.2, 2.5), sharey=False)
    for ax, h in zip(axes, HORIZONS):
        f = RESULTS / exp / "tables" / f"yearly_h{h}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        ref = d[d.model == "EWMA"].set_index("year").qlike
        d["rel"] = d.qlike / d.year.map(ref)
        others = [m for m in d.model.unique() if m not in highlight and m != "EWMA"]
        highlighted_lines(ax, d, "year", "rel", list(highlight), others)
        ax.set_title(f"h = {h}", color=INK, fontsize=9)
        ax.set_xticks(sorted(d.year.unique()))
        ax.set_xticklabels([str(y)[2:] for y in sorted(d.year.unique())])
        ax.set_xlabel("test year (20xx)")
    axes[0].set_ylabel("QLIKE relative to EWMA")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7.5)
    save(fig, f"yearly_{exp}")


def fig_by_size(exp="main", highlight=("SetGRU", "GRU", "DeepSets", "MLP")):
    from pvol.portfolios import load as load_ports

    n_all = max(len(q["ids"]) for q in load_ports(RESULTS / exp / "portfolios.json"))
    groups = {"N5": 5, "N10": 10, "N20": 20, "N30": 30, "N40": 40, "EW-all": n_all}
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(7.2, 2.5))
    for ax, h in zip(axes, HORIZONS):
        f = RESULTS / exp / "tables" / f"losses_h{h}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[d.block.isin(groups)].copy()
        d["N"] = d.block.map(groups)
        others = [m for m in d.model.unique() if m not in highlight and m != "EWMA"]
        highlighted_lines(ax, d, "N", "qlike_rel_ewma", list(highlight), others)
        ax.axvspan(20.5, 56, color="#f0efec", zorder=0, lw=0)
        ax.set_xscale("log")
        ax.set_xticks(list(groups.values()))
        ax.set_xticklabels([str(v) for v in groups.values()])
        ax.set_title(f"h = {h}", color=INK, fontsize=9)
        ax.set_xlabel("number of assets N")
    axes[0].set_ylabel("QLIKE relative to EWMA")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7.5)
    save(fig, f"by_size_{exp}")


def fig_permutation(exp="main", h=5):
    f = RESULTS / exp / "tables" / f"perm_h{h}.csv"
    if not f.exists():
        return
    d = pd.read_csv(f).sort_values("median_range_logvar")
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    y = np.arange(len(d))
    ax.barh(y, d.median_range_logvar * 100, color=SLOTS[0], height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([lab(m) for m in d.model])
    for yi, v in zip(y, d.median_range_logvar * 100):
        ax.text(v, yi, f" {v:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_xlabel("median range of log-variance forecast\nacross 5 orderings (×100)")
    ax.grid(axis="y", visible=False)
    save(fig, f"permutation_{exp}_h{h}")


def fig_conditions(highlight=("MLP", "PENet", "SetTransformer", "EQCov")):
    """QLIKE (relative to EWMA, in-range portfolios, h=5) of the MLP and three symmetric networks under the training
    conditions of the second experiment series; one panel per universe (missing conditions are skipped)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("hyp", Path(__file__).with_name("15_hypotheses.py"))
    hyp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hyp)
    from pvol.evaluate import group_index, load_run, qlike

    conds = [("", "unlimited\nportfolios"), ("-P500", "pool of\n500"), ("-P50", "pool of\n50"), ("-RAW", "raw\ninputs")]
    base = Path(str(RESULTS).replace("_stocks", ""))  # this figure always shows both universes
    roots = [("ETF universe", base), ("Stock universe", Path(str(base) + "_stocks"))]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, (title, root) in zip(axes, roots):
        f = root / "main" / "h5" / "targets.npy"
        if not f.exists():
            continue
        import pvol.evaluate as ev
        old = ev.RESULTS
        ev.RESULTS = root
        R = load_run("main", 5)
        ev.RESULTS = old
        y = R["y"]
        ci = hyp.cols_of(group_index(R["ports"]), "in-range")
        e = np.nanmean(qlike(y, R["F"]["EWMA"])[:, ci])
        xs = []
        for k, m in enumerate(highlight):
            vals = []
            for j, (suf, _) in enumerate(conds):
                if m + suf in R["F"]:
                    vals.append((j, np.nanmean(qlike(y, R["F"][m + suf])[:, ci]) / e))
            if vals:
                xs += [v[0] for v in vals]
                ax.plot([v[0] for v in vals], [v[1] for v in vals], color=SLOTS[k], marker=MARKERS[k], label=lab(m),
                        markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([c[1] for c in conds], fontsize=7.5)
        ax.set_title(title, color=INK, fontsize=9)
        ax.set_xlim(-0.3, len(conds) - 0.7)
    axes[0].set_ylabel("QLIKE relative to EWMA (h = 5)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7.5)
    fig.savefig(FIG / "conditions.pdf")
    fig.savefig(PREVIEW / "conditions.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    if UNIVERSE_NAME == "etf":
        fig_conditions()
    fig_yearly()
    fig_by_size()
    for h in HORIZONS:
        fig_permutation(h=h)
    print("figures written to", FIG)
