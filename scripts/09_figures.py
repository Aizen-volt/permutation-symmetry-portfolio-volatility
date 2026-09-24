"""Figures for the paper (vector PDF, drawn at the final printed width so that fonts are not scaled).

Style: Computer Modern like the text; the same four networks in the same validated colours in every figure (MLP,
PE-Net, Set Transformer, EQ-Cov; markers as secondary encoding); the best classical model of each panel as a black dashed
line labelled directly; one legend below the figure; ETF and stock universe as the two rows of one figure.
Run once (it reads both universes).
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pvol.evaluate as ev  # noqa: E402
from pvol.config import HORIZONS, RESULTS  # noqa: E402
from pvol.evaluate import CLASSICAL, group_index, load_run, qlike  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
PREVIEW = ROOT / "results" / "figure_previews"
FIG.mkdir(parents=True, exist_ok=True)
PREVIEW.mkdir(parents=True, exist_ok=True)
BASE = Path(str(RESULTS).replace("_stocks", ""))
UNIVERSES = [("ETF", BASE), ("Stocks", Path(str(BASE) + "_stocks"))]

TEXTWIDTH = 6.3  # inches (16 cm, the text width of the manuscript)
HIGHLIGHT = ["MLP", "PENet", "SetTransformer", "EQCov"]
COLOR = dict(zip(HIGHLIGHT, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
MARKER = dict(zip(HIGHLIGHT, ["o", "s", "^", "D"]))
INK, INK2, GRID, AXIS, SHADE = "#1a1a1a", "#4d4d4d", "#e6e6e6", "#9a9a9a", "#efefef"
LABEL = {"SetTransformer": "Set Transformer", "DeepSets": "Deep Sets", "EQCov": "EQ-Cov", "PENet": "PE-Net",
         "SetGRU": "Set-GRU", "HARX": "HAR-X", "GJR": "GJR-GARCH", "DCCNL": "DCC-NL"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False, "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.7, "lines.linewidth": 1.5, "lines.markersize": 4.5,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})

_spec = importlib.util.spec_from_file_location("hyp", Path(__file__).with_name("15_hypotheses.py"))
hyp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hyp)


def lab(m):
    return LABEL.get(m, m)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(PREVIEW / f"{name}.png", dpi=150)
    plt.close(fig)


def run(root, h):
    """load_run for an explicit universe (evaluate.load_run reads the module-level RESULTS)."""
    old, ev.RESULTS = ev.RESULTS, root
    try:
        return load_run("main", h)
    finally:
        ev.RESULTS = old


def best_classical(root, h):
    d = pd.read_csv(root / "main" / "tables" / f"losses_h{h}.csv")
    d = d[(d.block == "in-range") & d.model.isin(CLASSICAL)]
    return d.loc[d.qlike_rel_ewma.idxmin(), "model"]


def model_line(ax, x, y, m, **kw):
    ax.plot(x, y, color=COLOR[m], marker=MARKER[m], label=lab(m), markeredgecolor="white", markeredgewidth=0.6,
            zorder=3, **kw)


def bench_line(ax, x, y, name):
    ax.plot(x, y, color=INK, lw=1.2, ls=(0, (4, 2)), zorder=2, label="best classical model")


def panel_title(ax, u, h, b):
    ax.set_title(f"{u}, $h={h}$ (vs {lab(b)})", fontsize=8.5)


def bottom_legend(fig, axes, ncol=5):
    handles, labels = [], []
    for ax in np.ravel(axes):
        for hd, lb in zip(*ax.get_legend_handles_labels()):
            if lb not in labels:
                handles.append(hd)
                labels.append(lb)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.01), handlelength=2.4,
               columnspacing=1.6)




# ------------------------------------------------------------------------------------------------ figures
def fig_conditions():
    """QLIKE (in-range, h=5) of the four networks under the training conditions of the second experiment series."""
    conds = [("", "unlimited\nportfolios"), ("-P500", "pool of\n500"), ("-P50", "pool of\n50"), ("-RAW", "raw\ninputs")]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.55), sharey=True)
    for ax, (u, root) in zip(axes, UNIVERSES):
        R = run(root, 5)
        ci = hyp.cols_of(group_index(R["ports"]), "in-range")
        e = np.nanmean(qlike(R["y"], R["F"]["EWMA"])[:, ci])
        for m in HIGHLIGHT:
            pts = [(j, np.nanmean(qlike(R["y"], R["F"][m + s])[:, ci]) / e) for j, (s, _) in enumerate(conds)
                   if m + s in R["F"]]
            model_line(ax, [p[0] for p in pts], [p[1] for p in pts], m)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([c[1] for c in conds])
        ax.set_xlim(-0.3, len(conds) - 0.7)
        ax.set_title(f"{u} universe")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("QLIKE relative to EWMA")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    bottom_legend(fig, axes, ncol=4)
    save(fig, "conditions")


def fig_by_size():
    groups = {"N5": 5, "N10": 10, "N20": 20, "N30": 30, "N40": 40}
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 4.3))
    for r, (u, root) in enumerate(UNIVERSES):
        n_all = 54 if u == "ETF" else 122
        g = dict(groups, **{"EW-all": n_all})
        for c, h in enumerate(HORIZONS):
            ax = axes[r, c]
            d = pd.read_csv(root / "main" / "tables" / f"losses_h{h}.csv")
            d = d[d.block.isin(g)].assign(N=lambda x: x.block.map(g))
            for m in HIGHLIGHT:
                s = d[d.model == m].sort_values("N")
                model_line(ax, s.N.to_numpy(), s.qlike_rel_ewma.to_numpy(), m)
            b = best_classical(root, h)
            s = d[d.model == b].sort_values("N")
            bench_line(ax, s.N.to_numpy(), s.qlike_rel_ewma.to_numpy(), b)
            ax.axvspan(24, n_all * 1.15, color=SHADE, zorder=0, lw=0)
            ax.set_xscale("log")
            ticks = [5, 10, 20, 40, n_all]
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(v) for v in ticks])
            ax.minorticks_off()
            ax.set_xlim(4.3, n_all * 1.15)
            panel_title(ax, u, h, b)
            if r == 1:
                ax.set_xlabel("number of constituents $N$")
        axes[r, 0].set_ylabel("QLIKE relative to EWMA")
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.0, w_pad=1.2)
    bottom_legend(fig, axes)
    save(fig, "by_size")


def fig_yearly():
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 4.3), sharex=True)
    for r, (u, root) in enumerate(UNIVERSES):
        for c, h in enumerate(HORIZONS):
            ax = axes[r, c]
            d = pd.read_csv(root / "main" / "tables" / f"yearly_h{h}.csv")
            ref = d[d.model == "EWMA"].set_index("year").qlike
            d["rel"] = d.qlike / d.year.map(ref)
            for m in HIGHLIGHT:
                s = d[d.model == m].sort_values("year")
                model_line(ax, s.year.to_numpy(), s.rel.to_numpy(), m)
            b = best_classical(root, h)
            s = d[d.model == b].sort_values("year")
            bench_line(ax, s.year.to_numpy(), s.rel.to_numpy(), b)
            ax.axhline(1.0, color=AXIS, lw=0.7, zorder=1)
            ax.set_xticks([2020, 2022, 2024, 2026])
            ax.set_xlim(2019.7, 2026.3)
            panel_title(ax, u, h, b)
            if r == 1:
                ax.set_xlabel("test year")
        axes[r, 0].set_ylabel("QLIKE relative to EWMA")
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.0, w_pad=1.2)
    bottom_legend(fig, axes)
    save(fig, "yearly")


EPISODES = [("2020-02-20", "2020-05-29"), ("2025-03-31", "2025-05-30")]  # COVID crash, 2025 tariff shock


def fig_cumdiff():
    """Cumulative sum of the daily in-range QLIKE differential (network minus best classical model)."""
    from pvol.data import load_panel
    import pvol.data as pdata

    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 4.3), sharex=True)
    for r, (u, root) in enumerate(UNIVERSES):
        old = pdata.PROCESSED
        pdata.PROCESSED = ROOT / "data" / ("processed" if u == "ETF" else "processed_stocks")
        dates = pd.to_datetime(load_panel()["dates"])
        pdata.PROCESSED = old
        for c, h in enumerate(HORIZONS):
            ax = axes[r, c]
            R = run(root, h)
            ci = hyp.cols_of(group_index(R["ports"]), "in-range")
            t = dates[R["origins"]]
            b = best_classical(root, h)
            lb = np.nanmean(qlike(R["y"], R["F"][b])[:, ci], 1)
            for s, e in EPISODES:
                ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=SHADE, zorder=0, lw=0)
            for m in HIGHLIGHT:
                cs = np.cumsum(np.nanmean(qlike(R["y"], R["F"][m])[:, ci], 1) - lb)
                ax.plot(t, cs, color=COLOR[m], lw=1.2, label=lab(m), zorder=3)
            ax.axhline(0.0, color=AXIS, lw=0.7, zorder=1)
            panel_title(ax, u, h, b)
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[r, 0].set_ylabel("cumulative QLIKE difference")
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.0, w_pad=1.2)
    bottom_legend(fig, axes, ncol=4)
    save(fig, "cumdiff")


def fig_eqcov():
    """EQ-Cov (ETF, h=5, seed 0): mixing weights of the correlation forecast and its average pairwise correlation."""
    d = pd.read_csv(BASE / "main" / "tables" / "eqcov_diagnostics_h5.csv", parse_dates=["date"])
    parts = [("pi_ewma", "EWMA"), ("pi_corr66", "66-day"), ("pi_gram", "learned $AA^\\top$"), ("pi_corr22", "22-day")]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(TEXTWIDTH, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[1.25, 1]))
    a1.stackplot(d.date, *[d[c] for c, _ in parts], labels=[n for _, n in parts], colors=colors, lw=0)
    a1.set_ylim(0, 1)
    a1.set_ylabel("mixing weight $\\pi$")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, title=None)
    a2.plot(d.date, d.mean_corr_ewma, color=AXIS, lw=1.0, label="EWMA input")
    a2.plot(d.date, d.mean_corr_hat, color=INK, lw=1.0, label="EQ-Cov forecast")
    a2.set_ylabel("mean correlation")
    a2.legend(loc="upper right", ncol=2)
    for ax in (a1, a2):
        ax.grid(axis="x", visible=False)
        for y in range(2021, 2027):
            ax.axvline(pd.Timestamp(f"{y}-01-01"), color="white" if ax is a1 else GRID, lw=0.8, zorder=2)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout(h_pad=0.8)
    save(fig, "eqcov_diagnostics")


if __name__ == "__main__":
    fig_conditions()
    fig_by_size()
    fig_yearly()
    fig_cumdiff()
    fig_eqcov()
    print("figures written to", FIG)
