"""Figures for the paper (vector PDF, drawn at the final printed width so that fonts are not scaled).

Style: text set by LaTeX (usetex: STIX like the manuscript in the Elsevier CAS template, Type 1 fonts with correct text encoding); the
same networks in the same Okabe-Ito colours in every figure (markers or line styles as secondary encoding); the best
classical model of each panel as a black dashed line, named in the panel title; one legend below the figure; ETF and stock
universe as the two rows of one figure.
Run once (it reads both universes).
"""
import importlib.util
import os
import sys
from pathlib import Path

os.environ["PATH"] = r"C:\texlive\2026\bin\windows" + os.pathsep + os.environ["PATH"]  # LaTeX for usetex

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
UNIVERSES = [("ETF", BASE), ("Stock", Path(str(BASE) + "_stocks"))]

TEXTWIDTH = 6.84  # inches (494.5 pt, the text width of the cas-dc template)
HIGHLIGHT = ["MLP", "PENet", "SetTransformer", "EQCov"]
NETS = ["MLP", "MLP-sorted", "GRU", "DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov"]  # read the constituents
COLOR = {"MLP": "#0072B2", "PENet": "#D55E00", "SetTransformer": "#009E73", "EQCov": "#CC79A7", "DeepSets": "#E69F00"}
MARKER = {"MLP": "o", "PENet": "s", "SetTransformer": "^", "EQCov": "D", "DeepSets": "v"}
DASH = {"MLP": "-", "PENet": (0, (5, 1.5)), "SetTransformer": (0, (1.5, 1.2)), "EQCov": (0, (6, 1.5, 1.5, 1.5))}
INK, INK2, GRID, AXIS, SHADE = "#1a1a1a", "#4d4d4d", "#e6e6e6", "#9a9a9a", "#efefef"
LABEL = {"SetTransformer": "Set Transformer", "DeepSets": "Deep Sets", "EQCov": "EQ-Cov", "PENet": "PE-Net",
         "SetGRU": "Set-GRU", "HARX": "HAR-X", "GJR": "GJR-GARCH", "DCCNL": "DCC-NL"}

plt.rcParams.update({
    "text.usetex": True, "text.latex.preamble": r"\usepackage{stix}", "font.family": "serif", "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.7, "lines.linewidth": 1.6, "lines.markersize": 5,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
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




# figures
# One message per figure, few large panels, text at 9-10 pt at the printed size.
UCOL = {"ETF": "#0072B2", "Stock": "#D55E00"}


def _levels(root):
    """QLIKE relative to EWMA (in-range, h=5) of every network variant, from the evaluation tables."""
    out = {}
    for f in ("extra_levels.csv", "review_levels.csv"):
        p = root / "main" / "tables" / f
        if p.exists():
            d = pd.read_csv(p)
            if "h" in d.columns:
                d = d[d.h == 5]
            col = "qlike_rel_ewma_in_range" if "qlike_rel_ewma_in_range" in d.columns else "qlike_rel_ewma"
            out.update(dict(zip(d.model, d[col])))
    return out


def fig_conditions():
    """How much accuracy each network loses when the aids of the main design are removed (second and third series)."""
    panels = [
        ("Fixed pool of 50 training portfolios", "-P50",
         [("MLP", "MLP", "MLP-P50"), ("MLP, shuffled order", "MLP", "MLP-P50s"), ("MLP-sorted", "MLP-sorted", "MLP-sorted-P50"),
          ("Deep Sets", "DeepSets", "DeepSets-P50"), ("PE-Net", "PENet", "PENet-P50"),
          ("Set Transformer", "SetTransformer", "SetTransformer-P50"), ("EQ-Cov", "EQCov", "EQCov-P50")]),
        ("Raw inputs (no portfolio summaries)", "-RAW",
         [("MLP", "MLP", "MLP-RAW"), ("MLP-sorted", "MLP-sorted", "MLP-sorted-RAW"), ("Deep Sets", "DeepSets", "DeepSets-RAW"),
          ("Deep Sets + pairs", "DeepSetsP", "DeepSetsP-RAW"), ("PE-Net", "PENet", "PENet-RAW"),
          ("Set Transformer", "SetTransformer", "SetTransformer-RAW")]),
    ]
    lev = {u: _levels(root) for u, root in UNIVERSES}
    # the reference of every variant is the same network in the main design (MLP-sorted: its own main-design level)
    for u, root in UNIVERSES:
        d = pd.read_csv(root / "main" / "tables" / "losses_h5.csv")
        d = d[d.block == "in-range"]
        lev[u].update({m: v for m, v in zip(d.model, d.qlike_rel_ewma) if m not in lev[u]})
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 4.2), sharex=True)
    for ax, (title, _, rows) in zip(axes, panels):
        y = np.arange(len(rows))[::-1]
        for k, (u, _) in enumerate(UNIVERSES):
            vals = [100 * (lev[u].get(var, np.nan) - lev[u].get(ref, np.nan)) for _, ref, var in rows]
            yy = y + (0.18 if k == 0 else -0.18)
            ax.barh(yy, vals, height=0.32, color=UCOL[u], label=u, zorder=3)
            for yi, v in zip(yy, vals):
                if v == v:
                    ax.text(v + (0.25 if v >= 0 else -0.25), yi, f"${v:.1f}$" if v < 0 else f"$+{v:.1f}$",
                            va="center", ha="left" if v >= 0 else "right", fontsize=8.5, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows])
        ax.axvline(0, color=INK, lw=0.8, zorder=4)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_xlim(-5, 10)
    for ax in axes:
        ax.set_xlabel("change in QLIKE (points of EWMA)")
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.0)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, [f"{x} universe" for x in l_], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    save(fig, "conditions")


def fig_by_size():
    """h=5: accuracy by the number of constituents, both universes (two large panels)."""
    groups = {"N5": 5, "N10": 10, "N20": 20, "N30": 30, "N40": 40}
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.9))
    for ax, (u, root) in zip(axes, UNIVERSES):
        n_all = 54 if u == "ETF" else 122
        g = dict(groups, **{"EW-all": n_all})
        d = pd.read_csv(root / "main" / "tables" / "losses_h5.csv")
        d = d[d.block.isin(g)].assign(N=lambda x: x.block.map(g))
        for m in HIGHLIGHT:
            s_ = d[d.model == m].sort_values("N")
            model_line(ax, s_.N.to_numpy(), s_.qlike_rel_ewma.to_numpy(), m)
        b = best_classical(root, 5)
        s_ = d[d.model == b].sort_values("N")
        bench_line(ax, s_.N.to_numpy(), s_.qlike_rel_ewma.to_numpy(), b)
        ax.axvspan(24, n_all * 1.15, color=SHADE, zorder=0, lw=0)
        ax.text(np.sqrt(24 * n_all * 1.15), ax.get_ylim()[1], "never seen\nin training", ha="center", va="top",
                fontsize=8, color=INK2)
        ax.set_xscale("log")
        ticks = [5, 10, 20, 40, n_all]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(v) for v in ticks])
        ax.minorticks_off()
        ax.set_xlim(4.3, n_all * 1.15)
        ax.set_title(f"{u} universe (best classical: {lab(b)})")
        ax.set_xlabel("number of constituents $N$")
    axes[0].set_ylabel("QLIKE relative to EWMA")
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2.0)
    bottom_legend(fig, axes)
    save(fig, "by_size")


EPISODES = [("2020-02-20", "2020-05-29"), ("2025-03-31", "2025-05-30")]  # COVID crash, 2025 tariff shock


def fig_cumdiff():
    """Cumulative sum of the daily in-range QLIKE differential, network average minus the best classical model of the
    horizon, one panel per horizon and one line per universe."""
    import pvol.data as pdata
    from pvol.data import load_panel

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.9), sharex=True)
    for u, root in UNIVERSES:
        old = pdata.PROCESSED
        pdata.PROCESSED = ROOT / "data" / ("processed" if u == "ETF" else "processed_stocks")
        dates = pd.to_datetime(load_panel()["dates"])
        pdata.PROCESSED = old
        for ax, h in zip(axes, HORIZONS):
            R = run(root, h)
            ci = hyp.cols_of(group_index(R["ports"]), "in-range")
            b = best_classical(root, h)
            lb = np.nanmean(qlike(R["y"], R["F"][b])[:, ci], 1)
            ln = np.mean([np.nanmean(qlike(R["y"], R["F"][m])[:, ci], 1) for m in NETS], 0)
            ax.plot(dates[R["origins"]], np.cumsum(ln - lb), color=UCOL[u], lw=1.4,
                    ls="-" if u == "ETF" else (0, (5, 1.5)))
    for ax, h in zip(axes, HORIZONS):
        for s_, e in EPISODES:
            ax.axvspan(pd.Timestamp(s_), pd.Timestamp(e), color=SHADE, zorder=0, lw=0)
        ax.axhline(0.0, color=AXIS, lw=0.7, zorder=1)
        ax.set_title(f"$h={h}$")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].set_ylabel("cumulative QLIKE difference\n(absolute QLIKE)")
    handles = [plt.Line2D([], [], color=UCOL["ETF"], lw=1.4), plt.Line2D([], [], color=UCOL["Stock"], lw=1.4, ls=(0, (5, 1.5)))]
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=1.5)
    fig.legend(handles, ["ETF universe", "Stock universe"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    save(fig, "cumdiff")


def fig_eqcov():
    """EQ-Cov (ETF, h=5, seed 0): monthly averages of the mixing weights and of the average pairwise correlation."""
    d = pd.read_csv(BASE / "main" / "tables" / "eqcov_diagnostics_h5.csv", parse_dates=["date"])
    m = d.set_index("date").resample("MS").mean(numeric_only=True)
    parts = [("pi_ewma", "EWMA"), ("pi_corr66", "66-day"), ("pi_gram", "learned $AA^\\top$"), ("pi_corr22", "22-day")]
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(TEXTWIDTH, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1]))
    a1.stackplot(m.index, *[m[c] for c, _ in parts], labels=[n for _, n in parts], colors=colors, lw=0, step="post")
    a1.set_ylim(0, 1)
    a1.set_ylabel("mixing weight $\\pi$")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4)
    a2.plot(m.index, m.mean_corr_ewma, color=AXIS, lw=1.4, label="EWMA input", drawstyle="steps-post")
    a2.plot(m.index, m.mean_corr_hat, color=INK, lw=1.4, label="EQ-Cov forecast", drawstyle="steps-post")
    a2.set_ylabel("mean correlation")
    a2.legend(loc="upper right", ncol=2)
    for ax in (a1, a2):
        ax.grid(axis="x", visible=False)
        for y in range(2021, 2027):
            ax.axvline(pd.Timestamp(f"{y}-01-01"), color="white" if ax is a1 else GRID, lw=1.0, zorder=2)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlim(m.index.min(), m.index.max() + pd.offsets.MonthBegin(1))
    fig.tight_layout(h_pad=0.8)
    save(fig, "eqcov_diagnostics")


if __name__ == "__main__":
    fig_conditions()
    fig_by_size()
    fig_cumdiff()
    fig_eqcov()
    print("figures written to", FIG)
