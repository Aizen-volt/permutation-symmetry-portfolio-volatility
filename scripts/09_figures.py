"""Figures for the paper (vector PDF, drawn at the final printed width so that fonts are not scaled).

Style: text set by LaTeX (usetex: STIX like the manuscript in the Elsevier CAS template, Type 1 fonts with correct text encoding); the
same networks in the same Okabe-Ito colors in every figure (markers or line styles as secondary encoding); the best
classical model of each panel as a black dashed line; lines labeled directly at their ends instead of a legend; panel
titles (a), (b), ... aligned left; comparisons across networks as dot plots with the values printed at the right.
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


def end_labels(ax, items, x, x_text, gap):
    """Direct labels at the right end of lines: items = [(y, text, color)] for lines ending at x; the labels are set at
    x_text and moved apart vertically by at least gap (data units) so that they do not overlap."""
    items = sorted(items)
    ys = [y for y, *_ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    shift = (ys[-1] - items[-1][0]) / 2  # keep the block of labels centred on the line ends
    for (y0, text, c), y in zip(items, ys):
        ax.annotate(text, xy=(x, y0), xytext=(x_text, y - shift), textcoords="data", ha="left", va="center",
                    fontsize=8.5, color=c, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.5, shrinkA=1, shrinkB=1, alpha=0.7))


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
    """How much accuracy each network loses when an aid of the main design is removed (second and third series): a dot
    plot with one row per network, grouped into unconstrained and symmetric networks."""
    U, S = "unconstrained networks", "symmetric networks"
    panels = [
        ("(a) Fixed pool of 50 training portfolios",
         [(U, "MLP", "MLP", "MLP-P50"), (U, "MLP, shuffled order", "MLP", "MLP-P50s"),
          (U, "MLP-sorted", "MLP-sorted", "MLP-sorted-P50"), (S, "Deep Sets", "DeepSets", "DeepSets-P50"),
          (S, "PE-Net", "PENet", "PENet-P50"), (S, "Set Transformer", "SetTransformer", "SetTransformer-P50"),
          (S, "EQ-Cov", "EQCov", "EQCov-P50")]),
        ("(b) Raw inputs, without portfolio-level summaries",
         [(U, "MLP", "MLP", "MLP-RAW"), (U, "MLP-sorted", "MLP-sorted", "MLP-sorted-RAW"),
          (S, "Deep Sets", "DeepSets", "DeepSets-RAW"), (S, "Deep Sets + pairs", "DeepSetsP", "DeepSetsP-RAW"),
          (S, "PE-Net", "PENet", "PENet-RAW"), (S, "Set Transformer", "SetTransformer", "SetTransformer-RAW")]),
    ]
    lev = {u: _levels(root) for u, root in UNIVERSES}
    # the reference of every variant is the same network in the main design (MLP-sorted: its own main-design level)
    for u, root in UNIVERSES:
        d = pd.read_csv(root / "main" / "tables" / "losses_h5.csv")
        d = d[d.block == "in-range"]
        lev[u].update({m: v for m, v in zip(d.model, d.qlike_rel_ewma) if m not in lev[u]})
    marker = {"ETF": "o", "Stock": "s"}
    xlim, step = (-3.2, 9.2), 1.0  # step: vertical distance between rows
    layouts = []
    for title, rows in panels:  # y positions: a header line above each group, rows below it
        y, ys, heads, last = 0.0, [], [], None
        for g, *_ in rows:
            if g != last:
                y -= 0.35 if last is None else 0.75
                heads.append((g, y))
                y -= 0.85
                last = g
            ys.append(y)
            y -= step
        layouts.append((title, rows, ys, heads, y + step - 0.6))
    heights = [-lo + 0.5 for *_, lo in layouts]
    fig, axes = plt.subplots(2, 1, figsize=(TEXTWIDTH, 5.0), gridspec_kw=dict(height_ratios=heights, hspace=0.42))
    for ax, (title, rows, ys, heads, lo) in zip(axes, layouts):
        for g, yh in heads:
            if g == U:  # light band behind the unconstrained networks
                yb = [y_ for (g_, *_), y_ in zip(rows, ys) if g_ == U]
                ax.axhspan(min(yb) - 0.5, max(yb) + 0.5, xmin=0, xmax=1, color="#f4f4f4", lw=0, zorder=0)
            ax.text(xlim[0] - 0.02 * (xlim[1] - xlim[0]), yh, g, ha="right", va="center", fontsize=8.5, color=INK2,
                    style="italic")
        for (g, name, ref, var), y_ in zip(rows, ys):
            v = {u: 100 * (lev[u].get(var, np.nan) - lev[u].get(ref, np.nan)) for u, _ in UNIVERSES}
            close = abs(v["ETF"] - v["Stock"]) < 0.35  # nearly equal values: offset vertically so both stay visible
            dy = {"ETF": 0.24 * close, "Stock": -0.24 * close}
            ax.plot([v["ETF"], v["Stock"]], [y_ + dy["ETF"], y_ + dy["Stock"]], color="#c8c8c8", lw=2.2,
                    solid_capstyle="round", zorder=2)
            for u, _ in UNIVERSES:
                ax.plot(v[u], y_ + dy[u], marker=marker[u], ms=7 if u == "ETF" else 6.4, color=UCOL[u], mec="white", mew=1.0,
                        ls="none", zorder=3)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[1] for r in rows])
        ax.tick_params(axis="y", length=0, pad=6, labelsize=9.5, labelcolor=INK)
        ax.set_ylim(lo, 0.2)
        ax.set_xlim(*xlim)
        ax.set_xticks(range(-2, 10, 2))
        ax.axvline(0, color=INK, lw=0.9, zorder=1)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=4)
    axes[1].set_xlabel("change in QLIKE when the aid is removed (percentage points of the EWMA loss)")
    axes[1].annotate("", xy=(xlim[0] + 0.15, -0.16), xytext=(-0.25, -0.16), xycoords=("data", "axes fraction"),
                     arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7, shrinkA=0, shrinkB=0))
    axes[1].text(-0.35, -0.21, "more accurate", transform=axes[1].get_xaxis_transform(), ha="right", va="top",
                 fontsize=8.5, color=INK2)
    axes[1].annotate("", xy=(xlim[1] - 0.15, -0.16), xytext=(0.25, -0.16), xycoords=("data", "axes fraction"),
                     arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7, shrinkA=0, shrinkB=0))
    axes[1].text(0.35, -0.21, "less accurate", transform=axes[1].get_xaxis_transform(), ha="left", va="top",
                 fontsize=8.5, color=INK2)
    axes[1].xaxis.labelpad = 22
    handles = [plt.Line2D([], [], marker=marker[u], ms=7, color=UCOL[u], mec="white", mew=1.0, ls="none")
               for u, _ in UNIVERSES]
    fig.legend(handles, ["ETF universe", "Stock universe"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.3, columnspacing=2.0)
    save(fig, "conditions")


def fig_by_size():
    """h=5: accuracy by the number of constituents, both universes (two large panels, lines labeled at their ends)."""
    groups = {"N5": 5, "N10": 10, "N20": 20, "N30": 30, "N40": 40}
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.0))
    for k, (ax, (u, root)) in enumerate(zip(axes, UNIVERSES)):
        n_all = 54 if u == "ETF" else 122
        g = dict(groups, **{"EW-all": n_all})
        d = pd.read_csv(root / "main" / "tables" / "losses_h5.csv")
        d = d[d.block.isin(g)].assign(N=lambda x: x.block.map(g))
        labels = []
        for m in HIGHLIGHT:
            s_ = d[d.model == m].sort_values("N")
            model_line(ax, s_.N.to_numpy(), s_.qlike_rel_ewma.to_numpy(), m)
            labels.append((s_.qlike_rel_ewma.iloc[-1], lab(m), COLOR[m], s_.N.iloc[-1]))
        b = best_classical(root, 5)
        s_ = d[d.model == b].sort_values("N")
        bench_line(ax, s_.N.to_numpy(), s_.qlike_rel_ewma.to_numpy(), b)
        labels.append((s_.qlike_rel_ewma.iloc[-1], lab(b), INK, s_.N.iloc[-1]))
        lo, hi = ax.get_ylim()
        for x_end in sorted({x for *_, x in labels}):  # the MLP ends at N = 20 (20 slots)
            end_labels(ax, [(y, t, c) for y, t, c, x in labels if x == x_end], x_end, x_end * 1.12, 0.055 * (hi - lo))
        ax.axvspan(24, n_all * 1.15, color=SHADE, zorder=0, lw=0)
        ax.text(np.sqrt(24 * n_all * 1.15), hi, "never seen\nin training", ha="center", va="top", fontsize=8,
                color=INK2)
        ax.set_ylim(lo, hi)
        ax.set_xscale("log")
        ticks = [5, 10, 20, 40, n_all]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(v) for v in ticks])
        ax.minorticks_off()
        ax.set_xlim(4.3, n_all * 1.15)
        ax.set_title(f"({'ab'[k]}) {u} universe", loc="left")
        ax.set_xlabel("number of constituents $N$")
    axes[0].set_ylabel("QLIKE relative to EWMA")
    fig.tight_layout(w_pad=6.0)
    save(fig, "by_size")


EPISODES = [("2020-02-20", "2020-05-29"), ("2025-03-31", "2025-05-30")]  # COVID crash, 2025 tariff shock


def fig_cumdiff():
    """Cumulative sum of the daily in-range QLIKE differential, network average minus the best classical model of the
    horizon, one panel per horizon and one line per universe (labeled at its end); stress episodes shaded."""
    import pvol.data as pdata
    from pvol.data import load_panel

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.9), sharex=True)
    ends = {h: [] for h in HORIZONS}
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
            x, y = dates[R["origins"]], np.cumsum(ln - lb)
            ax.plot(x, y, color=UCOL[u], lw=1.4, ls="-" if u == "ETF" else (0, (5, 1.5)))
            ends[h].append((y[-1], "ETF" if u == "ETF" else "Stocks", UCOL[u], x[-1]))
    for k, (ax, h) in enumerate(zip(axes, HORIZONS)):
        for s_, e in EPISODES:
            ax.axvspan(pd.Timestamp(s_), pd.Timestamp(e), color=SHADE, zorder=0, lw=0)
        ax.axhline(0.0, color=AXIS, lw=0.7, zorder=1)
        ax.set_title(f"({'abc'[k]}) $h={h}$", loc="left")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        lo, hi = ax.get_ylim()
        x_end = max(x for *_, x in ends[h])
        end_labels(ax, [(y, t, c) for y, t, c, _ in ends[h]], x_end, x_end + pd.Timedelta(days=60), 0.09 * (hi - lo))
    axes[0].set_ylabel("cumulative QLIKE difference\n(networks minus best classical)")
    fig.tight_layout(w_pad=3.2)
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
