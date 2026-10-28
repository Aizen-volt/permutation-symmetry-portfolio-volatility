"""Learning curves of the main design from the stored training histories (no training).

Validation QLIKE per epoch relative to that after the first epoch, seed 0, h = 5, for three test years, both
universes; the kept epoch is marked. Also a table of the kept epoch over all seeds and test years (median and share of
runs stopped after one or two epochs) per network, horizon and universe.
Writes paper/figures/learning_curves.pdf and results*/main/tables/kept_epochs.csv.

Usage: python scripts/54_learning_curves.py   (both universes)
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("figs", Path(__file__).with_name("09_figures.py"))
figs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(figs)
plt = figs.plt

NETS = ["MLP", "DeepSets", "PENet", "SetTransformer", "EQCov", "Linear"]
YEARS = (2021, 2023, 2024)
STYLE = {**figs.COLOR, "SetTransformer": "#009E73", "Linear": "#1a1a1a"}


def history(root, h, m, seed, y):
    f = root / "main" / f"h{h}" / "info" / f"{m}__s{seed}__y{y}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return np.array(d["history"])  # (epoch, train loss, validation loss)


if __name__ == "__main__":
    fig, axes = plt.subplots(2, len(YEARS), figsize=(figs.TEXTWIDTH, 3.9), sharey=True)
    for i, (u, root) in enumerate(figs.UNIVERSES):
        for j, y in enumerate(YEARS):
            ax = axes[i, j]
            for m in NETS:
                H = history(root, 5, m, 0, y)
                if H is None:
                    continue
                v = H[:, 2] / H[0, 2]
                ax.plot(H[:, 0] + 1, v, color=STYLE.get(m, figs.INK2), lw=1.2, label=figs.lab(m) if (i, j) == (0, 0) else None)
                k = int(np.argmin(H[:, 2]))
                ax.scatter([k + 1], [v[k]], color=STYLE.get(m, figs.INK2), s=14, zorder=3)
            ax.set_xscale("log")
            ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
            ax.minorticks_off()
            if i == 0:
                ax.set_title(f"({'abc'[j]}) test year {y}", loc="left")
            if j == 0:
                ax.set_ylabel(f"{u}: validation loss\n(relative to epoch 1)")
            if i == 1:
                ax.set_xlabel("epoch")
    fig.legend(loc="upper center", ncol=len(NETS), bbox_to_anchor=(0.5, 1.04), handlelength=1.5, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(figs.FIG / "learning_curves.pdf")
    fig.savefig(figs.PREVIEW / "learning_curves.png", dpi=150)
    for u, root in figs.UNIVERSES:
        rows = []
        for h in (1, 5, 22):
            for m in NETS:
                ep = [int(np.argmin(H[:, 2])) + 1 for s in (0, 1, 2) for y in range(2020, 2027)
                      if (H := history(root, h, m, s, y)) is not None]
                if ep:
                    rows.append(dict(h=h, model=m, runs=len(ep), median_epoch=np.median(ep),
                                     share_epoch_1_2=np.mean(np.array(ep) <= 2)))
        out = pd.DataFrame(rows)
        out.to_csv(root / "main" / "tables" / "kept_epochs.csv", index=False)
        print(u)
        print(out.round(2).to_string(index=False))
