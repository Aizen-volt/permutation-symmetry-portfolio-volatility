"""Tests of the third series (R1-R5), decision rules as registered in docs/JOURNAL.md section 4b.

QLIKE, HLN-corrected Diebold-Mariano test on daily cross-sectional average loss differentials, 5%, Holm within family;
in-range portfolios. Writes results*/main/tables/hypotheses_review.csv and review_levels.csv for the current universe
(PVOL_UNIVERSE), and appends the permutation consistency of the TTA networks to perm_h*.csv.
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import RESULTS, UNIVERSE_NAME  # noqa: E402
from pvol.evaluate import daily_loss, dm_test, group_index, load_run, qlike  # noqa: E402
from pvol.portfolios import load as load_ports  # noqa: E402

_spec = importlib.util.spec_from_file_location("hyp", Path(__file__).with_name("15_hypotheses.py"))
hyp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hyp)

SYM = ["DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov"]
CP = {"DeepSets": "MLP", "PENet": "MLP", "SetTransformer": "MLP", "EQCov": "MLP", "SetGRU": "GRU"}


def comparisons(L, ci, h, fam, block, pairs):
    rows = []
    for a, b in pairs:
        if a in L and b in L:
            d = daily_loss(L[a], ci) - daily_loss(L[b], ci)
            stat, p = dm_test(d, h)
            rows.append(dict(hypothesis=fam, h=h, block=block, model=a, benchmark=b, mean_diff=np.mean(d), dm=stat, p=p))
    return hyp.family(rows)


def perm_consistency(R, h, tags):
    pp = load_ports(RESULTS / "main" / "perm_portfolios.json")
    src = np.array([p["src"] for p in pp], dtype=int)
    k_ = np.array([p["perm"] for p in pp], dtype=int)
    d = RESULTS / "main" / f"h{h}" / "preds"
    out = []
    for m in tags:
        stats = []
        for f in sorted(d.glob(f"{m}__s*__perm.npy")):
            s = np.load(f).astype(np.float64)
            S = np.stack([s[:, k_ == k] for k in range(k_.max() + 1)], 0)
            spread = S.max(0) - S.min(0)
            yq = R["y"][:, src[k_ == 0]]
            ql = np.stack([qlike(yq, np.exp(S[k])).mean() for k in range(len(S))])
            stats.append((np.mean(np.std(S, 0)), np.median(spread), np.percentile(spread, 99), ql.max() - ql.min()))
        if stats:
            a = np.array(stats).mean(0)
            out.append(dict(h=h, model=m, mean_sd_logvar=a[0], median_range_logvar=a[1], p99_range_logvar=a[2],
                            qlike_range_over_orderings=a[3]))
    return out


if __name__ == "__main__":
    rows, levels = [], []
    for h in (1, 5):
        if not (RESULTS / "main" / f"h{h}" / "targets.npy").exists():
            continue
        R = load_run("main", h)
        ci = hyp.cols_of(group_index(R["ports"]), "in-range")
        L = {m: qlike(R["y"], f) for m, f in R["F"].items()}
        e = np.nanmean(L["EWMA"][:, ci])
        for m in sorted(L):
            levels.append(dict(h=h, model=m, qlike_rel_ewma=np.nanmean(L[m][:, ci]) / e))
        if UNIVERSE_NAME == "stocks54":  # R5: H1 on the 54-stock subsample
            rows += comparisons(L, ci, h, "R5", "in-range", [(a, CP[a]) for a in SYM])
            continue
        # R3: symmetric network vs counterpart averaged over 8 orderings at prediction time
        rows += comparisons(L, ci, h, "R3", "in-range", [(a, CP[a] + "-TTA") for a in SYM])
        rows += comparisons(L, ci, h, "R3 (aux: TTA vs single ordering)", "in-range",
                            [(m + "-TTA", m) for m in ("MLP", "GRU")])
        perm = RESULTS / "main" / "tables" / f"perm_h{h}.csv"
        if perm.exists() and perm.stat().st_size > 10:
            old = pd.read_csv(perm)
            new = pd.DataFrame(perm_consistency(R, h, ["MLP-TTA", "GRU-TTA"]))
            if len(new):
                pd.concat([old[~old.model.isin(new.model)], new]).to_csv(perm, index=False)
        if h != 5:
            continue
        for n in (50, 500):  # R1
            rows += comparisons(L, ci, h, "R1a", f"in-range, pool {n}, shuffled counterpart",
                                [(f"{a}-P{n}", f"{CP[a]}-P{n}s") for a in SYM])
        rows += comparisons(L, ci, h, "R1b", "in-range, shuffled vs fixed order",
                            [(f"{m}-P{n}s", f"{m}-P{n}") for n in (50, 500) for m in ("MLP", "GRU")])
        rows += comparisons(L, ci, h, "R2", "in-range, raw inputs",
                            [(f"{a}-RAW", "MLP-sorted-RAW") for a in ("DeepSets", "PENet", "SetTransformer")])
        for n in (50, 500):
            rows += comparisons(L, ci, h, "R2", f"in-range, pool {n}",
                                [(f"{a}-P{n}", f"MLP-sorted-P{n}") for a in ("DeepSets", "PENet", "SetTransformer", "EQCov")])
        rows += comparisons(L, ci, h, "R2 (aux: sorted vs MLP)", "in-range",
                            [("MLP-sorted-RAW", "MLP-RAW"), ("MLP-sorted-P50", "MLP-P50"), ("MLP-sorted-P500", "MLP-P500")])
        rows += comparisons(L, ci, h, "R4", "in-range, raw inputs", [("DeepSetsP-RAW", "MLP-RAW")])
        rows += comparisons(L, ci, h, "R4 (aux)", "in-range",
                            [("DeepSetsP", "MLP"), ("DeepSetsP-RAW", "DeepSets-RAW")])
    out = RESULTS / "main" / "tables"
    pd.DataFrame(rows).to_csv(out / "hypotheses_review.csv", index=False)
    pd.DataFrame(levels).to_csv(out / "review_levels.csv", index=False)
    pd.set_option("display.width", 220)
    if rows:
        print(pd.DataFrame(rows)[["hypothesis", "h", "block", "model", "benchmark", "mean_diff", "dm", "p_holm", "verdict"]]
              .round(4).to_string(index=False))
