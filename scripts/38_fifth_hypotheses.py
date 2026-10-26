"""Tests of the fifth series (T, U), decision rules fixed before the series was run (docs/JOURNAL.md, section 4f).

QLIKE, HLN-corrected Diebold-Mariano test on daily cross-sectional average loss differentials, 5%, Holm within family;
in-range portfolios; seed ensembles (seeds 0-2; P50: 0-1).
T1 (H1-T):  symmetric network vs counterpart, both with the settings of the broader search, Holm over 5 per h.
T2 (S1b-T): each of the 7 networks (-T) vs Linear-T, Holm over 7 per h.
T3 (H2-T):  symmetric network (-T) vs the best classical model of the extended set (ex post), Holm over 5 per h.
T4 (aux):   model -T vs the same model in the main protocol, Holm over 8 per h; best validation epoch of every refit.
U1:         MLP-idsort vs MLP in the main design, with raw inputs and with a pool of 50 portfolios (h = 5), Holm over 3.
U2:         MLP-sorted vs MLP-idsort in the same three conditions, Holm over 3.
Writes results*/main/tables/hypotheses_fifth.csv, fifth_levels.csv and fifth_epochs.csv for the current universe.

Usage: [PVOL_UNIVERSE=stocks] python scripts/38_fifth_hypotheses.py
"""
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import RESULTS  # noqa: E402
from pvol.evaluate import CLASSICAL, daily_loss, dm_test, group_index, load_run, qlike  # noqa: E402

_spec = importlib.util.spec_from_file_location("hyp", Path(__file__).with_name("15_hypotheses.py"))
hyp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hyp)
SYM, CP = hyp.SYMMETRIC, hyp.COUNTERPART
NETS = ["MLP", "GRU"] + SYM
T_MODELS = NETS + ["Linear"]


def comparisons(L, ci, h, fam, block, pairs):
    rows = []
    for a, b in pairs:
        if a in L and b in L:
            d = daily_loss(L[a], ci) - daily_loss(L[b], ci)
            stat, p = dm_test(d, h)
            rows.append(dict(hypothesis=fam, h=h, block=block, model=a, benchmark=b, loss_a=np.nanmean(L[a][:, ci]),
                             loss_b=np.nanmean(L[b][:, ci]), mean_diff=np.mean(d), dm=stat, p=p))
    return hyp.family(rows)


def best_epochs(h):
    out = []
    for f in (RESULTS / "main" / f"h{h}" / "info").glob("*__s*__y*.json"):
        d = json.loads(f.read_text())
        tag = d.get("tag", d["model"])
        base = tag[:-2] if tag.endswith("-T") else tag
        if base not in T_MODELS or tag not in (base, base + "-T") or not d.get("history") or d["seed"] > 2:
            continue
        out.append(dict(h=h, model=base, protocol="T" if tag.endswith("-T") else "main",
                        best_epoch=1 + int(np.argmin([r[2] for r in d["history"]])), epochs=d["epochs"]))
    return out


if __name__ == "__main__":
    rows, levels, ep = [], [], []
    for h in (1, 5, 22):
        if not (RESULTS / "main" / f"h{h}" / "targets.npy").exists():
            continue
        R = load_run("main", h)
        ci = hyp.cols_of(group_index(R["ports"]), "in-range")
        L = {m: qlike(R["y"], f) for m, f in R["F"].items()}
        e = np.nanmean(L["EWMA"][:, ci])
        for m in sorted(L):
            if m.endswith("-T") or "idsort" in m or m in T_MODELS:
                levels.append(dict(h=h, model=m, qlike_rel_ewma=np.nanmean(L[m][:, ci]) / e))
        if all(m + "-T" in L for m in T_MODELS):
            rows += comparisons(L, ci, h, "T1", "in-range", [(a + "-T", CP[a] + "-T") for a in SYM])
            rows += comparisons(L, ci, h, "T2", "in-range", [(a + "-T", "Linear-T") for a in NETS])
            bench = min((m for m in CLASSICAL if m in L), key=lambda m: np.nanmean(L[m][:, ci]))
            rows += comparisons(L, ci, h, "T3", "in-range", [(a + "-T", bench) for a in SYM])
            rows += comparisons(L, ci, h, "T4 (aux: broader search vs main protocol)", "in-range",
                                [(m + "-T", m) for m in T_MODELS])
        ep += best_epochs(h)
        if h == 5:
            conds = [("", "main design", None), ("-RAW", "raw inputs", None), ("-P50", "pool 50", [0, 1])]
            pairs1, pairs2 = [], []
            for sfx, block, seeds in conds:
                Ls = L if seeds is None else {m: qlike(R["y"], f) for m, f in load_run("main", h, seeds=seeds)["F"].items()}
                pairs1 += comparisons(Ls, ci, h, "U1", block, [(f"MLP-idsort{sfx}", f"MLP{sfx}")])
                pairs2 += comparisons(Ls, ci, h, "U2", block, [(f"MLP-sorted{sfx}", f"MLP-idsort{sfx}")])
            for fam in (pairs1, pairs2):  # Holm over the three conditions
                for r in fam:
                    r.pop("p_holm", None)
                rows += hyp.family(fam)
    out = RESULTS / "main" / "tables"
    df = pd.DataFrame(rows)
    df.to_csv(out / "hypotheses_fifth.csv", index=False)
    pd.DataFrame(levels).to_csv(out / "fifth_levels.csv", index=False)
    E = pd.DataFrame(ep)
    if len(E):
        E = (E.assign(le2=E.best_epoch <= 2).groupby(["h", "model", "protocol"])
             .agg(runs=("best_epoch", "size"), median_best_epoch=("best_epoch", "median"), share_le2=("le2", "mean"),
                  median_epochs=("epochs", "median")).reset_index())
        E.to_csv(out / "fifth_epochs.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    if len(df):
        print(df[["hypothesis", "h", "block", "model", "benchmark", "loss_a", "loss_b", "mean_diff", "dm", "p_holm",
                  "verdict"]].round(4).to_string(index=False))
    if len(E):
        print(E.round(3).to_string(index=False))
