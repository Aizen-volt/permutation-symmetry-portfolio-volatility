"""Formal tests of hypotheses H1-H4 with the decision rules stated in the paper (Section 'Research hypotheses').

Rule: model a is MORE ACCURATE than b in (block, h) iff mean QLIKE of a < mean QLIKE of b AND the HLN-corrected
Diebold-Mariano test on the daily cross-sectional average loss differential rejects at 5% after the Holm correction
within the family (hypothesis, h, block). If the test does not reject: 'indistinguishable'; if it rejects with the
opposite sign: 'less accurate'. H3a and H4 test a negative mean of a difference of two differential series.

Usage: python scripts/15_hypotheses.py [main|heldout]   (MSE instead of QLIKE with --mse)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import HORIZONS, RESULTS  # noqa: E402
from pvol.evaluate import CLASSICAL, daily_loss, dm_test, group_index, load_run, mse, qlike  # noqa: E402

ALPHA = 0.05
SUFFIX = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--suffix=")), "")
SYMMETRIC = ["DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov"]
COUNTERPART = {"DeepSets": "MLP", "PENet": "MLP", "SetTransformer": "MLP", "EQCov": "MLP", "SetGRU": "GRU"}
BLOCKS = {"in-range": ["N5", "N10", "N20", "N10-conc", "N10-EW"], "larger-N": ["N30", "N40", "EW-all"],
          "long-short": ["N10-LS"], "structured": ["60/40", "AllWeather", "GlobalMulti"],
          "unseen-assets": ["H5", "H10", "H10-EW"]}
ABLATION_PAIRS = [("DeepSets", "MLP"), ("SetTransformer", "MLP"), ("EQCov", "MLP"), ("SetGRU", "GRU")]


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def verdict(mean_diff, p_adj):
    if p_adj < ALPHA:
        return "more accurate" if mean_diff < 0 else "less accurate"
    return "indistinguishable"


def family(rows):
    """Add Holm-adjusted p-values and verdicts to a list of comparisons that form one family."""
    if not rows:
        return rows
    adj = holm([r["p"] for r in rows])
    for r, pa in zip(rows, adj):
        r["p_holm"] = pa
        r["verdict"] = verdict(r["mean_diff"], pa)
    return rows


def compare(La, Lb, cols, h):
    da, db = daily_loss(La, cols), daily_loss(Lb, cols)
    d = da - db
    stat, p = dm_test(d, h)
    return dict(loss_a=da.mean(), loss_b=db.mean(), mean_diff=d.mean(), dm=stat, p=p), d


def cols_of(gidx, block):
    return np.concatenate([gidx[g] for g in BLOCKS[block] if g in gidx])


def run(exp, lossfn):
    out = []
    for h in HORIZONS:
        f = RESULTS / exp / f"h{h}" / "targets.npy"
        if not f.exists():
            continue
        R = load_run(exp, h)
        y, F = R["y"], R["F"]
        if SUFFIX:  # robustness variant: replace every network by its '<model><SUFFIX>' forecasts (e.g. protocol B)
            if not any(k.endswith(SUFFIX) for k in F):
                continue
            nets = [k for k in F if k not in CLASSICAL and not k.endswith(("-K2", "-K4", "-P2"))]
            F = {**{k: v for k, v in F.items() if k in CLASSICAL}, **{m: F[m + SUFFIX] for m in nets if m + SUFFIX in F}}
        gidx = group_index(R["ports"])
        L = {m: lossfn(y, F[m]) for m in F}
        blocks = [b for b in BLOCKS if any(g in gidx for g in BLOCKS[b])]
        for b in blocks:
            cols = cols_of(gidx, b)
            ok = {m for m in L if np.isfinite(L[m][:, cols]).all()}
            # H1 / H3b / H3c: symmetric vs counterpart (same information)
            hyp = {"in-range": "H1", "long-short": "H3b", "unseen-assets": "H3c"}.get(b)
            if hyp:
                rows = []
                for a in SYMMETRIC:
                    c = COUNTERPART[a]
                    if a in ok and c in ok:
                        res, _ = compare(L[a], L[c], cols, h)
                        rows.append(dict(hypothesis=hyp, h=h, block=b, model=a, benchmark=c, **res))
                out += family(rows)
            # H2: symmetric vs best classical benchmark of the block (chosen ex post, conservative)
            classical = [m for m in CLASSICAL if m in ok]
            if classical:
                bench = min(classical, key=lambda m: np.nanmean(L[m][:, cols]))
                rows = []
                for a in SYMMETRIC:
                    if a in ok:
                        res, _ = compare(L[a], L[bench], cols, h)
                        rows.append(dict(hypothesis="H2", h=h, block=b, model=a, benchmark=bench, **res))
                out += family(rows)
        # H3a: difference-in-differences larger-N vs in-range, same benchmark (best classical in-range)
        if "larger-N" in blocks and "in-range" in blocks:
            ci, cl = cols_of(gidx, "in-range"), cols_of(gidx, "larger-N")
            classical = [m for m in CLASSICAL if m in L]
            bench = min(classical, key=lambda m: np.nanmean(L[m][:, ci]))
            rows = []
            for a in SYMMETRIC:
                if a in L and np.isfinite(L[a][:, cl]).all():
                    d_in = daily_loss(L[a], ci) - daily_loss(L[bench], ci)
                    d_l = daily_loss(L[a], cl) - daily_loss(L[bench], cl)
                    D = d_l - d_in
                    stat, p = dm_test(D, h)
                    rows.append(dict(hypothesis="H3a", h=h, block="larger-N minus in-range", model=a, benchmark=bench,
                                     loss_a=d_l.mean(), loss_b=d_in.mean(), mean_diff=D.mean(), dm=stat, p=p))
            out += family(rows)
        # H4: ablation (short training history) vs full history, same seeds (0, 1)
        if exp == "main" and h == 5 and not SUFFIX:
            R2 = load_run(exp, h, seeds=[0, 1])
            F2 = R2["F"]
            ci = cols_of(gidx, "in-range")
            for K in (4, 2):
                rows = []
                for a, c in ABLATION_PAIRS:
                    ka, kc = f"{a}-K{K}", f"{c}-K{K}"
                    if all(m in F2 for m in (a, c, ka, kc)):
                        d_full = daily_loss(lossfn(y, F2[a]), ci) - daily_loss(lossfn(y, F2[c]), ci)
                        d_k = daily_loss(lossfn(y, F2[ka]), ci) - daily_loss(lossfn(y, F2[kc]), ci)
                        D = d_k - d_full
                        stat, p = dm_test(D, h)
                        rows.append(dict(hypothesis="H4", h=h, block=f"in-range, K={K} minus full", model=a, benchmark=c,
                                         loss_a=d_k.mean(), loss_b=d_full.mean(), mean_diff=D.mean(), dm=stat, p=p))
                out += family(rows)
    return pd.DataFrame(out)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_mse = "--mse" in sys.argv
    lossfn = mse if use_mse else qlike
    frames = [run(exp, lossfn) for exp in (args or ["main", "heldout"]) if (RESULTS / exp).exists()]
    df = pd.concat(frames, ignore_index=True)
    name = ("hypotheses_mse" if use_mse else "hypotheses") + (SUFFIX.replace("-", "_") if SUFFIX else "") + ".csv"
    for exp in args or ["main", "heldout"]:
        (RESULTS / exp / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "main" / "tables" / name, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    cols = ["hypothesis", "h", "block", "model", "benchmark", "mean_diff", "dm", "p", "p_holm", "verdict"]
    print(df[cols].round(4).to_string(index=False))
