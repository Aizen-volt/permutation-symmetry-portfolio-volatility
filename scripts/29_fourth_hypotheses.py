"""Tests of the fourth series that are not covered by 15_hypotheses.py.

S1b: symmetric network vs the Linear and GBM benchmarks, in-range, h = 1, 5, 22 (Holm over the five networks within each
     (h, benchmark)).
S2:  H1, R3 and R5 at h = 1 with ensembles of ten seeds (seeds 0-2 of the main runs plus seeds 3-9 of S2, stored under
     the same tags); stock universe and the 54-stock subsample.
(S1a = 15_hypotheses.py --ext, S3 = 15_hypotheses.py --suffix=-RF [--ext].)

Usage: [PVOL_UNIVERSE=stocks|stocks54] python scripts/29_fourth_hypotheses.py
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import RESULTS, UNIVERSE_NAME  # noqa: E402
from pvol.evaluate import daily_loss, dm_test, group_index, load_run, qlike  # noqa: E402

_spec = importlib.util.spec_from_file_location("hyp", Path(__file__).with_name("15_hypotheses.py"))
hyp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hyp)
SYM, CP = hyp.SYMMETRIC, hyp.COUNTERPART


def compare(La, Lb, ci, hz, **kw):
    d = daily_loss(La, ci) - daily_loss(Lb, ci)
    st, p = dm_test(d, hz)
    return dict(mean_diff=d.mean(), dm=st, p=p, loss_a=np.nanmean(La[:, ci]), loss_b=np.nanmean(Lb[:, ci]), **kw)


def ten_seed(R, m):
    runs = list(R["F_seed"].get(m, {}).values())
    return np.mean(np.stack(runs), 0), len(runs)


if __name__ == "__main__":
    rows = []
    for h in (1, 5, 22):
        if not (RESULTS / "main" / f"h{h}" / "targets.npy").exists():
            continue
        R = load_run("main", h, seeds=None)
        y = R["y"]
        F = load_run("main", h, seeds=[0, 1, 2])["F"]  # main analysis: seeds 0-2 only
        ci = hyp.cols_of(group_index(R["ports"]), "in-range")
        if UNIVERSE_NAME != "stocks54":
            for bench in ("Linear", "GBM"):
                if bench in F:
                    fam = [compare(qlike(y, F[a]), qlike(y, F[bench]), ci, h, hypothesis="S1b", h=h, block="in-range",
                                   model=a, benchmark=bench) for a in SYM if a in F]
                    rows += hyp.family(fam)
        if h == 1 and any(max(runs) >= 3 for runs in R["F_seed"].values()):
            E = {}
            for m in set(SYM) | set(CP.values()) | {"MLP-TTA", "GRU-TTA"}:
                if m in R["F_seed"]:
                    E[m], n = ten_seed(R, m)
            fam = "S2-R5" if UNIVERSE_NAME == "stocks54" else "S2-H1"
            rows += hyp.family([compare(qlike(y, E[a]), qlike(y, E[CP[a]]), ci, h, hypothesis=fam, h=h, block="in-range",
                                        model=a, benchmark=CP[a]) for a in SYM])
            if "MLP-TTA" in E:
                rows += hyp.family([compare(qlike(y, E[a]), qlike(y, E[CP[a] + "-TTA"]), ci, h, hypothesis="S2-R3", h=h,
                                            block="in-range", model=a, benchmark=CP[a] + "-TTA") for a in SYM])
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "main" / "tables" / "hypotheses_fourth.csv", index=False)
    pd.set_option("display.width", 220)
    if len(df):
        print(df[["hypothesis", "h", "model", "benchmark", "loss_a", "loss_b", "dm", "p_holm", "verdict"]].round(4)
              .to_string(index=False))
