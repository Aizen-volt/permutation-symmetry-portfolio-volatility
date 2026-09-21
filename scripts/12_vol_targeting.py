"""Economic evaluation 1: volatility targeting with each model's forecasts (uses saved forecasts only).

Every fixed evaluation portfolio is levered by sigma*/sigma_hat at each rebalancing origin (every h days, using the
h-day forecast) and held for h days. A good risk model delivers a managed portfolio whose realised volatility is
close to the target in every month. Metrics (averaged over the portfolios of a block):
  ratio   realised volatility of the managed series over the whole test period / target (ideal 1)
  rmse    root mean squared log deviation of monthly (21-day) realised volatility from the target (ideal 0)
  tail    share of months with realised volatility above 1.5 x target
The target cancels out of all three metrics, so it is set to 1.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import RESULTS  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.evaluate import CLASSICAL, NEURAL, group_index, load_run  # noqa: E402

BLOCKS = {"main": {"in-range": ["N5", "N10", "N20", "N10-conc", "N10-EW"], "larger-N": ["N30", "N40", "EW-all"],
                   "long-short": ["N10-LS"], "structured": ["60/40", "AllWeather", "GlobalMulti"]},
          "heldout": {"unseen-assets": ["H5", "H10", "H10-EW"]}}


def managed_returns(r, f, origins, h):
    """r: (T_all, P) daily portfolio returns; f: (T, P) variance forecasts at `origins`; rebalance every h origins."""
    reb = np.arange(0, len(origins), h)
    out = []
    for k in reb:
        t = origins[k]
        expo = 1.0 / np.sqrt(f[k])  # (P,)
        out.append(r[t + 1 : t + 1 + h] * expo)
    return np.concatenate(out)  # (~T, P) daily managed returns, in chronological order


def metrics(m, month=21):
    ratio = np.sqrt((m**2).mean(0) * 252)
    n = (len(m) // month) * month
    mv = np.sqrt((m[:n].reshape(-1, month, m.shape[1]) ** 2).mean(1) * 252)  # (months, P)
    rmse = np.sqrt(np.mean(np.log(mv) ** 2, axis=0))
    tail = np.mean(mv > 1.5, axis=0)
    return ratio, rmse, tail


if __name__ == "__main__":
    exp = sys.argv[1] if len(sys.argv) > 1 else "main"
    horizons = [int(x) for x in sys.argv[2:]] or [1, 5, 22]
    p = load_panel()
    R = p["R"].astype(np.float64)
    rows = []
    for h in horizons:
        run = load_run(exp, h)
        ports, origins = run["ports"], run["origins"]
        r = np.column_stack([R[:, q["ids"]] @ np.asarray(q["weights"]) for q in ports])
        gidx = group_index(ports)
        # annualise the target: daily forecasts are average daily variances -> vol target 1 in annual units
        for m in [x for x in CLASSICAL + NEURAL if x in run["F"]]:
            f = run["F"][m] * 252
            for b, groups in BLOCKS[exp].items():
                cols = np.concatenate([gidx[g] for g in groups if g in gidx])
                if not np.isfinite(f[:, cols]).all():
                    continue
                mr = managed_returns(r[:, cols], f[:, cols], origins, h)
                ratio, rmse, tail = metrics(mr)
                rows.append(dict(h=h, block=b, model=m, ratio=ratio.mean(), abs_ratio_dev=np.abs(ratio - 1).mean(),
                                 rmse_log_monthly=rmse.mean(), tail_share=tail.mean()))
        print(f"h={h} done", flush=True)
    df = pd.DataFrame(rows)
    out = RESULTS / exp / "tables" / "vol_targeting.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(df[df.block == list(BLOCKS[exp])[0]].pivot(index="model", columns="h", values="rmse_log_monthly").round(4))
