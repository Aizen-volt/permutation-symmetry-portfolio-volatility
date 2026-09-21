"""Shrinkage benchmarks: Ledoit-Wolf (2004), EWMA volatilities + nonlinear-shrinkage correlation (Ledoit-Wolf 2020),
and DCC-NL (Engle, Ledoit and Wolf, 2019). Forecasts saved next to the other classical benchmarks."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402

from pvol import baselines as bl  # noqa: E402
from pvol.config import HORIZONS, RESULTS, TEST_YEARS  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.dataset import origins_in_year, year_bounds  # noqa: E402
from pvol.portfolios import load  # noqa: E402

if __name__ == "__main__":
    p = load_panel()
    R = p["R"].astype(np.float64)
    S = p["S_ewma"]
    dates = p["dates"]
    exps = sys.argv[1:] or ["main", "heldout"]
    ports = {e: load(RESULTS / e / "portfolios.json") for e in exps}
    preds = {(e, h, k): [] for e in exps for h in HORIZONS for k in ("LW", "EWMANLS", "DCCNL")}
    for y in TEST_YEARS:
        t0 = time.time()
        fit_end = year_bounds(dates, y - 1)[1]
        dcc = bl.DCC().fit(R, fit_end, n_jobs=6, target="nls")
        for h in HORIZONS:
            t_idx = origins_in_year(dates, y, h)
            lw = bl.rolling_shrinkage_covs(R, t_idx, kind="lw")
            nls = bl.rolling_shrinkage_covs(R, t_idx, kind="ewma-nls", S_ewma=S)
            for e in exps:
                preds[(e, h, "LW")].append(bl.quad_forms(lw, ports[e]))
                preds[(e, h, "EWMANLS")].append(bl.quad_forms(nls, ports[e]))
                preds[(e, h, "DCCNL")].append(bl.dcc_portfolio_forecasts(dcc, ports[e], t_idx, h))
                d = RESULTS / e / f"h{h}" / "info"
                (d / f"DCCNL__y{y}.json").write_text(json.dumps(dict(a=dcc.a, b=dcc.b)))
        print(f"year {y}: DCC-NL a={dcc.a:.4f} b={dcc.b:.4f}  {time.time() - t0:.0f}s", flush=True)
    for (e, h, k), v in preds.items():
        np.save(RESULTS / e / f"h{h}" / "preds" / f"{k}.npy", np.log(np.maximum(np.concatenate(v), 1e-12)).astype(np.float32))
    print("done")
