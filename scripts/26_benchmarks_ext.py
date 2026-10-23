"""Fourth series S1: stronger classical benchmarks (HAR-WLS, LogHAR-X, HEAVY, PCA-F) for the main experiment,
h = 1, 5, 22; forecasts saved next to the other benchmarks (log variance). Runs on the CPU.

Usage: [PVOL_UNIVERSE=stocks] python scripts/26_benchmarks_ext.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from pvol import baselines as bl  # noqa: E402
from pvol import baselines_ext as bx  # noqa: E402
from pvol.config import HORIZONS, RESULTS, TEST_YEARS  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.dataset import SEQ_MARGIN, origins_in_year, year_bounds  # noqa: E402
from pvol.portfolios import load  # noqa: E402

if __name__ == "__main__":
    p = load_panel()
    R = p["R"].astype(np.float64)
    S = p["S_ewma"]
    X, P = p["X"], p["P"]
    dates, fv = p["dates"], int(p["first_valid"])
    names = list(p["feat_names"])
    fidx = [names.index(f"log_rng_{k}") for k in (1, 5, 22)]
    ports = load(RESULTS / "main" / "portfolios.json")
    lo = fv + SEQ_MARGIN
    t_all = np.arange(lo, len(dates))

    def per_portfolio(q):
        rp = bl.portfolio_returns(R, q)
        rk = np.zeros(len(dates))
        rk[t_all] = bl.range_portfolio_var(X, P, q, t_all, fidx[:1])[:, 0]
        out = {}
        for y in TEST_YEARS:
            fit_end = year_bounds(dates, y - 1)[1]
            heavy_t = {h: origins_in_year(dates, y, h) for h in HORIZONS}
            t_any = heavy_t[1]
            hv = bx.heavy_forecasts(rp, rk, lo, fit_end, t_any, HORIZONS)
            for h in HORIZONS:
                t_idx = heavy_t[h]
                fit_idx = np.arange(lo, fit_end - h + 1)
                out[("HAR-WLS", h, y)] = bx.har_wls_forecast(rp, fit_idx, t_idx, h)
                out[("LogHAR-X", h, y)] = bx.loghar_x_forecast(rp, X, P, q, fit_idx, t_idx, h, fidx)
                out[("HEAVY", h, y)] = hv[h][: len(t_idx)]
        return out

    t0 = time.time()
    res = Parallel(n_jobs=6, prefer="threads")(delayed(per_portfolio)(q) for q in ports)
    print(f"univariate models {time.time() - t0:.0f}s", flush=True)
    for name in ("HAR-WLS", "LogHAR-X", "HEAVY"):
        for h in HORIZONS:
            f = np.concatenate([np.column_stack([r[(name, h, y)] for r in res]) for y in TEST_YEARS])
            np.save(RESULTS / "main" / f"h{h}" / "preds" / f"{name}.npy", np.log(np.maximum(f, 1e-12)).astype(np.float32))
    for h in HORIZONS:  # PCA-F: one covariance forecast per origin, flat across the horizon
        t_idx = np.concatenate([origins_in_year(dates, y, h) for y in TEST_YEARS])
        f = bl.quad_forms(bx.pca_factor_covs(R, S, t_idx), ports)
        np.save(RESULTS / "main" / f"h{h}" / "preds" / "PCA-F.npy", np.log(np.maximum(f, 1e-12)).astype(np.float32))
    print(f"done {time.time() - t0:.0f}s")
