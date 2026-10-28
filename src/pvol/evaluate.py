"""Forecast evaluation: losses, Diebold-Mariano tests, Model Confidence Set, Mincer-Zarnowitz, VaR backtests."""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from arch.bootstrap import MCS
from scipy import stats

from .config import RESULTS
from .portfolios import load

# HAR-WLS, LogHAR-X, HEAVY and PCA-F: stronger classical benchmarks of the fourth series (scripts/26_benchmarks_ext.py)
CLASSICAL = ["HIST", "EWMA", "HAR", "HAR-WLS", "HARX", "LogHAR-X", "HEAVY", "GARCH", "GJR", "DCC", "LW", "EWMANLS", "DCCNL",
             "PCA-F"]
# Linear and GBM: learned benchmarks on the information of the networks (fourth series)
NEURAL = ["PortMLP", "Linear", "GBM", "MLP", "MLP-sorted", "GRU", "DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov"]
INVARIANT = {"PortMLP", "Linear", "GBM", "MLP-sorted", "DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov"}
Y_FLOOR = 1e-12


# loading
def load_run(exp, h, seeds=(0, 1, 2)):
    """Targets, origins, portfolios and forecasts (variance units). NN forecasts: per seed and seed-ensemble.

    seeds: the seeds to load, by default the seeds 0-2 of the main analysis (the fourth series S2 stores seeds 3-9 of
    the stock universe under the same tags); None loads every seed on disk."""
    d = RESULTS / exp / f"h{h}"
    y = np.load(d / "targets.npy")
    origins = np.load(d / "origins.npy")
    ports = load(RESULTS / exp / "portfolios.json")
    F, F_seed = {}, {}
    for f in sorted(glob.glob(str(d / "preds" / "*.npy"))):
        stem = Path(f).stem
        if stem.endswith("__perm"):
            continue
        m = re.match(r"(.+)__s(\d+)$", stem)
        if m is None:
            F[stem] = np.exp(np.load(f).astype(np.float64))
        elif seeds is None or int(m.group(2)) in seeds:
            F_seed.setdefault(m.group(1), {})[int(m.group(2))] = np.exp(np.load(f).astype(np.float64))
    for name, runs in F_seed.items():
        F[name] = np.mean(np.stack(list(runs.values())), 0)  # ensemble = average variance forecast over seeds
    return dict(y=y, origins=origins, ports=ports, F=F, F_seed=F_seed)


# losses
def qlike(y, f):
    r = np.maximum(y, Y_FLOOR) / f
    return r - np.log(r) - 1.0


def mse(y, f):
    return (y - f) ** 2


def patton_bm1(y, f):
    """Patton (2011) robust loss with b = -1 (homogeneous of degree 1), between QLIKE (b = -2) and MSE (b = 0)."""
    y = np.maximum(y, Y_FLOOR)
    return y * np.log(y / f) - y + f


def group_index(ports):
    g = pd.Series([p["group"] for p in ports])
    return {k: np.where(g == k)[0] for k in g.unique()}


def daily_loss(L, cols):
    """Cross-sectional average over the portfolios in cols -> (T,) daily loss series."""
    return np.nanmean(L[:, cols], axis=1)


# Diebold-Mariano (HLN-corrected, NW HAC)
def dm_test(d, h):
    """Two-sided DM test of E[d]=0 for loss differential series d; returns (stat, p-value).

    Long-run variance: Newey-West, Bartlett kernel, lag = max(h, floor(4 (T/100)^(2/9))) to cover the MA(h-1)
    structure of overlapping multi-day targets. Harvey, Leybourne and Newbold (1997) small-sample correction.
    """
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    T = len(d)
    dc = d - d.mean()
    lag = max(h, int(np.floor(4 * (T / 100) ** (2 / 9))))
    lrv = dc @ dc / T
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / T
    stat = d.mean() / np.sqrt(lrv / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat *= hln
    p = 2 * stats.t.sf(abs(stat), df=T - 1)
    return stat, p


# Model Confidence Set (Hansen, Lunde, Nason 2011)
def mcs(loss_df, size=0.10, reps=5000, block=None, seed=0, method="R"):
    """MCS p-values; method "R" (range statistic, main tables) or "max" (T_max statistic)."""
    block = block or max(5, int(round(len(loss_df) ** (1 / 3))))
    m = MCS(loss_df, size=size, reps=reps, block_size=block, method=method, bootstrap="stationary", seed=seed)
    m.compute()
    return m.pvalues["Pvalue"].reindex(loss_df.columns)


# Mincer-Zarnowitz
def mincer_zarnowitz(y, f):
    """Pooled regression y = a + b f + e; returns (a, b, R^2)."""
    ok = np.isfinite(f)
    X = np.column_stack([np.ones(ok.sum()), f[ok]])
    beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    resid = y[ok] - X @ beta
    r2 = 1 - resid.var() / y[ok].var()
    return beta[0], beta[1], r2


# VaR backtests (h = 1)
def kupiec_pof(hits, alpha):
    n, x = len(hits), int(hits.sum())
    pi = x / n
    if x in (0, n):
        ll1 = 0.0
    else:
        ll1 = x * np.log(pi) + (n - x) * np.log(1 - pi)
    ll0 = x * np.log(alpha) + (n - x) * np.log(1 - alpha)
    lr = -2 * (ll0 - ll1)
    return lr, stats.chi2.sf(lr, 1)


def christoffersen_ind(hits):
    h = hits.astype(int)
    a, b = h[:-1], h[1:]
    n00 = np.sum((a == 0) & (b == 0)); n01 = np.sum((a == 0) & (b == 1))
    n10 = np.sum((a == 1) & (b == 0)); n11 = np.sum((a == 1) & (b == 1))
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1); p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(q, k0, k1):
        return (k0 * np.log(1 - q) if k0 else 0.0) + (k1 * np.log(q) if k1 else 0.0)

    lr = -2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
    return lr, stats.chi2.sf(lr, 1)


def var_backtest(r_next, f, alpha):
    """Gaussian VaR at level alpha from variance forecast f; per-portfolio Kupiec and Christoffersen CC tests."""
    q = stats.norm.ppf(alpha)
    res = []
    for j in range(r_next.shape[1]):
        if not np.all(np.isfinite(f[:, j])):
            continue
        hits = r_next[:, j] < q * np.sqrt(f[:, j])
        lr_uc, p_uc = kupiec_pof(hits, alpha)
        lr_ind, _ = christoffersen_ind(hits)
        p_cc = stats.chi2.sf(lr_uc + lr_ind, 2)
        res.append((hits.mean(), p_uc, p_cc))
    res = np.array(res)
    return dict(hit_rate=res[:, 0].mean(), rej_uc=(res[:, 1] < 0.05).mean(), rej_cc=(res[:, 2] < 0.05).mean())
