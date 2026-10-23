"""Stronger classical benchmarks and a gradient-boosting benchmark (fourth series, S1).

HAR-WLS   HAR of Corsi (2009) on squared portfolio returns, weighted least squares with weights 1/fitted^2 (3 iterations),
          i.e. a QLIKE-consistent estimator (the OLS HAR is dominated by the largest days).
LogHAR-X  regression of log y on the logs of the HAR components and of the range-based portfolio variance (1/5/22 days);
          the forecast is exp(fit) times the smearing factor mean(exp(residual)) (Duan, 1983).
HEAVY     Shephard and Sheppard (2010): conditional variance driven by the lagged range-based portfolio variance, plus an
          equation for that measure; both estimated by Gaussian QML; multi-step forecasts in closed form.
PCA-F     EWMA volatilities with a five-factor statistical correlation model of the last 250 days (principal components
          of the sample correlation matrix plus a diagonal residual).
GBM       histogram gradient boosting (Poisson loss, log link) of y / EWMA on the features of the linear set benchmark.
All forecasts are average daily portfolio variances over t+1..t+h; parameters fixed within the test year, filters
updated daily, as for the other classical benchmarks.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

from .baselines import har_design, hist_forecast, range_portfolio_var
from .config import VAR_FLOOR


def _target(rp, idx, h):
    c = np.concatenate([[0.0], np.cumsum(rp**2)])
    return (c[idx + h + 1] - c[idx + 1]) / h


# HAR-WLS
def har_wls_forecast(rp, fit_idx, t_idx, h, iters=3):
    y = _target(rp, fit_idx, h)
    Z = har_design(rp, fit_idx)
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    floor = 0.05 * y.mean()
    for _ in range(iters):
        f = np.maximum(Z @ beta, floor)
        sw = 1.0 / f
        beta, *_ = np.linalg.lstsq(Z * sw[:, None], y * sw, rcond=None)
    f = har_design(rp, t_idx) @ beta
    return np.where(f > VAR_FLOOR, f, hist_forecast(rp, t_idx))


# LogHAR-X
def loghar_x_forecast(rp, X, P, p, fit_idx, t_idx, h, feat_idx, eps=1e-10):
    def Z(idx):
        har = har_design(rp, idx)[:, 1:]
        rng = range_portfolio_var(X, P, p, idx, feat_idx)
        return np.column_stack([np.ones(len(idx)), np.log(np.maximum(har, eps)), np.log(np.maximum(rng, eps))])

    y = np.log(np.maximum(_target(rp, fit_idx, h), eps))
    Zf = Z(fit_idx)
    beta, *_ = np.linalg.lstsq(Zf, y, rcond=None)
    smear = np.mean(np.exp(y - Zf @ beta))
    return np.exp(Z(t_idx) @ beta) * smear


# HEAVY (Shephard and Sheppard, 2010)
def _heavy_path(x, omega, alpha, beta, init):
    """v[t] = omega + alpha * x[t-1] + beta * v[t-1], v[0] = init: the forecast of day t given information up to t-1."""
    drive = np.concatenate([[0.0], x[:-1]])
    v = lfilter([alpha], [1.0, -beta], drive) + omega * (1 - beta ** np.arange(len(x))) / (1 - beta)
    return v + init * beta ** np.arange(len(x))


def _fit_heavy(target, driver, init):
    """Gaussian QML of target[t] ~ v[t] with v driven by driver[t-1]; returns (omega, alpha, beta)."""
    scale = np.mean(target)

    def unpack(z):
        a, b = 1 / (1 + np.exp(-z[1])), 1 / (1 + np.exp(-z[2]))
        b = b * (1 - 1e-4)
        return np.exp(z[0]) * scale, a, b

    def nll(z):
        om, a, b = unpack(z)
        v = np.maximum(_heavy_path(driver, om, a, b, init), 1e-12 * scale)
        return np.mean(np.log(v) + target / v)

    z0 = np.array([np.log(0.05), np.log(0.3 / 0.7), np.log(0.65 / 0.35)])
    res = minimize(nll, z0, method="Nelder-Mead", options=dict(maxiter=2000, xatol=1e-5, fatol=1e-8))
    return unpack(res.x)


def heavy_forecasts(rp, rk, fit_lo, fit_end, t_idx, horizons):
    """rp: portfolio returns, rk: range-based portfolio variance of each day; returns {h: forecasts at t_idx}."""
    r2 = rp**2
    init = np.mean(r2[fit_lo : fit_lo + 22])
    seg = slice(fit_lo, fit_end + 1)
    om, a, b = _fit_heavy(r2[seg], rk[seg], init)  # variance equation: E[r_t^2] = h_t driven by rk_{t-1}
    omr, ar, br = _fit_heavy(rk[seg], rk[seg], np.mean(rk[fit_lo : fit_lo + 22]))  # measure equation
    hpath = _heavy_path(rk[fit_lo:], om, a, b, init)  # h[t] for t >= fit_lo
    mpath = _heavy_path(rk[fit_lo:], omr, ar, br, np.mean(rk[fit_lo : fit_lo + 22]))
    k = np.asarray(t_idx) - fit_lo
    h1 = om + a * rk[t_idx] + b * hpath[k]  # forecast of t+1 given t
    m1 = omr + ar * rk[t_idx] + br * mpath[k]
    out = {}
    H = max(horizons)
    hs, ms = [h1], [m1]
    for _ in range(1, H):
        hs.append(om + a * ms[-1] + b * hs[-1])
        ms.append(omr + (ar + br) * ms[-1])
    cum = np.cumsum(np.column_stack(hs), 1)
    for h in horizons:
        out[h] = cum[:, h - 1] / h
    return out


# PCA factor correlation model
def pca_factor_covs(R, S_ewma, t_idx, k=5, window=250):
    """Covariance forecasts D (V_k L_k V_k' + diag residual) D for every origin, D from the EWMA covariance."""
    out = np.empty((len(t_idx), R.shape[1], R.shape[1]))
    for j, t in enumerate(t_idx):
        C = np.corrcoef(R[t - window + 1 : t + 1].T)
        lam, V = np.linalg.eigh(C)
        Vk, lk = V[:, -k:], lam[-k:]
        Cf = (Vk * lk) @ Vk.T
        Cf[np.diag_indices_from(Cf)] = 1.0
        d = np.sqrt(np.diag(S_ewma[t]).astype(np.float64))
        out[j] = Cf * d[:, None] * d[None, :]
    return out
