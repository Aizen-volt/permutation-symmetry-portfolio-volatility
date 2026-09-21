"""Classical benchmarks. All return forecasts of the average daily portfolio variance over t+1..t+h.

Univariate models work on the constant-weight backcast portfolio return series r_p = R[:, ids] @ w.
DCC-GARCH is a multivariate model of the whole universe; portfolio forecasts are w' Sigma w.
Parameters are estimated on data up to the end of test_year-1 and kept fixed during test_year,
while the filtered state (EWMA, GARCH variance, DCC correlation) is updated daily with realised returns.
"""
import warnings

import numpy as np
from arch import arch_model
from joblib import Parallel, delayed
from scipy.optimize import minimize

from .config import EWMA_LAMBDA, VAR_FLOOR

SCALE = 100.0  # arch works best with returns in percent


def portfolio_returns(R, p):
    return R[:, p["ids"]] @ np.asarray(p["weights"], dtype=np.float64)


# ------------------------------------------------------------------ simple filters
def hist_forecast(rp, t_idx, window=22):
    c = np.concatenate([[0.0], np.cumsum(rp**2)])
    return (c[t_idx + 1] - c[t_idx + 1 - window]) / window


def ewma_forecast(S_ewma, p, t_idx):
    w = np.asarray(p["weights"])
    ids = p["ids"]
    ix = np.asarray(ids)
    Ssub = S_ewma[np.asarray(t_idx)[:, None, None], ix[None, :, None], ix[None, None, :]].astype(np.float64)
    return np.einsum("i,tij,j->t", w, Ssub, w)


# ------------------------------------------------------------------ HAR (Corsi, 2009) on the portfolio series
def har_design(rp, t_idx):
    c = np.concatenate([[0.0], np.cumsum(rp**2)])
    cols = [(c[t_idx + 1] - c[t_idx + 1 - k]) / k for k in (1, 5, 22)]
    return np.column_stack([np.ones(len(t_idx))] + cols)


def har_forecast(rp, fit_idx, t_idx, h):
    c = np.concatenate([[0.0], np.cumsum(rp**2)])
    y = (c[fit_idx + h + 1] - c[fit_idx + 1]) / h
    beta, *_ = np.linalg.lstsq(har_design(rp, fit_idx), y, rcond=None)
    f = har_design(rp, t_idx) @ beta
    fallback = hist_forecast(rp, t_idx)
    return np.where(f > VAR_FLOOR, f, fallback), beta


# ------------------------------------------------------------------ HAR-X: HAR + range-based portfolio regressors
def range_portfolio_var(X, P, p, t_idx, feat_idx):
    """w' D_k R_ewma D_k w, with D_k = diag(sqrt(mean range-based variance over k days)); one column per k."""
    ids, w = p["ids"], np.asarray(p["weights"])
    ix = np.asarray(ids)
    Rc = P[np.asarray(t_idx)[:, None, None], ix[None, :, None], ix[None, None, :], 2].astype(np.float64)  # EWMA corr
    cols = []
    for f in feat_idx:
        sd = np.sqrt(np.exp(X[np.asarray(t_idx)[:, None], ix[None, :], f].astype(np.float64)))  # (T, N)
        ws = w * sd
        cols.append(np.einsum("ti,tij,tj->t", ws, Rc, ws))
    return np.column_stack(cols)


def harx_forecast(rp, X, P, p, fit_idx, t_idx, h, feat_idx):
    c = np.concatenate([[0.0], np.cumsum(rp**2)])
    y = (c[fit_idx + h + 1] - c[fit_idx + 1]) / h
    Z = lambda idx: np.column_stack([har_design(rp, idx), range_portfolio_var(X, P, p, idx, feat_idx)])
    beta, *_ = np.linalg.lstsq(Z(fit_idx), y, rcond=None)
    f = Z(t_idx) @ beta
    return np.where(f > VAR_FLOOR, f, hist_forecast(rp, t_idx)), beta


# ------------------------------------------------------------------ GARCH / GJR-GARCH(1,1)
def fit_garch(r, o=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(r * SCALE, mean="Zero", vol="GARCH", p=1, o=o, q=1, dist="normal").fit(disp="off")
    prm = res.params
    return prm["omega"], prm["alpha[1]"], prm.get("gamma[1]", 0.0), prm["beta[1]"]


def garch_filter(r, params, init_var=None):
    """sigma2[t] = conditional variance of r[t] given info up to t-1 (in SCALE^2 units); also one-step-ahead
    variance for day t+1 given info up to t, returned as nxt[t]."""
    omega, alpha, gamma, beta = params
    x = r * SCALE
    n = len(x)
    s2 = np.empty(n)
    s2[0] = np.mean(x[:22] ** 2) if init_var is None else init_var  # no look-ahead in the initial state
    for t in range(1, n):
        s2[t] = omega + (alpha + gamma * (x[t - 1] < 0)) * x[t - 1] ** 2 + beta * s2[t - 1]
    nxt = omega + (alpha + gamma * (x < 0)) * x**2 + beta * s2
    return nxt


def garch_multistep(nxt, params, h):
    """Average of the k=1..h step-ahead variance forecasts, given one-step forecasts nxt."""
    omega, alpha, gamma, beta = params
    persist = alpha + gamma / 2 + beta
    if persist < 1:
        lr = omega / (1 - persist)
        k = np.arange(h)
        avg_decay = np.mean(persist**k)
        f = lr + avg_decay * (nxt - lr)
    else:
        f = nxt
    return f / SCALE**2


def garch_portfolio(rp, fit_end, t_idx, h, o=0, first=1):
    params = fit_garch(rp[first : fit_end + 1], o=o)
    nxt = garch_filter(rp[first:], params)
    return garch_multistep(nxt[t_idx - first], params, h), params


def run_garch_many(R, portfolios, fit_end, t_idx, h, o=0, n_jobs=-1):
    def one(p):
        rp = portfolio_returns(R, p)
        return garch_portfolio(rp, fit_end, t_idx, h, o=o)[0]

    cols = Parallel(n_jobs=n_jobs)(delayed(one)(p) for p in portfolios)
    return np.column_stack(cols)


# ------------------------------------------------------------------ DCC-GARCH (Engle, 2002)
class DCC:
    """Two-step DCC(1,1)-GARCH(1,1) with correlation targeting, estimated on the whole universe."""

    def fit(self, R, fit_end, first=1, n_jobs=-1, target="sample"):
        """target='sample': correlation targeting with the sample correlation of the standardised residuals
        (Engle, 2002); target='nls': the target is shrunk with analytical nonlinear shrinkage (DCC-NL of
        Engle, Ledoit and Wolf, 2019)."""
        self.first = first
        Rf = R[first : fit_end + 1]
        self.uni = Parallel(n_jobs=n_jobs)(delayed(fit_garch)(Rf[:, i]) for i in range(R.shape[1]))
        nxt_all = np.column_stack([garch_filter(R[first:, i], self.uni[i]) for i in range(R.shape[1])])
        # conditional variance of r[t] given t-1 is nxt[t-1]; first obs uses the unconditional sample var
        s2 = np.vstack([np.mean((R[first : first + 22] * SCALE) ** 2, axis=0, keepdims=True), nxt_all[:-1]])
        self.nxt = nxt_all  # one-step variance for day t+1 given info at t (SCALE^2)
        self.z = R[first:] * SCALE / np.sqrt(s2)
        zf = self.z[: fit_end + 1 - first]
        self.Qbar = np.corrcoef(zf.T) if target == "sample" else cov_to_corr_np(analytical_shrinkage(zf))
        # unconstrained reparametrisation: a + b = 0.999 * sigmoid(u), a = (a + b) * sigmoid(v)
        def ab(x):
            pers = 0.999 / (1 + np.exp(-x[0]))
            a = pers / (1 + np.exp(-x[1]))
            return np.array([a, pers - a])

        x0 = np.array([np.log(0.97 / 0.029), np.log(0.02 / 0.95)])  # a=0.02, b=0.95
        res = minimize(lambda x: -self._loglik(ab(x), zf) / len(zf), x0, method="Nelder-Mead",
                       options=dict(xatol=1e-4, fatol=1e-7, maxiter=400))
        self.a, self.b = ab(res.x)
        self.opt = res
        self.Q = self._filter_Q(self.a, self.b, self.z)  # Q[t] = Q_{t+1|t}
        return self

    def _filter_Q(self, a, b, z):
        n, M = z.shape
        Q = np.empty((n, M, M))
        q = self.Qbar.copy()
        c = (1 - a - b) * self.Qbar
        for t in range(n):
            q = c + a * np.outer(z[t], z[t]) + b * q  # Q_{t+1} from z_t
            Q[t] = q
        return Q

    def _loglik(self, x, z):
        a, b = x
        if a + b >= 0.999:
            return -1e12
        Q = self._filter_Q(a, b, z)[:-1]  # Q_t used for z_t, t>=1
        d = np.sqrt(np.einsum("tii->ti", Q))
        Rt = Q / (d[:, :, None] * d[:, None, :])
        L = np.linalg.cholesky(Rt)
        u = np.linalg.solve(L, z[1:, :, None])[..., 0]
        logdet = 2 * np.log(np.einsum("tii->ti", L)).sum(1)
        return -0.5 * (logdet + (u**2).sum(1) - (z[1:] ** 2).sum(1)).sum()

    def covariance_forecast(self, t, h):
        """Average over k=1..h of the k-step covariance forecast made at origin t (Engle & Sheppard, 2001)."""
        i = t - self.first
        Q1 = self.Q[i]
        d = np.sqrt(np.diag(Q1))
        R1 = Q1 / np.outer(d, d)
        pers = self.a + self.b
        out = np.zeros_like(R1)
        for k in range(h):
            Rk = (1 - pers**k) * self.Qbar + pers**k * R1
            dk = np.sqrt(np.diag(Rk))
            Rk = Rk / np.outer(dk, dk)
            v = np.array([self._uni_kstep(j, i, k) for j in range(len(d))])
            sd = np.sqrt(v)
            out += Rk * np.outer(sd, sd)
        return out / h / SCALE**2

    def _uni_kstep(self, j, i, k):
        omega, alpha, gamma, beta = self.uni[j]
        pers = alpha + gamma / 2 + beta
        if pers >= 1:  # integrated GARCH: flat term structure
            return self.nxt[i, j]
        lr = omega / (1 - pers)
        return lr + pers**k * (self.nxt[i, j] - lr)


def dcc_portfolio_forecasts(dcc, portfolios, t_idx, h):
    out = np.empty((len(t_idx), len(portfolios)))
    for r, t in enumerate(t_idx):
        Sig = dcc.covariance_forecast(t, h)
        for j, p in enumerate(portfolios):
            w = np.asarray(p["weights"])
            ids = p["ids"]
            out[r, j] = w @ Sig[np.ix_(ids, ids)] @ w
    return out


# ------------------------------------------------------------------ shrinkage estimators
def analytical_shrinkage(X, demean=True):
    """Analytical nonlinear shrinkage of the covariance matrix (Ledoit and Wolf, 2020, Annals of Statistics),
    for p <= n. X: (n, p) observations."""
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    if demean:
        X = X - X.mean(0)
        n = n - 1
    if p > n:
        raise ValueError("analytical_shrinkage implemented for p <= n only")
    sample = X.T @ X / n
    lam, u = np.linalg.eigh(sample)
    lam = np.maximum(lam, 1e-18)
    L = np.tile(lam[:, None], (1, p))  # L[i, j] = lam_i
    h = n ** (-1 / 3)
    H = h * L.T  # H[i, j] = h lam_j
    x = (L - L.T) / H
    s5 = np.sqrt(5.0)
    ftilde = (3 / 4 / s5) * np.mean(np.maximum(1 - x**2 / 5, 0) / H, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Hftemp = (-3 / 10 / np.pi) * x + (3 / 4 / s5 / np.pi) * (1 - x**2 / 5) * np.log(np.abs((s5 - x) / (s5 + x)))
    edge = np.isclose(np.abs(x), s5)
    Hftemp[edge] = (-3 / 10 / np.pi) * x[edge]
    Hftilde = np.mean(Hftemp / H, axis=1)
    c = p / n
    dtilde = lam / ((np.pi * c * lam * ftilde) ** 2 + (1 - c - np.pi * c * lam * Hftilde) ** 2)
    return (u * dtilde) @ u.T


def cov_to_corr_np(S):
    d = np.sqrt(np.diag(S))
    C = S / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return C


def ledoit_wolf_2004(X):
    """Linear shrinkage towards a scaled identity (Ledoit and Wolf, 2004, JMVA)."""
    from sklearn.covariance import ledoit_wolf

    return ledoit_wolf(X)[0]


def rolling_shrinkage_covs(R, t_idx, window=250, kind="lw", S_ewma=None):
    """Covariance forecasts made at each origin t from the last `window` daily returns (up to and including t).
    kind='lw': Ledoit-Wolf (2004) covariance; kind='ewma-nls': EWMA volatilities with the correlation matrix
    from analytical nonlinear shrinkage of the window's covariance."""
    out = np.empty((len(t_idx), R.shape[1], R.shape[1]))
    for k, t in enumerate(t_idx):
        X = R[t - window + 1 : t + 1]
        if kind == "lw":
            out[k] = ledoit_wolf_2004(X)
        elif kind == "ewma-nls":
            C = cov_to_corr_np(analytical_shrinkage(X))
            sd = np.sqrt(np.diag(S_ewma[t]).astype(np.float64))
            out[k] = C * np.outer(sd, sd)
        else:
            raise ValueError(kind)
    return out


def quad_forms(Sig, portfolios):
    """w' Sigma_t w for every origin (rows of Sig) and portfolio."""
    out = np.empty((Sig.shape[0], len(portfolios)))
    for j, p in enumerate(portfolios):
        ids, w = p["ids"], np.asarray(p["weights"])
        out[:, j] = np.einsum("i,tij,j->t", w, Sig[:, ids][:, :, ids], w)
    return out
