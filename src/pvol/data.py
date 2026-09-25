"""Build the asset-level panel: returns, variance proxies, node features and pairwise dependence tensors.

Everything indexed by t uses information available at the close of day t only.
"""
import numpy as np
import pandas as pd

from .config import (
    CORR_WINDOWS,
    EWMA_LAMBDA,
    HAR_WINDOWS,
    PROCESSED,
    RAW,
    SAMPLE_END,
    SAMPLE_START,
    UNIVERSE_NAME,
    VAR_FLOOR,
)
from .universe import ASSET_CLASSES, SUBSAMPLE, UNIVERSE


def stratified_subsample(tickers, n, seed=0):
    rng = np.random.default_rng(seed)
    by = {}
    for t in tickers:
        by.setdefault(UNIVERSE[t], []).append(t)
    quota = {c: n * len(v) / len(tickers) for c, v in by.items()}
    k = {c: int(q) for c, q in quota.items()}
    for c in sorted(quota, key=lambda c: quota[c] - k[c], reverse=True)[: n - sum(k.values())]:
        k[c] += 1
    keep = {t for c, v in by.items() for t in rng.choice(sorted(v), k[c], replace=False)}
    return [t for t in tickers if t in keep]


def load_prices():
    df = pd.read_parquet(RAW / "ohlcv_adjusted.parquet").loc[SAMPLE_START:SAMPLE_END]
    tickers = list(UNIVERSE)
    if UNIVERSE_NAME != "etf":  # robustness universes: keep only names with a complete history in the sample
        complete = [t for t in tickers if all(df[k][t].notna().all() for k in ("Open", "High", "Low", "Close"))]
        dropped = sorted(set(tickers) - set(complete))
        if dropped:
            print(f"dropping {len(dropped)} names without complete history: {' '.join(dropped)}")
        tickers = complete
        if SUBSAMPLE:  # stratified by sector, proportional allocation (largest remainders), fixed seed
            tickers = stratified_subsample(tickers, SUBSAMPLE)
            print(f"subsample of {len(tickers)} stocks: {' '.join(tickers)}")
    fields = {k: df[k][tickers].astype("float64") for k in ("Open", "High", "Low", "Close")}
    assert not any(v.isna().any().any() for v in fields.values()), "missing prices in common sample"
    return fields, tickers


def variance_proxies(O, H, L, C):
    """Daily variance proxies per asset.

    sq:  squared close-to-close simple return.
    rng: overnight squared log return + Garman-Klass (1980) open-to-close range estimator.
    """
    R = C.pct_change()
    sq = R**2
    u = np.log(H / L)
    c = np.log(C / O)
    gk = 0.5 * u**2 - (2 * np.log(2) - 1) * c**2  # >= 0 because |c| <= u
    overnight = np.log(O / C.shift(1)) ** 2
    rng = overnight + gk
    return R, sq, rng


def rolling_mean(x, w):
    return x.rolling(w, min_periods=w).mean()


def node_features(R, sq, rng):
    """Per-asset features at t, shape (T, M, F), plus their names."""
    feats, names = [], []
    for proxy_name, proxy in (("sq", sq), ("rng", rng)):
        for w in HAR_WINDOWS:
            feats.append(np.log(np.maximum(rolling_mean(proxy, w), VAR_FLOOR)))
            names.append(f"log_{proxy_name}_{w}")
    for w in (1, 5, 22):
        # cumulative return over the window scaled by monthly vol: sign/size of recent moves (leverage effect)
        cum = R.rolling(w, min_periods=w).sum()
        vol = np.sqrt(np.maximum(rolling_mean(sq, 22), VAR_FLOOR))
        feats.append(cum / (vol * np.sqrt(w)))
        names.append(f"zret_{w}")
    neg = (sq * (R < 0)).rolling(22, min_periods=22).sum()
    tot = sq.rolling(22, min_periods=22).sum()
    feats.append(neg / np.maximum(tot, VAR_FLOOR) - 0.5)
    names.append("downside_share_22")
    X = np.stack([f.to_numpy() for f in feats], axis=-1)  # (T, M, F_num)
    return X, names


def asset_class_onehot(tickers):
    idx = {c: k for k, c in enumerate(ASSET_CLASSES)}
    oh = np.zeros((len(tickers), len(ASSET_CLASSES)))
    for i, t in enumerate(tickers):
        oh[i, idx[UNIVERSE[t]]] = 1.0
    return oh


def rolling_corr(R, w):
    """Rolling-window (demeaned) correlation matrices, shape (T, M, M); NaN before the window fills."""
    X = R.to_numpy()
    X = np.nan_to_num(X)
    T, M = X.shape
    cs_xx = np.cumsum(np.einsum("ti,tj->tij", X, X), axis=0)
    cs_x = np.cumsum(X, axis=0)
    out = np.full((T, M, M), np.nan)
    for t in range(w, T):  # returns start at t=1, so a full window ends at t >= w
        sxx = cs_xx[t] - cs_xx[t - w]
        sx = cs_x[t] - cs_x[t - w]
        cov = sxx / w - np.outer(sx, sx) / w**2
        sd = np.sqrt(np.maximum(np.diag(cov), 1e-16))
        out[t] = cov / np.outer(sd, sd)
    return out


def ewma_cov(R, lam=EWMA_LAMBDA, init=22):
    """RiskMetrics EWMA covariance (zero mean), shape (T, M, M). Sigma_t uses returns up to and including t."""
    X = np.nan_to_num(R.to_numpy())
    T, M = X.shape
    out = np.full((T, M, M), np.nan)
    S = X[1 : init + 1].T @ X[1 : init + 1] / init
    out[init] = S
    for t in range(init + 1, T):
        S = lam * S + (1 - lam) * np.outer(X[t], X[t])
        out[t] = S
    return out


def cov_to_corr(S):
    sd = np.sqrt(np.maximum(np.einsum("tii->ti", S), 1e-16))
    return S / (sd[:, :, None] * sd[:, None, :])


def build_panel():
    fields, tickers = load_prices()
    R, sq, rng = variance_proxies(fields["Open"], fields["High"], fields["Low"], fields["Close"])
    X, feat_names = node_features(R, sq, rng)
    S_ewma = ewma_cov(R)
    corr = [rolling_corr(R, w) for w in CORR_WINDOWS] + [cov_to_corr(S_ewma)]
    P = np.stack(corr, axis=-1)  # (T, M, M, 3)
    dates = R.index
    first_valid = int(np.argmax(~np.isnan(X).any(axis=(1, 2)) & ~np.isnan(P).any(axis=(1, 2, 3))))
    panel = dict(
        dates=dates.strftime("%Y-%m-%d").to_numpy().astype("U10"),
        tickers=np.array(tickers),
        asset_class=np.array([UNIVERSE[t] for t in tickers]),
        R=np.nan_to_num(R.to_numpy()).astype(np.float32),
        X=X.astype(np.float32),
        feat_names=np.array(feat_names),
        class_onehot=asset_class_onehot(tickers).astype(np.float32),
        P=P.astype(np.float32),
        pair_names=np.array([f"corr_{w}" for w in CORR_WINDOWS] + ["corr_ewma"]),
        S_ewma=S_ewma.astype(np.float32),
        first_valid=np.array(first_valid),
    )
    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROCESSED / "panel.npz", **panel)
    return panel


def load_panel():
    with np.load(PROCESSED / "panel.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
