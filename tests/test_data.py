"""Panel construction: variance proxies, features, correlations, EWMA covariance, the stratified subsample."""
import numpy as np
import pandas as pd
import pytest

from pvol import data
from pvol.config import EWMA_LAMBDA, HAR_WINDOWS, VAR_FLOOR
from pvol.universe import ASSET_CLASSES, ETF_UNIVERSE, UNIVERSE


def prices(rng, T=300, M=3):
    C = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (T, M)), 0))
    O = C * np.exp(rng.normal(0, 0.003, (T, M)))
    H = np.maximum(O, C) * np.exp(np.abs(rng.normal(0, 0.004, (T, M))))
    L = np.minimum(O, C) * np.exp(-np.abs(rng.normal(0, 0.004, (T, M))))
    idx = pd.bdate_range("2010-01-04", periods=T)
    f = lambda a: pd.DataFrame(a, index=idx, columns=[f"A{i}" for i in range(M)])
    return f(O), f(H), f(L), f(C)


def test_variance_proxies(rng):
    O, H, L, C = prices(rng)
    R, sq, rngv = data.variance_proxies(O, H, L, C)
    assert np.isnan(R.iloc[0]).all() and np.allclose(sq.iloc[1:], R.iloc[1:] ** 2)
    assert (rngv.iloc[1:] >= 0).all().all()
    t, a = 5, "A1"
    o, hi, lo, c, c0 = O[a].iloc[t], H[a].iloc[t], L[a].iloc[t], C[a].iloc[t], C[a].iloc[t - 1]
    gk = 0.5 * np.log(hi / lo) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    assert np.isclose(rngv[a].iloc[t], np.log(o / c0) ** 2 + gk)


def test_rolling_mean_needs_full_window():
    x = pd.DataFrame({"a": np.arange(10.0)})
    m = data.rolling_mean(x, 3)
    assert m["a"].isna().sum() == 2 and m["a"].iloc[2] == 1.0


def test_node_features(rng):
    O, H, L, C = prices(rng)
    R, sq, rngv = data.variance_proxies(O, H, L, C)
    X, names = data.node_features(R, sq, rngv)
    assert X.shape == (300, 3, len(names)) and len(names) == 2 * len(HAR_WINDOWS) + 4
    t = 150
    i = names.index("log_sq_5")
    assert np.isclose(X[t, 0, i], np.log(max(sq["A0"].iloc[t - 4 : t + 1].mean(), VAR_FLOOR)))
    j = names.index("downside_share_22")
    assert np.all(np.abs(X[100:, :, j]) <= 0.5 + 1e-12)
    z = names.index("zret_1")
    vol = np.sqrt(sq["A2"].iloc[t - 21 : t + 1].mean())
    assert np.isclose(X[t, 2, z], R["A2"].iloc[t] / vol)


def test_asset_class_onehot():
    tickers = list(UNIVERSE)[:5]
    oh = data.asset_class_onehot(tickers)
    assert oh.shape == (5, len(ASSET_CLASSES)) and np.all(oh.sum(1) == 1)
    assert oh[0, ASSET_CLASSES.index(UNIVERSE[tickers[0]])] == 1


def test_rolling_corr_matches_pandas(rng):
    *_, C = prices(rng, T=120)
    R = C.pct_change()
    out = data.rolling_corr(R, 22)
    assert np.isnan(out[21]).all()
    ref = R.iloc[100 - 21 : 101].corr().to_numpy()
    assert np.allclose(out[100], ref, atol=1e-8)


def test_ewma_cov_recursion(rng):
    R = pd.DataFrame(rng.normal(0, 0.01, (60, 2)))
    R.iloc[0] = np.nan
    S = data.ewma_cov(R, init=22)
    assert np.isnan(S[21]).all()
    X = np.nan_to_num(R.to_numpy())
    assert np.allclose(S[22], X[1:23].T @ X[1:23] / 22)
    assert np.allclose(S[40], EWMA_LAMBDA * S[39] + (1 - EWMA_LAMBDA) * np.outer(X[40], X[40]))


def test_cov_to_corr():
    S = np.array([[[4.0, 2.0], [2.0, 9.0]]])
    assert np.allclose(data.cov_to_corr(S), [[[1, 1 / 3], [1 / 3, 1]]])


def test_stratified_subsample_is_proportional(monkeypatch):
    uni = {f"S{i}": ("a" if i < 60 else "b" if i < 90 else "c") for i in range(100)}
    monkeypatch.setattr(data, "UNIVERSE", uni)
    sub = data.stratified_subsample(list(uni), 10)
    counts = pd.Series([uni[t] for t in sub]).value_counts()
    assert len(sub) == 10 and counts["a"] == 6 and counts["b"] == 3 and counts["c"] == 1
    assert sub == [t for t in uni if t in set(sub)]  # original order kept
    assert data.stratified_subsample(list(uni), 10) == sub  # fixed seed


def write_raw(tmp_path, tickers, rng, T=400, missing=None):
    O, H, L, C = prices(rng, T=T, M=len(tickers))
    frames = {}
    for k, v in zip(("Open", "High", "Low", "Close"), (O, H, L, C)):
        v = v.copy()
        v.columns = tickers
        v.index = pd.bdate_range("2008-01-02", periods=T)
        if missing:
            v.iloc[:5, v.columns.get_loc(missing)] = np.nan
        frames[k] = v
    df = pd.concat(frames, axis=1)
    raw = tmp_path / "raw"
    raw.mkdir()
    df.to_parquet(raw / "ohlcv_adjusted.parquet")
    return raw


def test_build_and_load_panel(tmp_path, monkeypatch, rng):
    tickers = list(ETF_UNIVERSE)
    raw = write_raw(tmp_path, tickers, rng)
    monkeypatch.setattr(data, "RAW", raw)
    monkeypatch.setattr(data, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(data, "UNIVERSE", ETF_UNIVERSE)
    monkeypatch.setattr(data, "UNIVERSE_NAME", "etf")
    panel = data.build_panel()
    T, M = 400, len(tickers)
    assert panel["R"].shape == (T, M) and panel["P"].shape == (T, M, M, 3)
    fv = int(panel["first_valid"])
    assert fv == max(HAR_WINDOWS)  # 66 days of history for the quarterly window
    assert not np.isnan(panel["X"][fv:]).any() and not np.isnan(panel["P"][fv:]).any()
    loaded = data.load_panel()
    assert set(loaded) == set(panel) and np.array_equal(loaded["X"], panel["X"], equal_nan=True)
    assert list(loaded["tickers"]) == tickers


def test_load_prices_drops_incomplete_and_subsamples(tmp_path, monkeypatch, rng, capsys):
    uni = {f"S{i}": ("a" if i < 6 else "b") for i in range(9)}
    raw = write_raw(tmp_path, list(uni), rng, T=50, missing="S2")
    monkeypatch.setattr(data, "RAW", raw)
    monkeypatch.setattr(data, "UNIVERSE", uni)
    monkeypatch.setattr(data, "UNIVERSE_NAME", "stocks")
    monkeypatch.setattr(data, "SUBSAMPLE", 0)
    _, tickers = data.load_prices()
    assert "S2" not in tickers and len(tickers) == 8 and "dropping 1" in capsys.readouterr().out
    monkeypatch.setattr(data, "SUBSAMPLE", 4)
    _, sub = data.load_prices()
    assert len(sub) == 4 and set(sub) <= set(tickers)


def test_load_prices_rejects_gaps_in_etf_universe(tmp_path, monkeypatch, rng):
    uni = {"A": "a", "B": "a"}
    monkeypatch.setattr(data, "RAW", write_raw(tmp_path, list(uni), rng, T=30, missing="B"))
    monkeypatch.setattr(data, "UNIVERSE", uni)
    monkeypatch.setattr(data, "UNIVERSE_NAME", "etf")
    with pytest.raises(AssertionError):
        data.load_prices()
