"""Losses, Diebold-Mariano test, Model Confidence Set, Mincer-Zarnowitz and value-at-risk backtests."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pvol import evaluate as ev


def test_qlike_minimum_at_target():
    y = np.array([0.5, 1.0, 2.0])
    assert np.allclose(ev.qlike(y, y), 0)
    assert np.all(ev.qlike(y, y * 1.3) > 0) and np.all(ev.qlike(y, y * 0.7) > 0)


def test_qlike_floors_zero_target():
    assert np.isfinite(ev.qlike(np.array([0.0]), np.array([1.0]))).all()


def test_qlike_scale_invariant():
    y, f = np.array([0.3, 1.2]), np.array([0.5, 0.9])
    assert np.allclose(ev.qlike(y, f), ev.qlike(100 * y, 100 * f))


def test_mse_and_patton():
    y, f = np.array([1.0, 2.0]), np.array([1.5, 2.0])
    assert np.allclose(ev.mse(y, f), [0.25, 0.0])
    assert np.allclose(ev.patton_bm1(y, y), 0)
    assert np.all(ev.patton_bm1(y, f * 2) > 0)
    # homogeneous of degree 1
    assert np.allclose(ev.patton_bm1(3 * y, 3 * f), 3 * ev.patton_bm1(y, f))


def test_group_index_and_daily_loss():
    ports = [dict(group="A"), dict(group="B"), dict(group="A")]
    g = ev.group_index(ports)
    assert list(g["A"]) == [0, 2] and list(g["B"]) == [1]
    L = np.array([[1.0, 5.0, np.nan], [2.0, 6.0, 4.0]])
    assert np.allclose(ev.daily_loss(L, g["A"]), [1.0, 3.0])


def test_dm_test_size_and_power(rng):
    p0 = [ev.dm_test(rng.standard_normal(500), 1)[1] for _ in range(300)]
    assert 0.02 < np.mean(np.array(p0) < 0.05) < 0.10  # close to nominal size
    stat, p = ev.dm_test(rng.standard_normal(500) - 0.5, 1)
    assert stat < 0 and p < 1e-6


def test_dm_test_matches_t_test_without_autocorrelation(rng):
    """With h = 1 and no lags beyond the minimum the statistic is close to the one-sample t statistic."""
    d = rng.standard_normal(2000) + 0.1
    stat, _ = ev.dm_test(d, 1)
    t = stats.ttest_1samp(d, 0).statistic
    assert abs(stat - t) / abs(t) < 0.1


def test_dm_test_drops_nonfinite(rng):
    d = rng.standard_normal(300)
    a = ev.dm_test(d, 5)
    b = ev.dm_test(np.concatenate([d, [np.nan, np.inf]]), 5)
    assert np.allclose(a, b)


def test_dm_test_overlapping_targets(rng):
    """MA(h-1) differentials: the HAC variance keeps the size near nominal."""
    rej = []
    for _ in range(200):
        e = rng.standard_normal(1004)
        d = np.convolve(e, np.ones(5), "valid")
        rej.append(ev.dm_test(d, 5)[1] < 0.05)
    assert np.mean(rej) < 0.12


def test_mcs_keeps_best_model(rng):
    T = 400
    df = pd.DataFrame({"good": rng.standard_normal(T) ** 2, "bad": rng.standard_normal(T) ** 2 + 1.0})
    p = ev.mcs(df, reps=200)
    assert list(p.index) == ["good", "bad"]
    assert p["good"] == 1.0 and p["bad"] < 0.1


def test_mincer_zarnowitz_recovers_coefficients(rng):
    f = rng.uniform(1, 2, 1000)
    y = 0.2 + 0.8 * f
    a, b, r2 = ev.mincer_zarnowitz(y, np.where(np.arange(1000) == 3, np.nan, f))
    assert np.isclose(a, 0.2) and np.isclose(b, 0.8) and np.isclose(r2, 1.0)


def test_kupiec_pof():
    hits = np.zeros(1000, dtype=bool)
    hits[:10] = True
    lr, p = ev.kupiec_pof(hits, 0.01)
    assert np.isclose(lr, 0) and np.isclose(p, 1)
    assert ev.kupiec_pof(np.zeros(500, bool), 0.01)[1] < 0.01  # no hits at all
    assert ev.kupiec_pof(np.ones(50, bool), 0.01)[1] < 1e-10  # only hits
    hits[:50] = True
    assert ev.kupiec_pof(hits, 0.01)[1] < 1e-6


def test_christoffersen_detects_clustering(rng):
    iid = rng.random(2000) < 0.05
    assert ev.christoffersen_ind(iid)[1] > 0.01
    clustered = np.zeros(2000, bool)
    for s in range(0, 2000, 200):
        clustered[s : s + 10] = True
    assert ev.christoffersen_ind(clustered)[1] < 1e-6
    lr, p = ev.christoffersen_ind(np.zeros(100, bool))
    assert np.isclose(lr, 0) and np.isclose(p, 1)


def test_var_backtest(rng):
    T = 3000
    f = np.column_stack([np.full(T, 1e-4), np.full(T, 1e-4), np.full(T, np.nan)])
    r = rng.standard_normal((T, 3)) * 1e-2
    out = ev.var_backtest(r, f, 0.01)
    assert abs(out["hit_rate"] - 0.01) < 0.006
    assert out["rej_uc"] in (0.0, 0.5, 1.0)
    under = ev.var_backtest(r[:, :2], f[:, :2] / 4, 0.01)  # volatility understated by half
    assert under["hit_rate"] > 0.1 and under["rej_uc"] == 1.0


def test_load_run(tmp_path, monkeypatch):
    d = tmp_path / "main" / "h5"
    (d / "preds").mkdir(parents=True)
    y = np.ones((4, 2))
    np.save(d / "targets.npy", y)
    np.save(d / "origins.npy", np.arange(4))
    (tmp_path / "main" / "portfolios.json").write_text(json.dumps([dict(group="N5", ids=[0], weights=[1.0])] * 2))
    np.save(d / "preds" / "EWMA.npy", np.zeros((4, 2), np.float32))
    for s, v in ((0, np.log(1.0)), (1, np.log(3.0)), (5, np.log(100.0))):
        np.save(d / "preds" / f"MLP__s{s}.npy", np.full((4, 2), v, np.float32))
    np.save(d / "preds" / "MLP__s0__perm.npy", np.zeros((8, 2), np.float32))
    monkeypatch.setattr(ev, "RESULTS", tmp_path)
    R = ev.load_run("main", 5)
    assert set(R["F"]) == {"EWMA", "MLP"} and sorted(R["F_seed"]["MLP"]) == [0, 1]
    assert np.allclose(R["F"]["EWMA"], 1.0)
    assert np.allclose(R["F"]["MLP"], 2.0)  # ensemble averages variances, not logs
    assert sorted(ev.load_run("main", 5, seeds=None)["F_seed"]["MLP"]) == [0, 1, 5]
    assert len(R["ports"]) == 2 and R["origins"].tolist() == [0, 1, 2, 3]


def test_model_lists_consistent():
    assert ev.INVARIANT <= set(ev.NEURAL)
    assert not set(ev.CLASSICAL) & set(ev.NEURAL)
    assert {"MLP", "GRU"} == set(ev.NEURAL) - ev.INVARIANT


@pytest.mark.parametrize("h", [1, 5, 22])
def test_dm_lag_grows_with_h(h, rng):
    d = rng.standard_normal(1500)
    stat, p = ev.dm_test(d, h)
    assert np.isfinite(stat) and 0 <= p <= 1
