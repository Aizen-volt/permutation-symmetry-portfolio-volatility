"""Classical benchmarks: simple filters, HAR, HAR-X, GARCH, DCC and shrinkage estimators."""
import numpy as np
import pytest

from conftest import simulate_garch
from pvol import baselines as bl
from pvol.config import VAR_FLOOR


def toy_panel(rng, T=400, M=4):
    R = rng.standard_normal((T, M)) * 0.01
    X = np.log(np.full((T, M, 3), 1e-4)) + rng.normal(0, 0.1, (T, M, 3))
    C = np.corrcoef(R.T)
    P = np.broadcast_to(C[None, :, :, None], (T, M, M, 3)).copy()
    return R, X, P


def test_portfolio_returns(rng):
    R = rng.standard_normal((10, 4))
    p = dict(ids=[2, 0], weights=[0.25, 0.75])
    assert np.allclose(bl.portfolio_returns(R, p), 0.25 * R[:, 2] + 0.75 * R[:, 0])


def test_hist_forecast_is_trailing_mean(rng):
    rp = rng.standard_normal(100)
    t = np.array([30, 50, 99])
    ref = [np.mean(rp[k - 21 : k + 1] ** 2) for k in t]
    assert np.allclose(bl.hist_forecast(rp, t), ref)
    assert np.allclose(bl.hist_forecast(rp, t, window=5), [np.mean(rp[k - 4 : k + 1] ** 2) for k in t])


def test_ewma_forecast_is_quadratic_form(rng):
    A = rng.standard_normal((6, 5, 5))
    S = A @ A.transpose(0, 2, 1)
    p = dict(ids=[4, 1, 2], weights=[0.5, 0.3, 0.2])
    t = np.array([1, 3, 5])
    w = np.array(p["weights"])
    ref = [w @ S[k][np.ix_(p["ids"], p["ids"])] @ w for k in t]
    assert np.allclose(bl.ewma_forecast(S, p, t), ref)


def test_har_design_columns(rng):
    rp = rng.standard_normal(60)
    Z = bl.har_design(rp, np.array([40]))
    assert np.allclose(Z[0], [1, rp[40] ** 2, np.mean(rp[36:41] ** 2), np.mean(rp[19:41] ** 2)])


def test_har_forecast_fit_uses_only_targets_inside():
    rp = simulate_garch(1500, 0.05, 0.08, 0.9)
    fit = np.arange(30, 1000)
    t = np.arange(1000, 1400)
    f, beta = bl.har_forecast(rp, fit, t, 5)
    assert beta.shape == (4,) and np.all(f > 0)
    rp2 = rp.copy()
    rp2[1006:] = 0.5  # changes after the last fit target leave the coefficients unchanged
    assert np.allclose(bl.har_forecast(rp2, fit, t, 5)[1], beta)


def test_har_forecast_falls_back_to_hist():
    """A fitted value below the variance floor is replaced by the 22-day historical variance."""
    rp = np.zeros(300)
    rp[200:] = 0.01  # no variation in the fit sample: the HAR fit predicts zero
    t = np.arange(240, 260)
    f, _ = bl.har_forecast(rp, np.arange(30, 150), t, 1)
    assert np.allclose(f, 1e-4)


def test_range_portfolio_var_matches_loop(rng):
    _, X, P = toy_panel(rng)
    p = dict(ids=[3, 1], weights=[0.6, 0.4])
    t = np.array([10, 20])
    out = bl.range_portfolio_var(X, P, p, t, [0, 2])
    w = np.array(p["weights"])
    for r, k in enumerate(t):
        for c, f in enumerate([0, 2]):
            sd = np.sqrt(np.exp(X[k, p["ids"], f]))
            Rc = P[k][np.ix_(p["ids"], p["ids"])][:, :, 2]
            assert np.isclose(out[r, c], (w * sd) @ Rc @ (w * sd))


def test_harx_forecast(rng):
    R, X, P = toy_panel(rng, T=600)
    p = dict(ids=[0, 1, 2], weights=[0.3, 0.3, 0.4])
    rp = bl.portfolio_returns(R, p)
    f, beta = bl.harx_forecast(rp, X, P, p, np.arange(30, 400), np.arange(400, 550), 5, [0, 1, 2])
    assert beta.shape == (7,) and f.shape == (150,) and np.all(f > VAR_FLOOR)


def test_fit_garch_recovers_parameters():
    r = simulate_garch(6000, 0.05, 0.08, 0.9, seed=1)
    _, a, g, b = bl.fit_garch(r)
    assert g == 0.0 and abs(a - 0.08) < 0.04 and abs(b - 0.9) < 0.05
    _, a, g, b = bl.fit_garch(simulate_garch(6000, 0.05, 0.03, 0.88, gamma=0.12, seed=2), o=1)
    assert g > 0.04


def test_garch_filter_recursion(rng):
    r = rng.standard_normal(50) * 0.01
    prm = (0.02, 0.1, 0.05, 0.85)
    nxt = bl.garch_filter(r, prm, init_var=1.0)
    x = r * bl.SCALE
    s2 = 1.0
    for t in range(50):
        if t:
            s2 = 0.02 + (0.1 + 0.05 * (x[t - 1] < 0)) * x[t - 1] ** 2 + 0.85 * s2
        assert np.isclose(nxt[t], 0.02 + (0.1 + 0.05 * (x[t] < 0)) * x[t] ** 2 + 0.85 * s2)
    assert np.isclose(bl.garch_filter(r, prm)[0], 0.02 + (0.1 + 0.05 * (x[0] < 0)) * x[0] ** 2
                      + 0.85 * np.mean(x[:22] ** 2))


def test_garch_multistep():
    nxt = np.array([2.0, 4.0])
    prm = (0.1, 0.1, 0.0, 0.8)
    assert np.allclose(bl.garch_multistep(nxt, prm, 1), nxt / bl.SCALE**2)
    lr = 0.1 / 0.1
    f22 = bl.garch_multistep(nxt, prm, 22) * bl.SCALE**2
    assert np.all(np.abs(f22 - lr) < np.abs(nxt - lr))  # pulled towards the long-run variance
    assert np.allclose(bl.garch_multistep(nxt, (0.1, 0.1, 0.0, 0.9), 22), nxt / bl.SCALE**2)  # integrated


def test_garch_portfolio_and_many():
    R = np.column_stack([simulate_garch(1200, 0.05, 0.08, 0.9, seed=s) for s in range(3)])
    ports = [dict(ids=[0, 1], weights=[0.5, 0.5]), dict(ids=[2], weights=[1.0])]
    t = np.arange(900, 1000)
    many = bl.run_garch_many(R, ports, 899, t, 5, n_jobs=1)
    one, prm = bl.garch_portfolio(bl.portfolio_returns(R, ports[1]), 899, t, 5)
    assert many.shape == (100, 2) and np.allclose(many[:, 1], one) and len(prm) == 4


@pytest.fixture(scope="module")
def dcc_fit():
    g = np.random.default_rng(3)
    T, M = 900, 3
    L = np.linalg.cholesky(np.array([[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]]))
    vol = np.column_stack([np.sqrt(simulate_garch(T, 0.05, 0.08, 0.9, seed=s) ** 2 + 1e-6) for s in range(M)])
    R = (g.standard_normal((T, M)) @ L.T) * vol
    return R, bl.DCC().fit(R, 700, n_jobs=1)


def test_dcc_fit(dcc_fit):
    R, dcc = dcc_fit
    assert 0 <= dcc.a and 0 <= dcc.b and dcc.a + dcc.b < 0.999
    assert dcc.Q.shape == (len(R) - 1, 3, 3) and np.allclose(np.diag(dcc.Qbar), 1)
    assert dcc._loglik(np.array([0.5, 0.5]), dcc.z[:100]) == -1e12


def test_dcc_covariance_forecast(dcc_fit):
    R, dcc = dcc_fit
    S1 = dcc.covariance_forecast(800, 1)
    assert np.allclose(np.diag(S1), dcc.nxt[800 - dcc.first] / bl.SCALE**2)
    S22 = dcc.covariance_forecast(800, 22)
    assert np.allclose(S22, S22.T) and np.all(np.linalg.eigvalsh(S22) > 0)
    ports = [dict(ids=[0, 2], weights=[0.5, 0.5])]
    out = bl.dcc_portfolio_forecasts(dcc, ports, [800, 801], 5)
    S5 = dcc.covariance_forecast(800, 5)
    assert np.isclose(out[0, 0], 0.25 * (S5[0, 0] + S5[2, 2] + 2 * S5[0, 2]))


def test_dcc_integrated_univariate(dcc_fit):
    _, dcc = dcc_fit
    saved = dcc.uni[0]
    dcc.uni[0] = (0.01, 0.1, 0.0, 0.9)
    try:
        assert dcc._uni_kstep(0, 10, 5) == dcc.nxt[10, 0]
    finally:
        dcc.uni[0] = saved


def test_dcc_nls_target(dcc_fit):
    R, _ = dcc_fit
    d = bl.DCC().fit(R, 700, n_jobs=1, target="nls")
    assert np.allclose(np.diag(d.Qbar), 1) and np.all(np.linalg.eigvalsh(d.Qbar) > 0)


def test_analytical_shrinkage(rng):
    X = rng.standard_normal((300, 20))
    S = bl.analytical_shrinkage(X)
    sample = np.cov(X.T)
    ev, es = np.linalg.eigvalsh(S), np.linalg.eigvalsh(sample)
    assert np.allclose(S, S.T) and ev.min() > 0
    assert ev.max() - ev.min() < es.max() - es.min()  # eigenvalues pulled together
    assert abs(np.trace(S) / np.trace(sample) - 1) < 0.1
    assert bl.analytical_shrinkage(X, demean=False).shape == (20, 20)
    with pytest.raises(ValueError):
        bl.analytical_shrinkage(rng.standard_normal((10, 20)))


def test_analytical_shrinkage_tied_eigenvalues():
    """Equal eigenvalues hit the edge case |x| = sqrt(5) only through rounding; the result must stay finite."""
    X = np.vstack([np.eye(5), -np.eye(5)] * 20)
    assert np.isfinite(bl.analytical_shrinkage(X)).all()


def test_cov_to_corr_and_ledoit_wolf(rng):
    S = np.array([[4.0, 1.0], [1.0, 9.0]])
    assert np.allclose(bl.cov_to_corr_np(S), [[1, 1 / 6], [1 / 6, 1]])
    X = rng.standard_normal((50, 5))
    lw = bl.ledoit_wolf_2004(X)
    assert lw.shape == (5, 5) and np.all(np.linalg.eigvalsh(lw) > 0)


def test_rolling_shrinkage_covs(rng):
    R = rng.standard_normal((300, 4)) * 0.01
    S_ewma = np.broadcast_to(np.diag([1.0, 2.0, 3.0, 4.0])[None] * 1e-4, (300, 4, 4))
    t = np.array([260, 299])
    lw = bl.rolling_shrinkage_covs(R, t, window=250)
    assert np.allclose(lw[0], bl.ledoit_wolf_2004(R[11:261]))
    nls = bl.rolling_shrinkage_covs(R, t, window=250, kind="ewma-nls", S_ewma=S_ewma)
    assert np.allclose(np.diag(nls[1]), np.diag(S_ewma[299]))
    with pytest.raises(ValueError):
        bl.rolling_shrinkage_covs(R, t, kind="other")


def test_quad_forms(rng):
    A = rng.standard_normal((3, 4, 4))
    S = A @ A.transpose(0, 2, 1)
    ports = [dict(ids=[0, 3], weights=[1.0, -1.0]), dict(ids=[1], weights=[2.0])]
    out = bl.quad_forms(S, ports)
    assert np.allclose(out[:, 0], S[:, 0, 0] + S[:, 3, 3] - 2 * S[:, 0, 3])
    assert np.allclose(out[:, 1], 4 * S[:, 1, 1])
