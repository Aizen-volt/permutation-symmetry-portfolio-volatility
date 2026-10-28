"""Stronger classical benchmarks of the fourth series: HAR-WLS, LogHAR-X, HEAVY and the PCA factor model."""
import numpy as np

from conftest import simulate_garch
from pvol import baselines as bl
from pvol import baselines_ext as bx


def test_target_is_forward_mean(rng):
    rp = rng.standard_normal(40)
    assert np.allclose(bx._target(rp, np.array([10]), 5), np.mean(rp[11:16] ** 2))


def test_har_wls_close_to_ols_and_positive():
    rp = simulate_garch(2000, 0.05, 0.08, 0.9)
    fit, t = np.arange(30, 1500), np.arange(1500, 1900)
    f = bx.har_wls_forecast(rp, fit, t, 5)
    ols, _ = bl.har_forecast(rp, fit, t, 5)
    assert np.all(f > 0) and np.corrcoef(f, ols)[0, 1] > 0.9
    assert np.allclose(bx.har_wls_forecast(rp, fit, t, 5, iters=0), ols)


def test_loghar_x():
    T, M = 800, 3
    rp_assets = np.column_stack([simulate_garch(T, 0.05, 0.08, 0.9, seed=s) for s in range(M)])
    X = np.log(np.maximum(rp_assets[:, :, None] ** 2, 1e-8)).repeat(3, axis=2)
    P = np.broadcast_to(np.eye(M)[None, :, :, None], (T, M, M, 3)).copy()
    p = dict(ids=[0, 1, 2], weights=[1 / 3] * 3)
    rp = bl.portfolio_returns(rp_assets, p)
    f = bx.loghar_x_forecast(rp, X, P, p, np.arange(30, 600), np.arange(600, 790), 5, [0, 1, 2])
    assert f.shape == (190,) and np.all(f > 0)
    y = bx._target(rp, np.arange(600, 790), 5)
    assert 0.2 < np.mean(f) / np.mean(y) < 5


def test_heavy_path_matches_recursion(rng):
    x = rng.uniform(0.5, 2, 30)
    v = bx._heavy_path(x, 0.1, 0.3, 0.6, 1.5)
    ref = [1.5]
    for t in range(1, 30):
        ref.append(0.1 + 0.3 * x[t - 1] + 0.6 * ref[-1])
    assert np.allclose(v, ref)


def test_fit_heavy_recovers_parameters():
    g = np.random.default_rng(1)
    n = 4000
    rk = np.exp(g.normal(0, 0.5, n))
    v = bx._heavy_path(rk, 0.2, 0.4, 0.5, 1.0)
    target = v * g.chisquare(1, n)
    om, a, b = bx._fit_heavy(target, rk, 1.0)
    assert abs(a - 0.4) < 0.1 and abs(b - 0.5) < 0.15 and om > 0


def test_heavy_forecasts():
    g = np.random.default_rng(2)
    rp = simulate_garch(1500, 0.05, 0.08, 0.9, seed=4)
    rk = rp**2 * np.exp(g.normal(0, 0.3, 1500)) + 1e-8
    t = np.arange(1200, 1300)
    out = bx.heavy_forecasts(rp, rk, 22, 1199, t, (1, 5, 22))
    assert set(out) == {1, 5, 22}
    assert all(np.all(out[h] > 0) for h in out)
    # multi-step forecasts move towards the unconditional level, so they vary less across origins
    assert out[22].std() < out[1].std()


def test_pca_factor_covs(rng):
    R = rng.standard_normal((300, 8)) @ rng.standard_normal((8, 8)) * 0.01
    S = np.broadcast_to(np.diag(np.arange(1.0, 9.0))[None] * 1e-4, (300, 8, 8))
    out = bx.pca_factor_covs(R, S, np.array([260, 299]), k=3)
    assert out.shape == (2, 8, 8)
    assert np.allclose(np.diag(out[0]), np.diag(S[260]))
    assert np.allclose(out[1], out[1].T)
    # with all components the correlation part equals the sample correlation
    full = bx.pca_factor_covs(R, S, np.array([299]), k=8)[0]
    d = np.sqrt(np.diag(S[299]))
    assert np.allclose(full, np.corrcoef(R[50:300].T) * np.outer(d, d))
