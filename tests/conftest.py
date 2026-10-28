import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_script(name):
    """Import a numbered script from scripts/ as a module (its __main__ block does not run)."""
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "scripts" / name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


@pytest.fixture(scope="session", autouse=True)
def panel_data(tmp_path_factory):
    """Use the real panel when it exists; otherwise build one from simulated prices over the sample period."""
    from pvol import data
    from pvol.config import SAMPLE_END, SAMPLE_START

    if (data.PROCESSED / "panel.npz").exists() and (data.RAW / "ohlcv_adjusted.parquet").exists():
        yield
        return
    import pandas as pd

    tmp = tmp_path_factory.mktemp("panel")
    idx = pd.bdate_range(SAMPLE_START, SAMPLE_END)
    tickers = list(data.UNIVERSE)
    g = np.random.default_rng(0)
    T, M = len(idx), len(tickers)
    vol = 0.01 * np.exp(np.cumsum(g.normal(0, 0.05, (T, M)), 0) * 0.1)
    C = 100 * np.exp(np.cumsum(vol * g.standard_normal((T, M)), 0))
    O = C * np.exp(g.normal(0, 0.003, (T, M)))
    H = np.maximum(O, C) * np.exp(np.abs(g.normal(0, 0.004, (T, M))))
    L = np.minimum(O, C) * np.exp(-np.abs(g.normal(0, 0.004, (T, M))))
    cols = pd.MultiIndex.from_product([("Open", "High", "Low", "Close"), tickers])
    df = pd.DataFrame(np.concatenate([O, H, L, C], 1), index=idx, columns=cols)
    raw = tmp / "raw"
    raw.mkdir()
    df.to_parquet(raw / "ohlcv_adjusted.parquet")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(data, "RAW", raw)
        mp.setattr(data, "PROCESSED", tmp / "processed")
        data.build_panel()
        yield


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def simulate_garch(n, omega, alpha, beta, gamma=0.0, seed=0):
    """Returns (in decimal units) of a GJR-GARCH(1,1) with parameters on the percent scale used by arch."""
    g = np.random.default_rng(seed)
    x = np.empty(n)
    s2 = omega / (1 - alpha - gamma / 2 - beta)
    for t in range(n):
        x[t] = np.sqrt(s2) * g.standard_normal()
        s2 = omega + (alpha + gamma * (x[t] < 0)) * x[t] ** 2 + beta * s2
    return x / 100.0
