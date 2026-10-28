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
