"""Correctness checks: permutation symmetry, target alignment, absence of look-ahead."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvol.data import load_panel  # noqa: E402
from pvol.dataset import PanelGPU, make_split, origins_in_year  # noqa: E402
from pvol.models import MODELS  # noqa: E402
from pvol.portfolios import make_eval_portfolios  # noqa: E402
from pvol.train import realized_targets  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def panel():
    p = PanelGPU(load_panel(), device=DEV)
    split = make_split(p.dates, p.first_valid, 2020, 5)
    p.fit_scalers(split.train, torch.Generator(device=DEV).manual_seed(0))
    return p, split


def perturbed_output(model, panel, t, S, w, perm, gru_len):
    inp, _ = panel.batch(t, S, w, gru_len=gru_len)
    inp_p, _ = panel.batch(t, S[:, perm], w[:, perm], gru_len=gru_len)
    with torch.no_grad():
        return model(inp).double(), model(inp_p).double()


@pytest.mark.parametrize("name", ["PortMLP", "DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov", "MLP-sorted"])
def test_invariant_models(panel, name):
    p, split = panel
    torch.manual_seed(0)
    model = MODELS[name]().to(DEV).eval()
    for m in model.modules():  # zero-initialised heads would make the test trivial
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.05)
    gen = torch.Generator(device=DEV).manual_seed(1)
    t, S, w = p.sample(64, 12, split.train, gen)
    perm = torch.randperm(12, device=DEV)
    a, b = perturbed_output(model, p, t, S, w, perm, 22 if name == "SetGRU" else 0)
    assert torch.isfinite(a).all() and a.std() > 1e-3
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-5), (a - b).abs().max()


@pytest.mark.parametrize("name", ["MLP", "GRU"])
def test_noninvariant_models_change(panel, name):
    p, split = panel
    torch.manual_seed(0)
    model = MODELS[name]().to(DEV).eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.05)
    gen = torch.Generator(device=DEV).manual_seed(1)
    t, S, w = p.sample(64, 12, split.train, gen)
    perm = torch.randperm(12, device=DEV)
    a, b = perturbed_output(model, p, t, S, w, perm, 22 if name == "GRU" else 0)
    assert (a - b).abs().max() > 1e-3


def test_target_alignment(panel):
    p, _ = panel
    ports = make_eval_portfolios(p.tickers, n_per_group=3)
    t_idx = origins_in_year(p.dates, 2021, 5)
    Y = realized_targets(p, ports, t_idx, 5)
    j = 7
    t = torch.as_tensor(t_idx, device=DEV)
    S = torch.as_tensor([ports[j]["ids"]] * len(t_idx), device=DEV)
    w = torch.as_tensor([ports[j]["weights"]] * len(t_idx), device=DEV, dtype=torch.float32)
    _, y = p.batch(t, S, w, h=5)
    np.testing.assert_allclose(y.cpu().numpy(), Y[:, j], rtol=1e-4)


def test_no_lookahead(panel):
    """Inputs at origin t must not change when returns after t are overwritten."""
    p, split = panel
    gen = torch.Generator(device=DEV).manual_seed(2)
    t, S, w = p.sample(32, 8, split.train, gen)
    t[:] = t[0]
    inp, _ = p.batch(t, S, w, gru_len=22)
    raw = load_panel()
    t0 = int(t[0])
    raw["R"][t0 + 1 :] = 0.123
    raw["X"][t0 + 1 :] = 9.9
    raw["P"][t0 + 1 :] = 0.5
    raw["S_ewma"][t0 + 1 :] = 7.0
    q = PanelGPU(raw, device=DEV)
    q.x_mu, q.x_sd, q.g_mu, q.g_sd = p.x_mu, p.x_sd, p.g_mu, p.g_sd
    inp2, _ = q.batch(t, S, w, gru_len=22)
    for k in inp:
        assert torch.equal(inp[k], inp2[k]), k


def test_panel_features_causal():
    """Rebuilding features from prices truncated at day t reproduces the panel at day t."""
    import pandas as pd
    from pvol.data import load_prices, node_features, variance_proxies

    fields, _ = load_prices()
    full = load_panel()
    t = 3000
    cut = {k: v.iloc[: t + 1] for k, v in fields.items()}
    R, sq, rng = variance_proxies(cut["Open"], cut["High"], cut["Low"], cut["Close"])
    X, _ = node_features(R, sq, rng)
    np.testing.assert_allclose(X[t], full["X"][t], rtol=1e-4, atol=1e-5)
