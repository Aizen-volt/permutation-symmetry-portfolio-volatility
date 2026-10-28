"""Correctness checks: permutation symmetry, target alignment, absence of look-ahead."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvol.data import load_panel  # noqa: E402
from pvol.dataset import PanelGPU, make_split, origins_in_year, year_bounds  # noqa: E402
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


@pytest.mark.parametrize("name", ["PortMLP", "DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov", "MLP-sorted", "MLP-tsort", "DeepSetsP",
                                  "Linear-noasset", "Linear-noeuler", "Linear-nopairs", "Linear-noport", "Linear-portonly"])
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


def test_tie_broken_sort_with_equal_weights(panel):
    p, split = panel
    gen = torch.Generator(device=DEV).manual_seed(1)
    t, S, _ = p.sample(64, 10, split.train, gen)
    w = torch.full(S.shape, 0.1, device=DEV)
    perm = torch.randperm(10, device=DEV)
    out = {}
    for name in ("MLP-sorted", "MLP-tsort"):
        torch.manual_seed(0)
        model = MODELS[name]().to(DEV).eval()
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.05)
        a, b = perturbed_output(model, p, t, S, w, perm, 0)
        out[name] = (a - b).abs().max().item()
    assert out["MLP-tsort"] < 1e-4 and out["MLP-sorted"] > 1e-3, out


def test_linear_full_features_unchanged(panel):
    p, split = panel
    gen = torch.Generator(device=DEV).manual_seed(1)
    t, S, w = p.sample(32, 9, split.train, gen)
    inp, _ = p.batch(t, S, w)
    from pvol.models import invariant_pools, node_inputs
    c = torch.einsum("bklf,bl->bkf", inp["pair"], inp["w"])
    ref = torch.cat([invariant_pools(torch.cat([node_inputs(inp), c], -1), inp), inp["g"]], -1)
    lin = MODELS["Linear"]()
    assert lin.lin.in_features == ref.shape[1] and torch.equal(lin.features(inp), ref)


@pytest.mark.parametrize("h", [1, 5, 22])
def test_split_targets_inside_blocks(panel, h):
    p, _ = panel
    for y in range(2020, 2027):
        split = make_split(p.dates, p.first_valid, y, h)
        v0, v1 = year_bounds(p.dates, y - 1)
        assert split.train[1] + h < v0  # training targets end before the validation year
        assert v0 <= split.val[0] and split.val[1] + h <= v1  # validation targets end inside it
        assert origins_in_year(p.dates, y, h)[0] > v1


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


def test_raw_mode_removes_portfolio_summaries(panel):
    p, split = panel
    gen = torch.Generator(device=DEV).manual_seed(4)
    t, S, w = p.sample(16, 7, split.train, gen)
    inp, _ = p.batch(t, S, w, raw=True)
    assert torch.count_nonzero(inp["g"]) == 0
    assert torch.count_nonzero(inp["x_port"][..., 1:]) == 0
    assert torch.allclose(inp["x_port"][..., 0], w * 7)
    assert torch.unique(inp["base"]).numel() == 1


@pytest.mark.parametrize("name", ["DeepSets", "PENet", "SetTransformer", "DeepSetsP"])
def test_invariance_in_raw_mode(panel, name):
    p, split = panel
    torch.manual_seed(0)
    model = MODELS[name]().to(DEV).eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.05)
    gen = torch.Generator(device=DEV).manual_seed(5)
    t, S, w = p.sample(32, 9, split.train, gen)
    perm = torch.randperm(9, device=DEV)
    with torch.no_grad():
        a = model(p.batch(t, S, w, raw=True)[0])
        b = model(p.batch(t, S[:, perm], w[:, perm], raw=True)[0])
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-5)


def test_portfolio_pool(panel):
    from pvol.dataset import PortfolioPool

    p, split = panel
    pool = PortfolioPool(p, 50, 2, 20, (0.3, 1.0, 3.0), seed=1)
    assert sum(len(S) for S, _ in pool.groups.values()) == 50
    gen = torch.Generator(device=DEV).manual_seed(0)
    N = pool.sizes[0]
    t, S, w = pool.sample(64, N, split.train, gen, DEV)
    rows = {tuple(r) for r in pool.groups[N][0].tolist()}
    assert all(tuple(r) in rows for r in S.tolist())  # only pool portfolios, in their fixed order
    assert torch.allclose(w.sum(1), torch.ones(64, device=DEV))


def test_portfolio_pool_shuffle(panel):
    """Experiment R1: the same pool portfolios, with the assets listed in a new random order in every draw."""
    from pvol.dataset import PortfolioPool

    p, split = panel
    pool = PortfolioPool(p, 50, 2, 20, (0.3, 1.0, 3.0), seed=1, shuffle=True)
    gen = torch.Generator(device=DEV).manual_seed(0)
    N = max(pool.sizes)
    t, S, w = pool.sample(64, N, split.train, gen, DEV)
    base = {tuple(sorted(zip(r, v))) for r, v in zip(pool.groups[N][0].tolist(), pool.groups[N][1].tolist())}
    assert all(tuple(sorted(zip(r, v))) in base for r, v in zip(S.tolist(), w.tolist()))  # same portfolios as sets
    fixed = {tuple(r) for r in pool.groups[N][0].tolist()}
    assert any(tuple(r) not in fixed for r in S.tolist())  # but not in the fixed order
