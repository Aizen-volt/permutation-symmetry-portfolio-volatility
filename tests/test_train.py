"""Training loop, prediction, sampling and model utilities on the real panel, with very short training runs."""
import numpy as np
import pytest
import torch

from pvol import models
from pvol import train as tr
from pvol.data import load_panel
from pvol.dataset import PanelGPU, PortfolioPool, make_split, origins_in_year, sample_weights, year_bounds
from pvol.portfolios import make_eval_portfolios, random_weights

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TINY = dict(batch=16, steps_per_epoch=2, max_epochs=3, patience=10, val_batches=2)


@pytest.fixture(scope="module")
def panel():
    p = PanelGPU(load_panel(), device=DEV)
    return p, make_split(p.dates, p.first_valid, 2021, 5)


def test_qlike_torch_matches_numpy():
    from pvol.evaluate import qlike as qlike_np

    y = torch.tensor([0.5, 1.0, 3.0], dtype=torch.float64)
    s = torch.log(torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64))
    assert np.allclose(tr.qlike(s, y).numpy(), qlike_np(y.numpy(), np.ones(3)))
    assert torch.allclose(tr.qlike(torch.log(y), y), torch.zeros(3, dtype=torch.float64))


def test_build_model_uses_hp():
    m = tr.build_model("DeepSets", dict(d=16, phi=(8,), rho=(8,)))
    assert isinstance(m, models.DeepSets) and sum(p.numel() for p in m.parameters()) < 2000
    assert isinstance(tr.build_model("Linear"), models.LinearSet)


def test_fixed_batches_cycle_sizes_and_alphas(panel):
    p, split = panel
    cfg = tr.TrainCfg(batch=8, n_min=2, n_max=4)
    b = tr.fixed_batches(p, split.val, 6, cfg, seed=1)
    assert [N for N, _ in b] == [2, 3, 4, 2, 3, 4]
    assert all(S.shape == (8, N) for N, (_, S, _) in b)
    again = tr.fixed_batches(p, split.val, 6, cfg, seed=1)
    assert all(torch.equal(a[1][1], c[1][1]) for a, c in zip(b, again))  # fixed validation set
    pool = PortfolioPool(p, 20, 2, 4, (1.0,), seed=0)
    bp = tr.fixed_batches(p, split.val, 4, cfg, seed=1, pool=pool)
    assert [N for N, _ in bp] == [pool.sizes[k % len(pool.sizes)] for k in range(4)]


def test_evaluate_batches_is_mean_loss(panel):
    p, split = panel
    p.fit_scalers(split.train, torch.Generator(device=DEV).manual_seed(0))
    cfg = tr.TrainCfg(batch=8)
    b = tr.fixed_batches(p, split.val, 3, cfg, seed=2)
    m = models.MODELS["Linear"]().to(DEV)
    v = tr.evaluate_batches(m, p, b, 5, 0)
    ref = np.mean([tr.qlike(m(p.batch(t, S, w, h=5)[0]), p.batch(t, S, w, h=5)[1]).mean().item()
                   for _, (t, S, w) in b])
    assert np.isclose(v, ref, rtol=1e-5)


@pytest.mark.parametrize("name", ["Linear", "DeepSets", "SetGRU", "MLP-sorted"])
def test_train_model_runs(panel, name):
    p, split = panel
    hp = dict(d=16, phi=(16,), rho=(16,)) if name == "DeepSets" else None
    model, info = tr.train_model(name, p, split, 5, seed=0, cfg=tr.TrainCfg(**TINY), hp=hp)
    assert info["epochs"] == 3 and len(info["history"]) == 3 and np.isfinite(info["best_val"])
    assert info["n_params"] == sum(q.numel() for q in model.parameters()) and model.raw_inputs is False


def test_train_model_is_deterministic(panel):
    p, split = panel
    cfg = tr.TrainCfg(**TINY)
    a, ia = tr.train_model("Linear", p, split, 5, seed=3, cfg=cfg)
    b, ib = tr.train_model("Linear", p, split, 5, seed=3, cfg=cfg)
    assert ia["best_val"] == ib["best_val"]
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_train_model_early_stopping(panel):
    p, split = panel
    cfg = tr.TrainCfg(**{**TINY, "max_epochs": 20, "patience": 1, "lr": 0.0})
    _, info = tr.train_model("Linear", p, split, 5, seed=0, cfg=cfg)
    assert info["epochs"] == 2  # no improvement with a zero learning rate
    cfg = tr.TrainCfg(**{**TINY, "max_epochs": 20, "patience": 1, "lr": 0.0, "min_epochs": 4})
    assert tr.train_model("Linear", p, split, 5, seed=0, cfg=cfg)[1]["epochs"] == 4


def test_train_model_protocol_options(panel):
    p, split = panel
    _, info = tr.train_model("Linear", p, split, 5, 0, cfg=tr.TrainCfg(**TINY, top_k_avg=2))
    assert np.isfinite(info["best_val"])
    model, info = tr.train_model("Linear", p, split, 5, 0, cfg=tr.TrainCfg(**TINY, refit=True, refit_min_epochs=2))
    assert info["refit_epochs"] >= 2
    model, info = tr.train_model("DeepSets", p, split, 5, 0, cfg=tr.TrainCfg(**TINY, raw_inputs=True),
                                 hp=dict(d=16, phi=(16,), rho=(16,)))
    assert model.raw_inputs is True
    _, info = tr.train_model("Linear", p, split, 5, 0, cfg=tr.TrainCfg(**TINY, portfolio_pool=30, pool_shuffle=True))
    assert info["cfg"]["portfolio_pool"] == 30


def test_train_model_rejects_bad_configs(panel):
    p, split = panel
    with pytest.raises(ValueError):
        tr.train_model("Linear", p, split, 5, 0, cfg=tr.TrainCfg(**TINY, refit=True, portfolio_pool=10))
    with pytest.raises(ValueError):
        tr.train_model("Linear", p, split, 5, 0, cfg=tr.TrainCfg(**TINY, gru_len=10))


def test_train_model_reports_divergence(panel, monkeypatch):
    p, split = panel
    monkeypatch.setattr(tr, "evaluate_batches", lambda *a, **k: float("nan"))
    with pytest.raises(RuntimeError, match="diverged"):
        tr.train_model("Linear", p, split, 5, 0, cfg=tr.TrainCfg(**TINY))


def test_train_model_verbose(panel, capsys):
    p, split = panel
    tr.train_model("Linear", p, split, 5, 0, cfg=tr.TrainCfg(**{**TINY, "max_epochs": 1}), verbose=True)
    assert "ep   0" in capsys.readouterr().out


def test_predict_portfolios(panel):
    p, split = panel
    model, _ = tr.train_model("MLP", p, split, 5, 0, cfg=tr.TrainCfg(**{**TINY, "max_epochs": 1}),
                              hp=dict(hidden=(16,)))
    ports = make_eval_portfolios(p.tickers, n_per_group=2)
    t = origins_in_year(p.dates, 2021, 5)[:7]
    out = tr.predict_portfolios(model, "MLP", p, ports, t)
    n = np.array([len(q["ids"]) for q in ports])
    assert out.shape == (7, len(ports))
    assert np.isnan(out[:, n > 20]).all() and np.isfinite(out[:, n <= 20]).all()  # slots limit the MLP
    # one portfolio by hand, the larger chunked path (N > 10) included
    j = int(np.where(n == 20)[0][0])
    inp, _ = p.batch(torch.as_tensor(t, device=DEV), torch.as_tensor([ports[j]["ids"]] * 7, device=DEV),
                     torch.as_tensor([ports[j]["weights"]] * 7, device=DEV, dtype=torch.float32))
    with torch.no_grad():
        assert np.allclose(out[:, j], model(inp).cpu().numpy(), atol=1e-5)
    small = tr.predict_portfolios(model, "MLP", p, ports, t, chunk=3)
    assert np.allclose(small, out, atol=1e-5, equal_nan=True)


def test_predict_portfolios_tta(panel):
    p, split = panel
    model, _ = tr.train_model("MLP", p, split, 5, 0, cfg=tr.TrainCfg(**{**TINY, "max_epochs": 1}),
                              hp=dict(hidden=(16,)))
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.05)
    ports = make_eval_portfolios(p.tickers, n_per_group=2, groups={"N5": (5, "long_only", 1.0)}, structured=False)
    t = origins_in_year(p.dates, 2021, 5)[:4]
    plain = tr.predict_portfolios(model, "MLP", p, ports, t)
    avg = tr.predict_portfolios(model, "MLP", p, ports, t, tta_k=4)
    assert not np.allclose(plain, avg)
    # averaging over orderings makes the forecast close to invariant to the input order
    flipped = [dict(q, ids=q["ids"][::-1], weights=q["weights"][::-1]) for q in ports]
    a1 = tr.predict_portfolios(model, "MLP", p, flipped, t, tta_k=64)
    a2 = tr.predict_portfolios(model, "MLP", p, ports, t, tta_k=64)
    assert np.abs(a1 - a2).max() < np.abs(plain - tr.predict_portfolios(model, "MLP", p, flipped, t)).max()


def test_realized_targets_by_hand(panel):
    p, _ = panel
    ports = [dict(group="x", ids=[0, 3], weights=[0.5, 0.5])]
    t = np.array([1000, 1001])
    y = tr.realized_targets(p, ports, t, 3)
    R = p.R.cpu().numpy().astype(np.float64)
    rp = 0.5 * R[:, 0] + 0.5 * R[:, 3]
    assert np.allclose(y[:, 0], [np.mean(rp[1001:1004] ** 2), np.mean(rp[1002:1005] ** 2)])


def test_sample_weights():
    gen = torch.Generator().manual_seed(0)
    assert torch.allclose(sample_weights(4, 5, gen, "cpu", alpha=None), torch.full((4, 5), 0.2))
    assert torch.allclose(sample_weights(4, 5, gen, "cpu", alpha=np.inf), torch.full((4, 5), 0.2))
    w = sample_weights(1000, 5, gen, "cpu", alpha=1.0)
    assert torch.allclose(w.sum(1), torch.ones(1000)) and (w > 0).all()
    ls = sample_weights(100, 5, gen, "cpu", scheme="long_short")
    assert torch.allclose(ls.abs().sum(1), torch.ones(100)) and (ls < 0).any()
    with pytest.raises(ValueError):
        sample_weights(2, 2, gen, "cpu", scheme="other")


def test_make_split_train_years(panel):
    p, _ = panel
    full = make_split(p.dates, p.first_valid, 2022, 5)
    short = make_split(p.dates, p.first_valid, 2022, 5, train_years=2)
    assert short.train[0] == year_bounds(p.dates, 2019)[0] and short.train[1] == full.train[1]
    assert make_split(p.dates, p.first_valid, 2022, 5, train_years=50).train == full.train


def test_random_weights_and_eval_portfolios():
    g = np.random.default_rng(0)
    assert np.allclose(random_weights(g, 4, "long_only", np.inf), 0.25)
    assert np.isclose(random_weights(g, 4, "long_only", 1.0).sum(), 1)
    ls = random_weights(g, 6, "long_short", None)
    assert np.isclose(np.abs(ls).sum(), 1)
    ports = make_eval_portfolios(["A", "B", "C"], n_per_group=2)
    assert {q["group"] for q in ports} == {"EW-all"}  # every random group needs more assets
    held = make_eval_portfolios([f"T{i}" for i in range(20)], pool_ids=[1, 5, 7, 9, 11], n_per_group=3,
                                structured=False)
    assert {q["group"] for q in held} == {"N5"} and all(set(q["ids"]) == {1, 5, 7, 9, 11} for q in held)


def test_save_and_load_portfolios(tmp_path):
    from pvol.portfolios import load, save

    ports = make_eval_portfolios([f"T{i}" for i in range(12)], n_per_group=1)
    save(ports, tmp_path / "x" / "p.json")
    assert load(tmp_path / "x" / "p.json") == ports


def test_pad_slots_rejects_large_portfolios(panel):
    p, split = panel
    p.fit_scalers(split.train, torch.Generator(device=DEV).manual_seed(0))
    t, S, w = p.sample(4, 6, split.train, torch.Generator(device=DEV).manual_seed(0))
    inp, _ = p.batch(t, S, w)
    with pytest.raises(ValueError, match="slots"):
        models.pad_slots(inp, 5)
