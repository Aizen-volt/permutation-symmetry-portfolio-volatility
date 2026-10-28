"""Experiment runner on a reduced panel of twelve assets: targets, networks, benchmarks."""
import json

import numpy as np
import pytest
import torch

from pvol import experiment as ex
from pvol.data import load_panel
from pvol.dataset import origins_in_year
from pvol.train import TrainCfg

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ASSETS = [0, 1, 2, 5, 6, 14, 15, 18, 25, 30, 40, 50]  # fit_scalers draws ten-asset portfolios


@pytest.fixture(scope="module")
def small_panel():
    full = load_panel()
    a = np.array(ASSETS)
    out = dict(full)
    for k in ("R", "X"):
        out[k] = full[k][:, a]
    out["S_ewma"] = full["S_ewma"][:, a][:, :, a]
    out["P"] = full["P"][:, a][:, :, a]
    out["tickers"], out["asset_class"], out["class_onehot"] = full["tickers"][a], full["asset_class"][a], \
        full["class_onehot"][a]
    return out


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "RESULTS", tmp_path)
    return tmp_path


def test_heldout_assets_stratified(small_panel):
    held = ex.heldout_assets(small_panel)
    classes = small_panel["asset_class"]
    assert held == sorted(held) and set(classes[held]) == set(classes)  # at least one asset of every class
    assert held == ex.heldout_assets(small_panel)


def test_experiment_main(small_panel, results):
    e = ex.Experiment("main", small_panel, device=DEV)
    assert (results / "main" / "portfolios.json").exists() and e.train_ids is None
    assert {p["group"] for p in e.portfolios} == {"N5", "N10", "N10-conc", "N10-EW", "N10-LS", "EW-all"}
    perm = json.loads((results / "main" / "perm_portfolios.json").read_text())
    assert perm == ex.Experiment("main", small_panel, device=DEV).perm_portfolios  # reused, not redrawn
    assert e.origins(5, years=(2021,)).tolist() == origins_in_year(e.panel.dates, 2021, 5).tolist()
    t = e.write_targets(1)
    y = np.load(results / "main" / "h1" / "targets.npy")
    assert y.shape == (len(t), len(e.portfolios)) and (y >= 0).all()


def test_experiment_heldout(small_panel, results):
    e = ex.Experiment("heldout", small_panel, device=DEV)
    held = json.loads((results / "heldout" / "heldout_assets.json").read_text())
    tick = list(small_panel["tickers"])
    held_ids = {tick.index(t) for t in held}
    assert set(e.train_ids) == set(range(len(tick))) - held_ids
    assert all(set(p["ids"]) <= held_ids for p in e.portfolios)
    assert set(e.panel.asset_ids.tolist()) == set(e.train_ids)


@pytest.mark.usefixtures("results")
def test_experiment_rejects_unknown(small_panel):
    with pytest.raises(ValueError):
        ex.Experiment("other", small_panel, device=DEV)


def test_run_nn_saves_and_skips(small_panel, results, capsys):
    e = ex.Experiment("main", small_panel, device=DEV)
    cfg = TrainCfg(batch=16, steps_per_epoch=2, max_epochs=1, val_batches=2, n_max=8)
    e.run_nn("Linear", 5, seeds=[0], cfg=cfg, years=(2021,), tag="Lin")
    d = results / "main" / "h5"
    pred = np.load(d / "preds" / "Lin__s0.npy")
    assert pred.shape == (len(origins_in_year(e.panel.dates, 2021, 5)), len(e.portfolios))
    assert np.load(d / "preds" / "Lin__s0__perm.npy").shape[1] == len(e.perm_portfolios)
    info = json.loads((d / "info" / "Lin__s0__y2021.json").read_text())
    assert info["tag"] == "Lin" and info["year"] == 2021 and "Lin seed=0 year=2021" in capsys.readouterr().out
    e.run_nn("Linear", 5, seeds=[0], cfg=cfg, years=(2021,), tag="Lin")  # finished: skipped
    assert capsys.readouterr().out == ""


def test_run_baselines(small_panel, results):
    e = ex.Experiment("main", small_panel, device=DEV)
    e.portfolios = e.portfolios[:2] + e.portfolios[-1:]
    e.run_baselines(5, years=(2021,), which=("HIST", "EWMA", "HAR", "HARX", "GARCH", "GJR", "DCC"))
    d = results / "main" / "h5"
    n = len(origins_in_year(e.panel.dates, 2021, 5))
    for k in ("HIST", "EWMA", "HAR", "HARX", "GARCH", "GJR", "DCC"):
        f = np.load(d / "preds" / f"{k}.npy")
        assert f.shape == (n, 3) and np.isfinite(f).all(), k
    dcc = json.loads((d / "info" / "DCC__y2021.json").read_text())
    assert dcc["a"] + dcc["b"] < 1
    # the EWMA forecast is w' Sigma_t w of the panel's EWMA covariance
    p, t = e.portfolios[0], origins_in_year(e.panel.dates, 2021, 5)
    w = np.array(p["weights"])
    S = small_panel["S_ewma"][t][:, p["ids"]][:, :, p["ids"]].astype(np.float64)
    assert np.allclose(np.exp(np.load(d / "preds" / "EWMA.npy")[:, 0]), np.einsum("i,tij,j->t", w, S, w), rtol=1e-5)


def test_load_tuned_hp(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "TUNED_HP", tmp_path / "best_hp.json")
    assert ex.load_tuned_hp() == {}
    (tmp_path / "best_hp.json").write_text(json.dumps({"MLP": {"dropout": 0.3}}))
    assert ex.load_tuned_hp() == {"MLP": {"dropout": 0.3}}
