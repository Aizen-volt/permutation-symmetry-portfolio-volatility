"""Experiment runners: expanding-window refits (one per test year), predictions saved per model/seed."""
import json
import time

import numpy as np
import torch

from . import baselines as bl
from .config import RESULTS, TEST_YEARS
from .dataset import SEQ_MARGIN, PanelGPU, make_split, origins_in_year, year_bounds
from .portfolios import load, make_eval_portfolios, save
from .train import TrainCfg, predict_portfolios, realized_targets, train_model

HELDOUT_SEED = 2026
TUNED_HP = RESULTS / "tuning" / "best_hp.json"


def heldout_assets(panel_np, frac=0.22, seed=HELDOUT_SEED):
    """Stratified (by asset class) random subset of assets that is never used during training in exp 'heldout'."""
    rng = np.random.default_rng(seed)
    out = []
    for c in sorted(set(panel_np["asset_class"])):
        ids = np.where(panel_np["asset_class"] == c)[0]
        k = max(1, int(round(frac * len(ids))))
        out += rng.choice(ids, size=k, replace=False).tolist()
    return sorted(out)


class Experiment:
    """exp='main': all 54 assets. exp='heldout': train on 42 assets, evaluate on portfolios of unseen assets."""

    def __init__(self, name, panel_np, device="cuda"):
        self.name = name
        self.np = panel_np
        self.dir = RESULTS / name
        self.dir.mkdir(parents=True, exist_ok=True)
        tickers = panel_np["tickers"]
        if name == "main":
            self.train_ids = None
            ports = make_eval_portfolios(tickers)
        elif name == "heldout":
            held = heldout_assets(panel_np)
            self.train_ids = [i for i in range(len(tickers)) if i not in held]
            groups = {"H5": (5, "long_only", 1.0), "H10": (10, "long_only", 1.0), "H10-EW": (10, "long_only", np.inf)}
            ports = make_eval_portfolios(tickers, pool_ids=held, groups=groups, structured=False)
            (self.dir / "heldout_assets.json").write_text(json.dumps([str(tickers[i]) for i in held]))
        else:
            raise ValueError(name)
        path = self.dir / "portfolios.json"
        if not path.exists():
            save(ports, path)
        self.portfolios = load(path)
        self.perm_portfolios = self._perm_set()
        self.panel = PanelGPU(panel_np, device=device, asset_ids=self.train_ids)

    def _perm_set(self, group="N10", n_ports=100, n_perm=5, seed=777):
        """The same portfolios with the assets listed in n_perm different orders (for permutation-consistency)."""
        path = self.dir / "perm_portfolios.json"
        if path.exists():
            return load(path)
        rng = np.random.default_rng(seed)
        base = [(j, p) for j, p in enumerate(self.portfolios) if p["group"] in (group, "H10")][:n_ports]
        out = []
        for k in range(n_perm):
            for j, p in base:
                o = rng.permutation(len(p["ids"]))
                out.append(dict(group=p["group"], src=j, perm=k, ids=[p["ids"][i] for i in o],
                                weights=[p["weights"][i] for i in o]))
        save(out, path)
        return out

    def hdir(self, h):
        d = self.dir / f"h{h}"
        (d / "preds").mkdir(parents=True, exist_ok=True)
        (d / "info").mkdir(parents=True, exist_ok=True)
        return d

    def origins(self, h, years=TEST_YEARS):
        return np.concatenate([origins_in_year(self.panel.dates, y, h) for y in years])

    def write_targets(self, h):
        d = self.hdir(h)
        t_idx = self.origins(h)
        np.save(d / "origins.npy", t_idx)
        np.save(d / "targets.npy", realized_targets(self.panel, self.portfolios, t_idx, h))
        return t_idx

    # neural networks
    def run_nn(self, name, h, seeds, hp=None, cfg=None, years=TEST_YEARS, tag=None, skip_existing=True, train_years=None,
               tta_k=0):
        d = self.hdir(h)
        tag = tag or name
        for seed in seeds:
            out = d / "preds" / f"{tag}__s{seed}.npy"
            if skip_existing and out.exists():
                continue
            parts, perm_parts = [], []
            for y in years:
                split = make_split(self.panel.dates, self.panel.first_valid, y, h, train_years=train_years)
                model, info = train_model(name, self.panel, split, h, seed, cfg=cfg, hp=hp)
                t_idx = origins_in_year(self.panel.dates, y, h)
                parts.append(predict_portfolios(model, name, self.panel, self.portfolios, t_idx, tta_k=tta_k))
                perm_parts.append(predict_portfolios(model, name, self.panel, self.perm_portfolios, t_idx, tta_k=tta_k))
                info.update(year=y, tag=tag)
                (d / "info" / f"{tag}__s{seed}__y{y}.json").write_text(json.dumps(info))
                print(f"[{self.name} h={h}] {tag} seed={seed} year={y}: val {info['best_val']:.4f} "
                      f"epochs {info['epochs']} {info['seconds']:.0f}s", flush=True)
                del model
                torch.cuda.empty_cache()
            np.save(out, np.concatenate(parts).astype(np.float32))
            np.save(d / "preds" / f"{tag}__s{seed}__perm.npy", np.concatenate(perm_parts).astype(np.float32))

    # classical benchmarks
    def run_baselines(self, h, years=TEST_YEARS, which=("HIST", "EWMA", "HAR", "HARX", "GARCH", "GJR", "DCC")):
        d = self.hdir(h)
        R = self.np["R"].astype(np.float64)
        S = self.np["S_ewma"]
        dates = self.panel.dates
        fv = self.panel.first_valid
        preds = {k: [] for k in which}
        for y in years:
            t0 = time.time()
            t_idx = origins_in_year(dates, y, h)
            fit_end = year_bounds(dates, y - 1)[1]
            fit_idx = np.arange(fv + SEQ_MARGIN, fit_end - h + 1)
            rps = [bl.portfolio_returns(R, p) for p in self.portfolios]
            if "HIST" in which:
                preds["HIST"].append(np.column_stack([bl.hist_forecast(rp, t_idx) for rp in rps]))
            if "EWMA" in which:
                preds["EWMA"].append(np.column_stack([bl.ewma_forecast(S, p, t_idx) for p in self.portfolios]))
            if "HAR" in which:
                preds["HAR"].append(np.column_stack([bl.har_forecast(rp, fit_idx, t_idx, h)[0] for rp in rps]))
            if "HARX" in which:
                names = list(self.np["feat_names"])
                fidx = [names.index(f"log_rng_{k}") for k in (1, 5, 22)]
                preds["HARX"].append(np.column_stack([
                    bl.harx_forecast(rp, self.np["X"], self.np["P"], p, fit_idx, t_idx, h, fidx)[0]
                    for rp, p in zip(rps, self.portfolios)]))
            if "GARCH" in which:
                preds["GARCH"].append(bl.run_garch_many(R, self.portfolios, fit_end, t_idx, h))
            if "GJR" in which:
                preds["GJR"].append(bl.run_garch_many(R, self.portfolios, fit_end, t_idx, h, o=1))
            if "DCC" in which:
                dcc = bl.DCC().fit(R, fit_end)
                preds["DCC"].append(bl.dcc_portfolio_forecasts(dcc, self.portfolios, t_idx, h))
                (d / "info" / f"DCC__y{y}.json").write_text(json.dumps(dict(a=dcc.a, b=dcc.b)))
            print(f"[{self.name} h={h}] baselines year={y} {time.time() - t0:.0f}s", flush=True)
        for k, v in preds.items():
            np.save(d / "preds" / f"{k}.npy", np.log(np.maximum(np.concatenate(v), 1e-12)).astype(np.float32))


def load_tuned_hp():
    return json.loads(TUNED_HP.read_text()) if TUNED_HP.exists() else {}
