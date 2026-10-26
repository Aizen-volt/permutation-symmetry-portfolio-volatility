"""Fifth series T: broader hyper-parameter search with validation over several years (decision rules fixed before running).

Random search: the current configuration (#0) plus 20 random draws per architecture (search seed 2610) over the learning
rate, weight decay, patience, minimum number of epochs, width/depth and dropout. Criterion: mean validation QLIKE over three
expanding-window folds (validation 2017, 2018, 2019; training up to the end of the previous year), h = 5, seed 0. All folds
precede the first test year (2020). Writes results*/tuning/tuning_log_T.jsonl and best_hp_T.json
({model: {"hp": ..., "cfg": ...}}); best_hp.json of the main protocol is not changed.

Usage: [PVOL_UNIVERSE=stocks] python scripts/36_tune_wide.py      (resumable: finished (model, config) pairs are skipped)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pvol.config import RESULTS  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.dataset import PanelGPU, make_split  # noqa: E402
from pvol.experiment import load_tuned_hp  # noqa: E402
from pvol.train import TrainCfg, train_model  # noqa: E402

H = 5
FOLDS = (2018, 2019, 2020)  # "test years" whose validation years 2017-2019 are the folds; no test origin is used
N_RANDOM = 20
SEARCH_SEED = 2610
MODELS = ["MLP", "GRU", "DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov", "Linear"]
CFG_SPACE = dict(lr=[3e-5, 1e-4, 3e-4, 1e-3], weight_decay=[0.0, 1e-5, 1e-4, 1e-3], patience=[10, 20], min_epochs=[0, 10])
RHO = [(64,), (128,), (128, 128)]
ARCH_SPACE = {
    "MLP": dict(hidden=[(256,), (256, 256), (512, 256), (1024, 512), (512, 256, 128)], dropout=[0.0, 0.1, 0.3, 0.5]),
    "GRU": dict(d_hidden=[32, 64, 128, 256], head=[(128,), (256,), (256, 256)], dropout=[0.0, 0.1, 0.3]),
    "DeepSets": dict(d=[32, 64, 128, 256], phi=["d", "dd"], rho=["d", "dd"], dropout=[0.0, 0.1, 0.3]),
    "PENet": dict(d=[32, 64, 128], n_layers=[1, 2, 3], rho=RHO, dropout=[0.0, 0.1, 0.3]),
    "SetTransformer": dict(d=[32, 64, 128], n_layers=[1, 2, 3], rho=RHO, dropout=[0.0, 0.1, 0.3]),
    "SetGRU": dict(d=[32, 64, 128], d_gru=[32, 64], n_layers=[1, 2], rho=RHO, dropout=[0.0, 0.1, 0.3]),
    "EQCov": dict(d=[32, 64, 128], n_layers=[1, 2, 3], k_emb=[8, 16, 32]),
    "Linear": {},
}


def draw(rng, space):
    out = {k: v[rng.integers(len(v))] for k, v in space.items()}
    if "phi" in out:  # Deep Sets: layer widths follow d
        out["phi"] = (out["d"],) * len(out["phi"])
        out["rho"] = (out["d"],) * len(out["rho"])
    return {k: list(v) if isinstance(v, tuple) else v for k, v in out.items()}


def configs(name, hp0):
    rng = np.random.default_rng([SEARCH_SEED, MODELS.index(name)])
    base = TrainCfg()
    out = [(hp0, dict(lr=base.lr, weight_decay=base.weight_decay, patience=base.patience, min_epochs=base.min_epochs))]
    for _ in range(N_RANDOM):
        out.append((draw(rng, ARCH_SPACE[name]), {k: float(v) if k in ("lr", "weight_decay") else int(v)
                                                  for k, v in draw(rng, CFG_SPACE).items()}))
    return out


if __name__ == "__main__":
    out_dir = RESULTS / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    hp_main = load_tuned_hp()
    hp_main["Linear"] = {}
    panel = PanelGPU(load_panel())
    log_path = out_dir / "tuning_log_T.jsonl"
    done = {}
    if log_path.exists():
        for r in map(json.loads, log_path.read_text().splitlines()):
            done[(r["model"], r["config"])] = r
    for name in MODELS:
        for k, (hp, cfg) in enumerate(configs(name, hp_main[name])):
            if (name, k) in done:
                continue
            vals, epochs, secs = [], [], 0.0
            for y in FOLDS:
                split = make_split(panel.dates, panel.first_valid, y, H)
                model, info = train_model(name, panel, split, H, seed=0, cfg=TrainCfg(**cfg), hp=hp)
                vals.append(info["best_val"])
                epochs.append(1 + int(np.argmin([r[2] for r in info["history"]])))
                secs += info["seconds"]
                del model
                torch.cuda.empty_cache()
            rec = dict(model=name, config=k, hp=hp, cfg=cfg, val=float(np.mean(vals)), val_folds=vals, best_epochs=epochs,
                       seconds=secs, n_params=info["n_params"])
            print(json.dumps(rec), flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            done[(name, k)] = rec
    best = {}
    for r in done.values():
        if r["model"] not in best or r["val"] < best[r["model"]]["val"]:
            best[r["model"]] = r
    (out_dir / "best_hp_T.json").write_text(json.dumps({m: dict(hp=r["hp"], cfg=r["cfg"]) for m, r in best.items()}, indent=1))
    for m, r in best.items():
        r0 = done[(m, 0)]
        print(f"{m:15s} config {r['config']:2d} val {r['val']:.5f} (current {r0['val']:.5f}) params {r['n_params']:7d} "
              f"hp {r['hp']} cfg {r['cfg']}")
