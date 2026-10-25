"""Hyper-parameter selection on a tuning fold that precedes all test years.

Train origins up to end-2016, validation (early stopping + selection) on 2017-2018, horizon h=5, seed 0.
Every architecture gets the same budget: 4 configurations.

Check of the tuning choice (not used for the forecasts): --h H --tag NAME repeats the search at horizon H and writes
tuning_log_NAME.jsonl and best_hp_NAME.json, leaving best_hp.json unchanged.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402

from pvol.config import RESULTS  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.dataset import SEQ_MARGIN, PanelGPU, Split, year_bounds  # noqa: E402
from pvol.train import train_model  # noqa: E402

GRIDS = {
    "PortMLP": [dict(hidden=(32, 32)), dict(hidden=(64, 64)), dict(hidden=(128, 128)), dict(hidden=(64, 64, 64))],
    "MLP": [dict(hidden=(256,), dropout=0.1), dict(hidden=(512, 256), dropout=0.1),
            dict(hidden=(512, 256), dropout=0.3), dict(hidden=(256, 256), dropout=0.3)],
    "MLP-sorted": [dict(hidden=(256,), dropout=0.1), dict(hidden=(512, 256), dropout=0.1),
                   dict(hidden=(512, 256), dropout=0.3), dict(hidden=(256, 256), dropout=0.3)],
    "GRU": [dict(d_hidden=64, dropout=0.1), dict(d_hidden=128, dropout=0.1),
            dict(d_hidden=64, dropout=0.3), dict(d_hidden=128, dropout=0.3)],
    "DeepSets": [dict(d=64, phi=(64,), rho=(64,)), dict(d=128, phi=(128,), rho=(128,)),
                 dict(d=128, phi=(128, 128), rho=(128,)), dict(d=256, phi=(256,), rho=(256,))],
    "PENet": [dict(d=32, n_layers=2), dict(d=64, n_layers=1), dict(d=64, n_layers=2), dict(d=64, n_layers=3)],
    "SetTransformer": [dict(d=32, n_layers=2), dict(d=64, n_layers=1), dict(d=64, n_layers=2), dict(d=64, n_layers=3)],
    "SetGRU": [dict(d=64, d_gru=32, n_layers=1), dict(d=64, d_gru=64, n_layers=1),
               dict(d=64, d_gru=32, n_layers=2), dict(d=32, d_gru=32, n_layers=2)],
    "EQCov": [dict(d=32, n_layers=2), dict(d=64, n_layers=1), dict(d=64, n_layers=2), dict(d=64, n_layers=2, k_emb=16)],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    h = args.h
    sfx = f"_{args.tag}" if args.tag else ""
    out_dir = RESULTS / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = load_panel()
    panel = PanelGPU(p)
    v0 = year_bounds(panel.dates, 2017)[0]
    v1 = year_bounds(panel.dates, 2018)[1]
    split = Split(train=(panel.first_valid + SEQ_MARGIN, v0 - 1 - h), val=(v0, v1 - h))
    log_path = out_dir / f"tuning_log{sfx}.jsonl"
    done = set()
    if log_path.exists():
        done = {(r["model"], json.dumps(r["hp"], sort_keys=True)) for r in map(json.loads, log_path.read_text().splitlines())}
    for name, grid in GRIDS.items():
        for hp in grid:
            key = (name, json.dumps(json.loads(json.dumps(hp)), sort_keys=True))
            if key in done:
                continue
            model, info = train_model(name, panel, split, h, seed=0, hp=hp)
            rec = dict(model=name, hp=hp, best_val=info["best_val"], epochs=info["epochs"],
                       seconds=info["seconds"], n_params=info["n_params"])
            print(json.dumps(rec), flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            del model
            torch.cuda.empty_cache()
    recs = [json.loads(line) for line in log_path.read_text().splitlines()]
    best = {}
    for r in recs:
        if r["model"] not in best or r["best_val"] < best[r["model"]]["best_val"]:
            best[r["model"]] = r
    (out_dir / f"best_hp{sfx}.json").write_text(json.dumps({k: v["hp"] for k, v in best.items()}, indent=1))
    for k, v in best.items():
        print(f"{k:15s} val {v['best_val']:.4f} params {v['n_params']:7d} hp {v['hp']}")
