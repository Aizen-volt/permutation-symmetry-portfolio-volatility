"""Validation-only diagnostic: on the tuning fold (train <= 2016, validation 2017-2018), how do networks trained for
h=22 compare with HAR and EWMA on the fixed validation portfolios? No test-year data are used."""
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch

from pvol.data import load_panel
from pvol.dataset import SEQ_MARGIN, PanelGPU, Split, year_bounds
from pvol.experiment import load_tuned_hp
from pvol.train import TrainCfg, evaluate_batches, fixed_batches, qlike, train_model

p = load_panel()
panel = PanelGPU(p)
hp = load_tuned_hp()
R = p["R"].astype(np.float64)
for h in (22, 5):
    v0 = year_bounds(panel.dates, 2017)[0]
    v1 = year_bounds(panel.dates, 2018)[1]
    split = Split(train=(panel.first_valid + SEQ_MARGIN, v0 - 1 - h), val=(v0, v1 - h))
    cfg = TrainCfg()
    panel.fit_scalers(split.train, torch.Generator(device="cuda").manual_seed(0))
    val = fixed_batches(panel, split.val, cfg.val_batches, cfg, seed=987 + h)
    # HAR (per sampled portfolio, OLS on the training origins) and EWMA on exactly the same validation samples
    fit_idx = np.arange(split.train[0], split.train[1] + 1)
    L_har, L_ewma = [], []
    for _, (t, S, w) in val:
        t_np, S_np, w_np = t.cpu().numpy(), S.cpu().numpy(), w.cpu().numpy().astype(np.float64)
        inp, y = panel.batch(t, S, w, h=h)
        y = y.double().cpu().numpy()
        L_ewma.append(qlike(torch.as_tensor(inp["base"].double().cpu()), torch.as_tensor(y)).numpy())
        f = np.empty(len(t_np))
        for k in range(len(t_np)):
            rp = R[:, S_np[k]] @ w_np[k]
            c = np.concatenate([[0.0], np.cumsum(rp**2)])
            X = lambda idx: np.column_stack([np.ones(len(idx))] + [(c[idx + 1] - c[idx + 1 - m]) / m for m in (1, 5, 22)])
            yy = (c[fit_idx + h + 1] - c[fit_idx + 1]) / h
            beta, *_ = np.linalg.lstsq(X(fit_idx), yy, rcond=None)
            fk = X(np.array([t_np[k]])) @ beta
            f[k] = fk[0] if fk[0] > 1e-8 else (c[t_np[k] + 1] - c[t_np[k] - 21]) / 22
        L_har.append(qlike(torch.as_tensor(np.log(f)), torch.as_tensor(y)).numpy())
    print(f"h={h}: validation QLIKE  EWMA {np.concatenate(L_ewma).mean():.4f}  HAR {np.concatenate(L_har).mean():.4f}", flush=True)
    for name in ("PortMLP", "MLP", "DeepSets", "EQCov"):
        model, info = train_model(name, panel, split, h, seed=0, hp=hp[name])
        print(f"   {name:10s} val {info['best_val']:.4f}  epochs {info['epochs']}", flush=True)
