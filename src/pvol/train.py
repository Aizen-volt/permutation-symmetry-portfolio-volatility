"""Training loop (QLIKE loss on randomly sampled portfolios) and batched prediction for fixed portfolios."""
import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .models import MODELS, SEQ_MODELS, SLOT_MODELS


@dataclass
class TrainCfg:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch: int = 256
    steps_per_epoch: int = 125
    max_epochs: int = 60
    patience: int = 10
    n_min: int = 2
    n_max: int = 20
    alphas: tuple = (0.3, 1.0, 3.0)
    grad_clip: float = 1.0
    val_batches: int = 38  # 38 * 256 ~ 9.7k fixed validation portfolios-days
    gru_len: int = 22
    ema_decay: float = 0.99  # exponential moving average of weights, used for validation and prediction
    min_epochs: int = 0  # robustness protocol: never stop before this many epochs
    top_k_avg: int = 1  # robustness protocol: average the EMA weights of the k best validation epochs


def qlike(s, y):
    """Patton (2011) QLIKE with forecast exp(s) of the variance proxy y; minimum 0 at exp(s) = y."""
    y = y.clamp_min(1e-12)
    return y * torch.exp(-s) + s - torch.log(y) - 1.0


def build_model(name, hp=None):
    return MODELS[name](**(hp or {}))


def fixed_batches(panel, t_range, n_batches, cfg, seed):
    gen = torch.Generator(device=panel.device).manual_seed(seed)
    out = []
    for k in range(n_batches):
        N = cfg.n_min + k % (cfg.n_max - cfg.n_min + 1)
        alpha = cfg.alphas[k % len(cfg.alphas)]
        out.append((N, panel.sample(cfg.batch, N, t_range, gen, alpha=alpha)))
    return out


@torch.no_grad()
def evaluate_batches(model, panel, batches, h, gru_len):
    model.eval()
    tot, n = 0.0, 0
    for _, (t, S, w) in batches:
        inp, y = panel.batch(t, S, w, h=h, gru_len=gru_len)
        loss = qlike(model(inp), y)
        tot += loss.sum().item()
        n += loss.numel()
    return tot / n


def train_model(name, panel, split, h, seed, cfg=None, hp=None, verbose=False):
    cfg = cfg or TrainCfg()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(name, hp).to(panel.device)
    ema = copy.deepcopy(model)
    for q in ema.parameters():
        q.requires_grad_(False)
    gru_len = cfg.gru_len if name in SEQ_MODELS else 0
    gen = torch.Generator(device=panel.device).manual_seed(seed)
    panel.fit_scalers(split.train, torch.Generator(device=panel.device).manual_seed(0))
    val = fixed_batches(panel, split.val, cfg.val_batches, cfg, seed=987 + h)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
    best, best_state, wait, hist, top = np.inf, None, 0, [], []
    t0 = time.time()
    rng = np.random.default_rng(seed)
    for epoch in range(cfg.max_epochs):
        model.train()
        tr = 0.0
        for _ in range(cfg.steps_per_epoch):
            N = int(rng.integers(cfg.n_min, cfg.n_max + 1))
            alpha = float(rng.choice(cfg.alphas))
            t, S, w = panel.sample(cfg.batch, N, split.train, gen, alpha=alpha)
            inp, y = panel.batch(t, S, w, h=h, gru_len=gru_len)
            loss = qlike(model(inp), y).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            with torch.no_grad():
                for e, q in zip(ema.parameters(), model.parameters()):
                    e.lerp_(q, 1 - cfg.ema_decay)
            tr += loss.item()
        sched.step()
        v = evaluate_batches(ema, panel, val, h, gru_len)
        hist.append((epoch, tr / cfg.steps_per_epoch, v))
        if verbose:
            print(f"  ep {epoch:3d} train {tr / cfg.steps_per_epoch:.4f} val {v:.4f}  ({time.time() - t0:.0f}s)")
        if cfg.top_k_avg > 1:
            top.append((v, copy.deepcopy(ema.state_dict())))
            top = sorted(top, key=lambda z: z[0])[: cfg.top_k_avg]
        if v < best - 1e-5:
            best, best_state, wait = v, copy.deepcopy(ema.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg.patience and epoch + 1 >= cfg.min_epochs:
                break
    if cfg.top_k_avg > 1:
        best_state = {k: sum(st[k] for _, st in top) / len(top) for k in best_state}
    model.load_state_dict(best_state)
    info = dict(model=name, h=h, seed=seed, best_val=best, epochs=len(hist), seconds=time.time() - t0,
                n_params=sum(p.numel() for p in model.parameters()), history=hist, cfg=asdict(cfg), hp=hp or {})
    return model, info


@torch.no_grad()
def predict_portfolios(model, name, panel, portfolios, t_idx, gru_len=22, chunk=8192):
    """Forecasts s = log variance for every (origin, portfolio): array (len(t_idx), len(portfolios)); NaN if unsupported."""
    model.eval()
    dev = panel.device
    out = np.full((len(t_idx), len(portfolios)), np.nan, dtype=np.float32)
    n_slots = getattr(model, "n_slots", None)
    by_n = {}
    for j, p in enumerate(portfolios):
        by_n.setdefault(len(p["ids"]), []).append(j)
    t_all = torch.as_tensor(np.asarray(t_idx), device=dev)
    for N, cols in by_n.items():
        if name in SLOT_MODELS and N > n_slots:
            continue
        S = torch.as_tensor([portfolios[j]["ids"] for j in cols], device=dev)
        W = torch.as_tensor([portfolios[j]["weights"] for j in cols], device=dev, dtype=torch.float32)
        P = len(cols)
        tt = t_all.repeat_interleave(P)
        SS, WW = S.repeat(len(t_idx), 1), W.repeat(len(t_idx), 1)
        step = max(1, chunk * 100 // (N * N)) if N > 10 else chunk  # pairwise tensors grow with N^2
        preds = []
        for a in range(0, len(tt), step):
            inp, _ = panel.batch(tt[a : a + step], SS[a : a + step], WW[a : a + step], gru_len=gru_len if name in SEQ_MODELS else 0)
            preds.append(model(inp).float())
        out[:, cols] = torch.cat(preds).view(len(t_idx), P).cpu().numpy()
    return out


@torch.no_grad()
def realized_targets(panel, portfolios, t_idx, h):
    """Average daily realised variance over t+1..t+h for each origin/portfolio: (len(t_idx), len(portfolios))."""
    R = panel.R.cpu().numpy().astype(np.float64)
    t_idx = np.asarray(t_idx)
    out = np.empty((len(t_idx), len(portfolios)))
    for j, p in enumerate(portfolios):
        rp = R[:, p["ids"]] @ np.asarray(p["weights"])
        c = np.concatenate([[0.0], np.cumsum(rp**2)])
        out[:, j] = (c[t_idx + h + 1] - c[t_idx + 1]) / h
    return out
