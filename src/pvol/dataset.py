"""GPU-resident panel, portfolio sampler and batch construction.

A sample is (t, S, w): forecast origin t (close of day t), an ordered tuple S of N asset ids and
weights w. The target is the average daily realised variance of the constant-weight portfolio over
days t+1..t+h:  y = (1/h) * sum_k (w' R_{t+k, S})^2 = w' RC_{t,h} w / h.
Every quantity used as input is measurable w.r.t. information at the close of day t.
"""
from dataclasses import dataclass

import numpy as np
import torch

from .config import HAR_WINDOWS, MAX_LOOKBACK, VAR_FLOOR


@dataclass
class Split:
    """Index ranges (inclusive) of forecast origins, already trimmed so targets stay inside the range."""

    train: tuple
    val: tuple


def year_bounds(dates, year):
    idx = np.where(np.char.startswith(dates.astype(str), str(year)))[0]
    return int(idx[0]), int(idx[-1])


SEQ_MARGIN = 22  # the GRU needs 22 days of node features, so every model starts 22 days after first_valid


def make_split(dates, first_valid, test_year, h, train_years=None):
    """Train on origins up to end of test_year-2, validate on test_year-1 (targets kept inside each block).

    train_years restricts training to the last train_years calendar years before the validation year (ablation)."""
    v0, v1 = year_bounds(dates, test_year - 1)
    start = first_valid + SEQ_MARGIN
    if train_years is not None:
        start = max(start, year_bounds(dates, test_year - 1 - train_years)[0])
    return Split(train=(start, v0 - 1 - h), val=(v0, v1 - h))


def origins_in_year(dates, test_year, h):
    """Origins in test_year whose h-day target is observed in the sample."""
    a, b = year_bounds(dates, test_year)
    return np.arange(a, min(b, len(dates) - 1 - h) + 1)


class PanelGPU:
    def __init__(self, panel, device="cuda", asset_ids=None):
        self.device = device
        self.dates = panel["dates"]
        self.tickers = panel["tickers"]
        self.first_valid = int(panel["first_valid"])
        self.T, self.M = panel["R"].shape
        f = lambda a: torch.as_tensor(a, device=device)
        self.R = f(panel["R"])
        self.X_raw = f(panel["X"])
        self.onehot = f(panel["class_onehot"])
        self.P = f(panel["P"])
        self.S = f(panel["S_ewma"])
        self.asset_ids = torch.arange(self.M, device=device) if asset_ids is None else f(np.asarray(asset_ids))
        self.x_mu = None

    # ---------- standardisation (fit on training origins only) ----------
    def fit_scalers(self, train_range, gen, n=20000):
        a, b = train_range
        X = self.X_raw[a : b + 1][:, self.asset_ids]
        self.x_mu = X.mean(dim=(0, 1))
        self.x_sd = X.std(dim=(0, 1)).clamp_min(1e-6)
        t, S, w = self.sample(n, 10, train_range, gen)
        g = self._portfolio_history_feats(t, S, w)[0]
        self.g_mu = g.mean(0)
        self.g_sd = g.std(0).clamp_min(1e-6)

    # ---------- sampling ----------
    def sample(self, B, N, t_range, gen, alpha=1.0, scheme="long_only", pool=None):
        pool = self.asset_ids if pool is None else pool
        t = torch.randint(t_range[0], t_range[1] + 1, (B,), device=self.device, generator=gen)
        perm = torch.rand(B, len(pool), device=self.device, generator=gen).argsort(1)[:, :N]
        S = pool[perm]
        w = sample_weights(B, N, gen, self.device, alpha, scheme)
        return t, S, w

    # ---------- features ----------
    def _portfolio_history_feats(self, t, S, w):
        L = MAX_LOOKBACK
        lags = torch.arange(L - 1, -1, -1, device=self.device)  # oldest ... newest
        tt = t[:, None] - lags[None, :]  # (B, L)
        Rh = self.R[tt[:, :, None], S[:, None, :]]  # (B, L, N)
        rp = (Rh * w[:, None, :]).sum(-1)  # (B, L) backcast portfolio returns
        g = [torch.log((rp[:, -k:] ** 2).mean(1).clamp_min(VAR_FLOOR)) for k in HAR_WINDOWS]
        Ssub = self.S[t[:, None, None], S[:, :, None], S[:, None, :]]  # (B, N, N)
        Sw = torch.einsum("bij,bj->bi", Ssub, w)
        pv_ewma = (w * Sw).sum(-1).clamp_min(VAR_FLOOR)
        g.append(torch.log(pv_ewma))
        R22 = Rh[:, -22:]
        C22w = torch.einsum("bli,bl->bi", R22, rp[:, -22:]) / 22  # (C22 w)_i
        pv22 = (w * C22w).sum(-1).clamp_min(VAR_FLOOR)
        return torch.stack(g, -1), dict(Ssub=Ssub, Sw=Sw, pv_ewma=pv_ewma, C22w=C22w, pv22=pv22, Rh=Rh)

    def batch(self, t, S, w, h=None, gru_len=0):
        """Model inputs (dict of tensors) and, if h is given, the target y (B,)."""
        B, N = S.shape
        g, aux = self._portfolio_history_feats(t, S, w)
        X = (self.X_raw[t[:, None], S] - self.x_mu) / self.x_sd  # (B, N, F)
        oh = self.onehot[S]
        sig = torch.sqrt(torch.diagonal(aux["Ssub"], dim1=1, dim2=2).clamp_min(VAR_FLOOR))  # (B, N) EWMA vols
        rc_ewma = w * aux["Sw"] / aux["pv_ewma"][:, None]  # Euler risk contributions, sum to 1
        rc22 = w * aux["C22w"] / aux["pv22"][:, None]
        rho_ip = aux["Sw"] / (sig * aux["pv_ewma"].sqrt()[:, None])  # corr(asset i, portfolio)
        node_w = torch.stack([w * N, rc_ewma * N, rc22 * N, rho_ip], -1)
        out = dict(
            x_asset=torch.cat([X, oh], -1),  # weight-free asset features (B, N, F+C)
            x_port=node_w,  # portfolio-conditional node features (B, N, 4)
            pair=self.P[t[:, None, None], S[:, :, None], S[:, None, :]],  # (B, N, N, 3)
            g=(g - self.g_mu) / self.g_sd,  # (B, 5) portfolio-level HAR/EWMA features
            base=torch.log(aux["pv_ewma"]),  # log EWMA portfolio variance (residual anchor)
            w=w,
            sig=sig,
            corr_ewma=aux["Ssub"] / (sig[:, :, None] * sig[:, None, :]),
        )
        if gru_len:
            lags = torch.arange(gru_len - 1, -1, -1, device=self.device)
            tt = t[:, None] - lags[None, :]
            out["x_seq"] = (self.X_raw[tt[:, :, None], S[:, None, :]] - self.x_mu) / self.x_sd  # (B, L, N, F)
        y = None
        if h is not None:
            steps = torch.arange(1, h + 1, device=self.device)
            Rf = self.R[(t[:, None] + steps[None, :])[:, :, None], S[:, None, :]]  # (B, h, N)
            y = ((Rf * w[:, None, :]).sum(-1) ** 2).mean(1)
        return out, y


def sample_weights(B, N, gen, device, alpha=1.0, scheme="long_only"):
    if scheme == "long_only":
        if alpha is None or np.isinf(alpha):
            return torch.full((B, N), 1.0 / N, device=device)
        # Dirichlet(alpha) via normalised Gamma draws (torch.distributions has no generator argument)
        g = torch._standard_gamma(torch.full((B, N), float(alpha), device=device), generator=gen)
        return g / g.sum(1, keepdim=True)
    if scheme == "long_short":
        z = torch.randn(B, N, device=device, generator=gen)
        return z / z.abs().sum(1, keepdim=True)  # gross exposure 1
    raise ValueError(scheme)
