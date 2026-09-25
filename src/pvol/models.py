"""Neural forecasters of (log) portfolio variance.

All models map a batch dict (see dataset.PanelGPU.batch) to s = log of the forecast of average daily
portfolio variance over the horizon. Except for EQCov, the forecast is a residual on top of the log
EWMA portfolio variance ("base"), so every network starts from the RiskMetrics forecast.

Symmetry classes (w.r.t. a joint permutation of the assets in a portfolio):
  PortMLP   invariant by construction (sees only portfolio-level aggregates, no constituents)
  MLPConcat not invariant: constituents are concatenated into fixed slots (random order)
  MLPSorted invariant by canonicalisation: slots ordered by descending weight
  GRUConcat not invariant: recurrent model over the concatenated slot features
  DeepSets  invariant: rho(pool_i phi(x_i))  (Zaheer et al., 2017)
  PENet     equivariant layers with correlation-gated pairwise messages + invariant pooling
  SetTransformer  self-attention with correlation-biased logits + PMA pooling (Lee et al., 2019)
  SetGRU    shared per-asset GRU encoder + equivariant layers + invariant pooling (symmetric counterpart of GRU)
  EQCov     equivariant covariance model: Sigma_hat = D R_hat D, forecast = w' Sigma_hat w
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .universe import ASSET_CLASSES

N_NUM = 12  # numeric asset features (see data.node_features)
N_CLASS = len(ASSET_CLASSES)  # asset-class (ETF) or sector (stocks) dummies
F_ASSET = N_NUM + N_CLASS  # 19 for the ETF universe
F_PORT = 4  # w*N, Euler contribution (EWMA) * N, Euler contribution (22d) * N, corr(asset, portfolio)
F_PAIR = 3  # corr_22, corr_66, corr_ewma
F_G = 5  # log portfolio backcast variance over 1/5/22/66 days, log EWMA portfolio variance


def mlp(d_in, hidden, d_out, dropout=0.0, act=nn.GELU):
    layers, d = [], d_in
    for hdim in hidden:
        layers += [nn.Linear(d, hdim), act()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        d = hdim
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


def zero_last(module):
    """Initialise the last linear layer at zero so the untrained model reproduces the EWMA base forecast."""
    last = [m for m in module.modules() if isinstance(m, nn.Linear)][-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


def node_inputs(inp):
    B, N, _ = inp["x_asset"].shape
    g = inp["g"][:, None, :].expand(B, N, F_G)
    return torch.cat([inp["x_asset"], inp["x_port"], g], -1)  # (B, N, F_ASSET + F_PORT + F_G)


F_NODE = F_ASSET + F_PORT + F_G


# ---------------------------------------------------------------- aggregate / non-invariant baselines
class PortMLP(nn.Module):
    def __init__(self, hidden=(64, 64), dropout=0.0):
        super().__init__()
        self.net = mlp(F_G, hidden, 1, dropout)
        zero_last(self.net)

    def forward(self, inp):
        return inp["base"] + self.net(inp["g"]).squeeze(-1)


def pad_slots(inp, n_slots, sort_by_weight=False):
    x = torch.cat([inp["x_asset"], inp["x_port"]], -1)
    p = inp["pair"]
    B, N, Fx = x.shape
    if N > n_slots:
        raise ValueError(f"portfolio has {N} assets but the model has only {n_slots} slots")
    if sort_by_weight:
        order = inp["w"].argsort(1, descending=True)
        x = torch.gather(x, 1, order[:, :, None].expand(B, N, Fx))
        p = p[torch.arange(B, device=p.device)[:, None, None], order[:, :, None], order[:, None, :]]
    xp = x.new_zeros(B, n_slots, Fx + 1)
    xp[:, :N, :Fx] = x
    xp[:, :N, Fx] = 1.0  # slot-occupied flag
    pp = p.new_zeros(B, n_slots, n_slots, F_PAIR)
    pp[:, :N, :N] = p
    iu = torch.triu_indices(n_slots, n_slots, 1, device=p.device)
    return xp, pp[:, iu[0], iu[1]]  # (B, S, Fx+1), (B, S(S-1)/2, F_PAIR)


class MLPConcat(nn.Module):
    def __init__(self, n_slots=20, hidden=(512, 256), dropout=0.1, sort_by_weight=False):
        super().__init__()
        self.n_slots, self.sort = n_slots, sort_by_weight
        d_in = n_slots * (F_ASSET + F_PORT + 1) + n_slots * (n_slots - 1) // 2 * F_PAIR + F_G
        self.net = mlp(d_in, hidden, 1, dropout)
        zero_last(self.net)

    def forward(self, inp):
        xp, pu = pad_slots(inp, self.n_slots, self.sort)
        z = torch.cat([xp.flatten(1), pu.flatten(1), inp["g"]], -1)
        return inp["base"] + self.net(z).squeeze(-1)


class GRUConcat(nn.Module):
    """GRU over the last L days of slot-concatenated asset features; current pair/portfolio features in the head."""

    def __init__(self, n_slots=20, d_hidden=128, head=(256,), dropout=0.1):
        super().__init__()
        self.n_slots = n_slots
        self.gru = nn.GRU(n_slots * N_NUM, d_hidden, batch_first=True)
        d_head = d_hidden + n_slots * (N_CLASS + F_PORT + 1) + n_slots * (n_slots - 1) // 2 * F_PAIR + F_G
        self.head = mlp(d_head, head, 1, dropout)
        zero_last(self.head)

    def forward(self, inp):
        seq = inp["x_seq"]  # (B, L, N, 12)
        B, L, N, Fs = seq.shape
        sp = seq.new_zeros(B, L, self.n_slots, Fs)
        sp[:, :, :N] = seq
        _, hN = self.gru(sp.flatten(2))
        xp, pu = pad_slots(inp, self.n_slots)
        static = torch.cat([xp[..., N_NUM:]], -1)  # class dummies, portfolio features, occupied flag
        z = torch.cat([hN[-1], static.flatten(1), pu.flatten(1), inp["g"]], -1)
        return inp["base"] + self.head(z).squeeze(-1)


# ---------------------------------------------------------------- permutation-invariant models
def invariant_pools(h, inp):
    """Mean, weight-weighted and risk-contribution-weighted sums over the set (all permutation invariant)."""
    w = inp["w"][:, :, None]
    rc = inp["x_port"][:, :, 1:2] / inp["x_port"].shape[1]  # Euler contributions (EWMA), sum to 1
    return torch.cat([h.mean(1), (w * h).sum(1), (rc * h).sum(1)], -1)


class DeepSets(nn.Module):
    def __init__(self, d=128, phi=(128,), rho=(128,), dropout=0.0):
        super().__init__()
        self.phi = mlp(F_NODE, phi, d, dropout)
        self.rho = mlp(3 * d + F_G, rho, 1, dropout)
        zero_last(self.rho)

    def forward(self, inp):
        h = self.phi(node_inputs(inp))
        z = torch.cat([invariant_pools(F.gelu(h), inp), inp["g"]], -1)
        return inp["base"] + self.rho(z).squeeze(-1)


class DeepSetsP(DeepSets):
    """Deep Sets whose element features also include the weight-averaged correlation of each constituent with the others,
    c_k = sum_l w_l p_kl, for the three correlation measures (experiment R4). The summary is permutation equivariant and
    parameter-free, so the model stays a Deep Set; it gives the element map first-order pairwise information even when
    the engineered portfolio summaries are removed (raw inputs)."""

    def __init__(self, d=128, phi=(128,), rho=(128,), dropout=0.0):
        super().__init__(d, phi, rho, dropout)
        self.phi = mlp(F_NODE + F_PAIR, phi, d, dropout)

    def forward(self, inp):
        c = torch.einsum("bklf,bl->bkf", inp["pair"], inp["w"])
        h = self.phi(torch.cat([node_inputs(inp), c], -1))
        z = torch.cat([invariant_pools(F.gelu(h), inp), inp["g"]], -1)
        return inp["base"] + self.rho(z).squeeze(-1)


class PELayer(nn.Module):
    """h_i <- h_i + MLP([h_i, mean_j h_j, (1/N) sum_j gate(p_ij) * h_j]); equivariant to joint permutations."""

    def __init__(self, d, d_edge=F_PAIR):
        super().__init__()
        self.gate = mlp(d_edge, (32,), d)
        self.upd = mlp(3 * d, (2 * d,), d)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, pair):
        N = h.shape[1]
        msg = torch.einsum("bijd,bjd->bid", self.gate(pair), h) / N
        return h + self.upd(torch.cat([self.norm(h), h.mean(1, keepdim=True).expand_as(h), msg], -1))


class PENet(nn.Module):
    def __init__(self, d=64, n_layers=2, rho=(128,), dropout=0.0):
        super().__init__()
        self.emb = nn.Linear(F_NODE, d)
        self.layers = nn.ModuleList([PELayer(d) for _ in range(n_layers)])
        self.rho = mlp(3 * d + F_G, rho, 1, dropout)
        zero_last(self.rho)

    def forward(self, inp):
        h = self.emb(node_inputs(inp))
        for layer in self.layers:
            h = layer(h, inp["pair"])
        z = torch.cat([invariant_pools(h, inp), inp["g"]], -1)
        return inp["base"] + self.rho(z).squeeze(-1)


class BiasedMHA(nn.Module):
    """Multi-head attention whose logits get an additive bias computed from pairwise (correlation) features."""

    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.bias = mlp(F_PAIR, (16,), heads)

    def forward(self, x, pair):
        B, N, d = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dk) + self.bias(pair).permute(0, 3, 1, 2)
        a = logits.softmax(-1) @ v  # (B, h, N, dk)
        return self.out(a.transpose(1, 2).reshape(B, N, d))


class SAB(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.att = BiasedMHA(d, heads)
        self.ff = mlp(d, (2 * d,), d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x, pair):
        x = x + self.att(self.n1(x), pair)
        return x + self.ff(self.n2(x))


class PMA(nn.Module):
    """Pooling by multi-head attention with k learned seeds (Lee et al., 2019)."""

    def __init__(self, d, heads, k=1):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, k, d) / math.sqrt(d))
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        s = self.seed.expand(x.shape[0], -1, -1)
        o, _ = self.att(s, self.norm(x), self.norm(x), need_weights=False)
        return (s + o).flatten(1)


class SetTransformer(nn.Module):
    def __init__(self, d=64, heads=4, n_layers=2, rho=(128,), dropout=0.0):
        super().__init__()
        self.emb = nn.Linear(F_NODE, d)
        self.blocks = nn.ModuleList([SAB(d, heads) for _ in range(n_layers)])
        self.pma = PMA(d, heads)
        self.rho = mlp(d + 3 * d + F_G, rho, 1, dropout)
        zero_last(self.rho)

    def forward(self, inp):
        x = self.emb(node_inputs(inp))
        for blk in self.blocks:
            x = blk(x, inp["pair"])
        z = torch.cat([self.pma(x), invariant_pools(x, inp), inp["g"]], -1)
        return inp["base"] + self.rho(z).squeeze(-1)


class SetGRU(nn.Module):
    """Symmetric counterpart of GRUConcat (same information set).

    A GRU with weights shared across assets encodes each asset's last L days of features; the codes are combined
    with the current element/portfolio features, passed through correlation-gated equivariant layers (PELayer)
    and pooled invariantly. Sharing the encoder across assets makes the whole map permutation invariant.
    """

    def __init__(self, d=64, d_gru=32, n_layers=1, rho=(128,), dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(N_NUM, d_gru, batch_first=True)
        self.emb = nn.Linear(d_gru + F_NODE, d)
        self.layers = nn.ModuleList([PELayer(d) for _ in range(n_layers)])
        self.rho = mlp(3 * d + F_G, rho, 1, dropout)
        zero_last(self.rho)

    def forward(self, inp):
        seq = inp["x_seq"]  # (B, L, N, 12)
        B, L, N, Fs = seq.shape
        _, hN = self.gru(seq.permute(0, 2, 1, 3).reshape(B * N, L, Fs))
        h = self.emb(torch.cat([hN[-1].view(B, N, -1), node_inputs(inp)], -1))
        for layer in self.layers:
            h = layer(h, inp["pair"])
        z = torch.cat([invariant_pools(h, inp), inp["g"]], -1)
        return inp["base"] + self.rho(z).squeeze(-1)


class EQCov(nn.Module):
    """Permutation-equivariant covariance forecaster.

    Weight-free equivariant encoder over the assets -> per-asset log-vol adjustments delta_i and unit-norm
    embeddings a_i. Correlation forecast R_hat = sum_k pi_k P_k + pi_G * Gram(a), a convex combination of
    valid correlation matrices (PSD, unit diagonal), with mixing weights pi from the pooled encoder state.
    Sigma_hat = D R_hat D, D = diag(sigma_ewma * exp(delta)); forecast = w' Sigma_hat w.
    Permuting the assets permutes Sigma_hat accordingly (equivariance), so w' Sigma_hat w is invariant.
    """

    def __init__(self, d=64, n_layers=2, k_emb=8):
        super().__init__()
        self.emb = nn.Linear(F_ASSET, d)
        self.layers = nn.ModuleList([PELayer(d) for _ in range(n_layers)])
        self.vol = mlp(d, (d,), 1)
        self.a = nn.Linear(d, k_emb)
        self.mix = mlp(d, (d,), F_PAIR + 1)
        zero_last(self.vol)
        zero_last(self.mix)
        with torch.no_grad():  # start from the EWMA correlation
            self.mix[-1].bias.copy_(torch.tensor([0.0, 0.0, 3.0, -1.0]))

    def forward(self, inp, return_sigma=False):
        h = self.emb(inp["x_asset"])
        for layer in self.layers:
            h = layer(h, inp["pair"])
        sig = inp["sig"] * torch.exp(self.vol(h).squeeze(-1))
        a = F.normalize(self.a(h), dim=-1)
        pi = self.mix(h.mean(1)).softmax(-1)  # (B, 4)
        R = torch.einsum("bijk,bk->bij", inp["pair"], pi[:, :F_PAIR]) + pi[:, F_PAIR, None, None] * (a @ a.transpose(1, 2))
        Sigma = sig[:, :, None] * R * sig[:, None, :]
        w = inp["w"]
        var = torch.einsum("bi,bij,bj->b", w, Sigma, w).clamp_min(1e-12)
        return (torch.log(var), Sigma) if return_sigma else torch.log(var)


MODELS = {
    "PortMLP": PortMLP,
    "MLP": MLPConcat,
    "MLP-sorted": lambda **kw: MLPConcat(sort_by_weight=True, **kw),
    "GRU": GRUConcat,
    "DeepSets": DeepSets,
    "DeepSetsP": DeepSetsP,
    "PENet": PENet,
    "SetTransformer": SetTransformer,
    "SetGRU": SetGRU,
    "EQCov": EQCov,
}
SLOT_MODELS = {"MLP", "MLP-sorted", "GRU"}  # cannot score portfolios larger than n_slots
SEQ_MODELS = {"GRU", "SetGRU"}  # need the last 22 days of asset features (x_seq)
