"""Economic evaluation 2: long-only minimum-variance portfolios built with each model as the risk model (h = 5).

At every 5th test origin, K random subsets of N=10 funds are drawn (identical for all models; SHY, a cash proxy, is
excluded). Each model chooses long-only weights w = softmax(z) minimising its own 5-day variance forecast, by gradient
descent through the model (neural networks) or through w' Sigma_hat w (covariance models). The realised variance
of the chosen portfolio over the next 5 days is recorded: a better risk model yields lower realised risk.
Networks are retrained (seed 0) with the tuned hyper-parameters and the same expanding-window protocol.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from pvol import baselines as bl  # noqa: E402
from pvol import baselines_ext as bx  # noqa: E402
from pvol.config import RESULTS, TEST_YEARS  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.dataset import PanelGPU, make_split, origins_in_year, year_bounds  # noqa: E402
from pvol.experiment import load_tuned_hp  # noqa: E402
from pvol.models import SEQ_MODELS  # noqa: E402
from pvol.train import train_model  # noqa: E402

H, N, K, STEPS, LR = 5, 10, 100, 200, 0.1
NETS = ["PortMLP", "MLP", "MLP-sorted", "GRU", "DeepSets", "PENet", "SetTransformer", "SetGRU", "EQCov"]
COV_MODELS = ["EWMA", "DCC", "DCCNL", "LW", "EWMANLS", "PCA-F"]


def optimise(loss_fn, B, device, steps=STEPS, lr=LR):
    z = torch.zeros(B, N, device=device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        loss = loss_fn(torch.softmax(z, 1)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.softmax(z, 1).detach()


def net_loss(model, name, panel, t, S):
    gru_len = 22 if name in SEQ_MODELS else 0

    def f(w):
        inp, _ = panel.batch(t, S, w, gru_len=gru_len)
        return model(inp)  # log variance forecast

    return f


def cov_loss(Sig):
    def f(w):
        return torch.log(torch.einsum("bi,bij,bj->b", w, Sig, w))

    return f


def realised(R, t, S, w):
    steps = torch.arange(1, H + 1, device=R.device)
    Rf = R[(t[:, None] + steps[None, :])[:, :, None], S[:, None, :]]
    return ((Rf * w[:, None, :]).sum(-1) ** 2).mean(1)  # average daily realised variance


if __name__ == "__main__":
    models = sys.argv[1:] or NETS + COV_MODELS + ["EW"]
    p = load_panel()
    panel = PanelGPU(p)
    dev = panel.device
    hp = load_tuned_hp()
    R_np = p["R"].astype(np.float64)
    pool = torch.as_tensor([i for i, t in enumerate(p["tickers"]) if t != "SHY"], device=dev)
    rows = []
    for Y in TEST_YEARS:
        t0 = time.time()
        reb = origins_in_year(panel.dates, Y, H)[::H]
        gen = torch.Generator(device=dev).manual_seed(10_000 + Y)
        perm = torch.rand(len(reb), K, len(pool), device=dev, generator=gen).argsort(-1)[..., :N]
        subsets = pool[perm]  # (dates, K, N) identical for every model
        fit_end = year_bounds(panel.dates, Y - 1)[1]
        covs = {}
        if "DCC" in models or "DCCNL" in models:
            for name, target in (("DCC", "sample"), ("DCCNL", "nls")):
                if name in models:
                    d = bl.DCC().fit(R_np, fit_end, n_jobs=6, target=target)
                    covs[name] = np.stack([d.covariance_forecast(t, H) for t in reb])
        if "LW" in models:
            covs["LW"] = bl.rolling_shrinkage_covs(R_np, reb, kind="lw")
        if "EWMANLS" in models:
            covs["EWMANLS"] = bl.rolling_shrinkage_covs(R_np, reb, kind="ewma-nls", S_ewma=p["S_ewma"])
        if "PCA-F" in models:
            covs["PCA-F"] = bx.pca_factor_covs(R_np, p["S_ewma"], reb)
        if "EWMA" in models:
            covs["EWMA"] = p["S_ewma"][reb].astype(np.float64)
        for m in models:
            if m in NETS:
                split = make_split(panel.dates, panel.first_valid, Y, H)
                net, _ = train_model(m, panel, split, H, seed=0, hp=hp[m])
                net.eval()
                for q in net.parameters():
                    q.requires_grad_(False)
            for k, t in enumerate(reb):
                S = subsets[k]
                tt = torch.full((K,), int(t), device=dev)
                if m == "EW":
                    w = torch.full((K, N), 1.0 / N, device=dev)
                elif m in NETS:
                    w = optimise(net_loss(net, m, panel, tt, S), K, dev)
                else:
                    Sig = torch.as_tensor(covs[m][k], device=dev, dtype=torch.float32)
                    Sig = Sig[S[:, :, None], S[:, None, :]]
                    w = optimise(cov_loss(Sig), K, dev)
                rv = realised(panel.R, tt, S, w).cpu().numpy()
                hhi = (w**2).sum(1).cpu().numpy()
                for j in range(K):
                    rows.append(dict(year=Y, origin=int(t), subset=j, model=m, realised_var=rv[j], hhi=hhi[j]))
            if m in NETS:
                del net
                torch.cuda.empty_cache()
        print(f"{Y} done in {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    out = RESULTS / "main" / "tables" / "minvar_h5.csv"
    if out.exists() and sys.argv[1:]:
        df = pd.concat([pd.read_csv(out).query("model not in @models"), df])
    df.to_csv(out, index=False)
    summ = df.groupby("model").agg(ann_vol=("realised_var", lambda v: np.sqrt(v.mean() * 252)), hhi=("hhi", "mean"))
    print(summ.sort_values("ann_vol").round(4))
