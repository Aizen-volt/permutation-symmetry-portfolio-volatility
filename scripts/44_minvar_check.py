"""Sixth series V5 (docs/JOURNAL.md, section 4g): does the optimizer of the minimum-variance evaluation matter?

Same design as scripts/13_minvar.py (h = 5, every 5th test origin, 100 random ten-asset subsets, long-only weights,
networks retrained with seed 0). For each model the weights of the original optimizer (200 Adam steps on softmax
logits from equal weights) are compared with
  covariance models: the exact minimiser of w' Sigma w on the simplex (projected gradient to convergence);
  networks: the best, by the model's own forecast, of equal start with 200 steps, five random starts with 200 steps
            and equal start with 1000 steps.
Rule: annualized realized volatility changes by less than 0.05 points for every model and the ranking is unchanged ->
the optimizer does not affect the result. Writes results*/main/tables/minvar_check.csv.

Usage: [PVOL_UNIVERSE=stocks] python scripts/44_minvar_check.py
"""
import importlib.util
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
from pvol.dataset import PanelGPU, make_split, origins_in_year  # noqa: E402
from pvol.experiment import load_tuned_hp  # noqa: E402
from pvol.train import train_model  # noqa: E402

_spec = importlib.util.spec_from_file_location("mv", Path(__file__).with_name("13_minvar.py"))
mv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mv)
H, N, K = mv.H, mv.N, mv.K
COV_MODELS = ["EWMA", "LW", "EWMANLS", "PCA-F"]


def optimise_from(loss_fn, z0, steps):
    z = z0.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=mv.LR)
    for _ in range(steps):
        loss = loss_fn(torch.softmax(z, 1)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    w = torch.softmax(z, 1).detach()
    with torch.no_grad():
        return w, loss_fn(w)


def project_simplex(v):
    u = torch.sort(v, 1, descending=True).values
    css = u.cumsum(1) - 1
    k = torch.arange(1, v.shape[1] + 1, device=v.device, dtype=v.dtype)
    rho = ((u - css / k) > 0).sum(1, keepdim=True)
    theta = css.gather(1, rho - 1) / rho
    return (v - theta).clamp_min(0)


def exact_minvar(Sig, iters=5000):
    """Projected gradient for min w' Sigma w on the simplex (convex), step 1 / (2 lambda_max)."""
    Sig = Sig.double()
    step = 1.0 / (2 * torch.linalg.eigvalsh(Sig)[:, -1:])
    w = torch.full(Sig.shape[:2], 1.0 / Sig.shape[1], device=Sig.device, dtype=Sig.dtype)
    for _ in range(iters):
        w = project_simplex(w - step * 2 * torch.einsum("bij,bj->bi", Sig, w))
    return w.float()


if __name__ == "__main__":
    p = load_panel()
    panel = PanelGPU(p)
    dev = panel.device
    hp = load_tuned_hp()
    R_np = p["R"].astype(np.float64)
    pool = torch.as_tensor([i for i, t in enumerate(p["tickers"]) if t != "SHY"], device=dev)
    rows = []
    for Y in TEST_YEARS:
        part = RESULTS / "main" / "tables" / f"minvar_check_{Y}.csv"  # per-year checkpoint, so a stopped run resumes
        if part.exists():
            rows += pd.read_csv(part).to_dict("records")
            continue
        t0 = time.time()
        n0 = len(rows)
        reb = origins_in_year(panel.dates, Y, H)[::H]
        gen = torch.Generator(device=dev).manual_seed(10_000 + Y)  # the subsets of scripts/13_minvar.py
        perm = torch.rand(len(reb), K, len(pool), device=dev, generator=gen).argsort(-1)[..., :N]
        subsets = pool[perm]
        covs = {"LW": bl.rolling_shrinkage_covs(R_np, reb, kind="lw"),
                "EWMANLS": bl.rolling_shrinkage_covs(R_np, reb, kind="ewma-nls", S_ewma=p["S_ewma"]),
                "PCA-F": bx.pca_factor_covs(R_np, p["S_ewma"], reb), "EWMA": p["S_ewma"][reb].astype(np.float64)}
        for m in mv.NETS + COV_MODELS:
            if m in mv.NETS:
                split = make_split(panel.dates, panel.first_valid, Y, H)
                net, _ = train_model(m, panel, split, H, seed=0, hp=hp[m])
                net.eval()
                for q in net.parameters():
                    q.requires_grad_(False)
            for k, t in enumerate(reb):
                S = subsets[k]
                tt = torch.full((K,), int(t), device=dev)
                z_eq = torch.zeros(K, N, device=dev)
                if m in mv.NETS:
                    f = mv.net_loss(net, m, panel, tt, S)
                    w0, l0 = optimise_from(f, z_eq, mv.STEPS)
                    best_w, best_l = w0, l0
                    g = torch.Generator(device=dev).manual_seed(int(t))
                    starts = [(torch.randn(K, N, device=dev, generator=g), mv.STEPS) for _ in range(5)]
                    for z0, steps in starts + [(z_eq, 1000)]:
                        w, l_ = optimise_from(f, z0, steps)
                        better = l_ < best_l
                        best_w = torch.where(better[:, None], w, best_w)
                        best_l = torch.where(better, l_, best_l)
                    w1 = best_w
                else:
                    Sig = torch.as_tensor(covs[m][k], device=dev, dtype=torch.float32)[S[:, :, None], S[:, None, :]]
                    f = mv.cov_loss(Sig)
                    w0, l0 = optimise_from(f, z_eq, mv.STEPS)
                    w1 = exact_minvar(Sig)
                with torch.no_grad():
                    l1 = f(w1)
                rv0 = mv.realised(panel.R, tt, S, w0).cpu().numpy()
                rv1 = mv.realised(panel.R, tt, S, w1).cpu().numpy()
                gap = (l0 - l1).cpu().numpy()  # log forecast of the original minus that of the better solution
                for j in range(K):
                    rows.append(dict(year=Y, origin=int(t), subset=j, model=m, rv_adam=rv0[j], rv_best=rv1[j],
                                     log_forecast_gap=gap[j]))
            if m in mv.NETS:
                del net
                torch.cuda.empty_cache()
        pd.DataFrame(rows[n0:]).to_csv(part, index=False)
        print(f"{Y} done in {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "main" / "tables" / "minvar_check.csv", index=False)
    vol = lambda v: 100 * np.sqrt(v.mean() * 252)
    summ = df.groupby("model").agg(vol_adam=("rv_adam", vol), vol_best=("rv_best", vol),
                                   median_gap=("log_forecast_gap", "median"), p99_gap=("log_forecast_gap",
                                                                                      lambda x: np.quantile(x, 0.99)))
    summ["change_pp"] = summ.vol_best - summ.vol_adam
    summ["rank_adam"] = summ.vol_adam.rank()
    summ["rank_best"] = summ.vol_best.rank()
    summ.to_csv(RESULTS / "main" / "tables" / "minvar_check_summary.csv")
    print(summ.sort_values("vol_adam").round(4))
