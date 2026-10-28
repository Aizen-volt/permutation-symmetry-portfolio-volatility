"""Seventh series W4: minimum variance with EQ-Cov solved exactly.

The encoder of EQ-Cov does not receive the portfolio weights, so its covariance forecast Sigma_hat does not depend on
w and min w' Sigma_hat w on the simplex is a convex quadratic programme. Same design as scripts/44_minvar_check.py
(h = 5, every 5th test origin, the same 100 ten-asset subsets, EQ-Cov retrained with seed 0); the exact minimiser
(projected gradient to convergence) is compared with the weights of the original optimizer (200 Adam steps from equal
weights). Rule (as V5): the annualized realized volatility changes by less than 0.05 points.
Writes results*/main/tables/minvar_eqcov_exact.csv.

Usage: [PVOL_UNIVERSE=stocks] python scripts/52_minvar_eqcov_exact.py
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from pvol.config import RESULTS, TEST_YEARS  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.dataset import PanelGPU, make_split, origins_in_year  # noqa: E402
from pvol.experiment import load_tuned_hp  # noqa: E402
from pvol.train import train_model  # noqa: E402


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mv = load("mv", "13_minvar.py")
chk = load("chk", "44_minvar_check.py")
H, N, K = mv.H, mv.N, mv.K

if __name__ == "__main__":
    p = load_panel()
    panel = PanelGPU(p)
    dev = panel.device
    hp = load_tuned_hp()
    pool = torch.as_tensor([i for i, t in enumerate(p["tickers"]) if t != "SHY"], device=dev)
    rows = []
    for Y in TEST_YEARS:
        reb = origins_in_year(panel.dates, Y, H)[::H]
        gen = torch.Generator(device=dev).manual_seed(10_000 + Y)  # the subsets of scripts/13_minvar.py
        perm = torch.rand(len(reb), K, len(pool), device=dev, generator=gen).argsort(-1)[..., :N]
        subsets = pool[perm]
        split = make_split(panel.dates, panel.first_valid, Y, H)
        net, _ = train_model("EQCov", panel, split, H, seed=0, hp=hp["EQCov"])
        net.eval()
        for q in net.parameters():
            q.requires_grad_(False)
        for k, t in enumerate(reb):
            S = subsets[k]
            tt = torch.full((K,), int(t), device=dev)
            f = mv.net_loss(net, "EQCov", panel, tt, S)
            w0, l0 = chk.optimise_from(f, torch.zeros(K, N, device=dev), mv.STEPS)
            with torch.no_grad():
                inp, _ = panel.batch(tt, S, torch.full((K, N), 1.0 / N, device=dev))
                _, Sig = net(inp, return_sigma=True)
                w1 = chk.exact_minvar(Sig)
                l1 = f(w1)
            rv0 = mv.realised(panel.R, tt, S, w0).cpu().numpy()
            rv1 = mv.realised(panel.R, tt, S, w1).cpu().numpy()
            gap = (l0 - l1).cpu().numpy()
            for j in range(K):
                rows.append(dict(year=Y, origin=int(t), subset=j, rv_adam=rv0[j], rv_exact=rv1[j],
                                 log_forecast_gap=gap[j]))
        del net
        torch.cuda.empty_cache()
        print(f"{Y} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "main" / "tables" / "minvar_eqcov_exact.csv", index=False)
    vol = lambda v: 100 * np.sqrt(v.mean() * 252)
    a, e = vol(df.rv_adam), vol(df.rv_exact)
    print(f"EQ-Cov annual realized vol: Adam {a:.3f}, exact {e:.3f}, change {e - a:+.3f} points; "
          f"median forecast gap {df.log_forecast_gap.median():.2e}, min {df.log_forecast_gap.min():.2e}")
