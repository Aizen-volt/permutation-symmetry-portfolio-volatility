"""Interpretation of EQ-Cov: correlation mixing weights and volatility adjustments over the test period (h=5, seed 0).

For every test origin the encoder is run on the whole universe (54 assets); we record the mixing weights pi of the
correlation forecast (rolling 22d, rolling 66d, EWMA, learned Gram matrix) and the average log-volatility
adjustment delta by asset class.
"""
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
from pvol.models import F_PAIR  # noqa: E402
from pvol.train import train_model  # noqa: E402


@torch.no_grad()
def diagnostics(model, inp):
    h = model.emb(inp["x_asset"])
    for layer in model.layers:
        h = layer(h, inp["pair"])
    delta = model.vol(h).squeeze(-1)
    pi = model.mix(h.mean(1)).softmax(-1)
    a = torch.nn.functional.normalize(model.a(h), dim=-1)
    R = torch.einsum("bijk,bk->bij", inp["pair"], pi[:, :F_PAIR]) + pi[:, F_PAIR, None, None] * (a @ a.transpose(1, 2))
    return pi, delta, R


if __name__ == "__main__":
    h, seed = 5, 0
    p = load_panel()
    panel = PanelGPU(p)
    hp = load_tuned_hp()["EQCov"]
    classes = p["asset_class"]
    M = panel.M
    class_masks = {c: torch.as_tensor(classes == c, device=panel.device) for c in sorted(set(classes))}
    rows = []
    for Y in TEST_YEARS:
        split = make_split(panel.dates, panel.first_valid, Y, h)
        model, _ = train_model("EQCov", panel, split, h, seed, hp=hp)
        model.eval()
        t_idx = origins_in_year(panel.dates, Y, h)
        t = torch.as_tensor(t_idx, device=panel.device)
        S = torch.arange(M, device=panel.device).repeat(len(t_idx), 1)
        w = torch.full((len(t_idx), M), 1.0 / M, device=panel.device)
        inp, _ = panel.batch(t, S, w)
        pi, delta, R = diagnostics(model, inp)
        iu = torch.triu_indices(M, M, 1)
        mean_corr_hat = R[:, iu[0], iu[1]].mean(1)
        mean_corr_ewma = inp["pair"][:, iu[0], iu[1], 2].mean(1)
        for k, tt in enumerate(t_idx):
            row = dict(date=panel.dates[tt], pi_corr22=pi[k, 0].item(), pi_corr66=pi[k, 1].item(),
                       pi_ewma=pi[k, 2].item(), pi_gram=pi[k, 3].item(),
                       mean_corr_hat=mean_corr_hat[k].item(), mean_corr_ewma=mean_corr_ewma[k].item())
            for c, mask in class_masks.items():
                row[f"delta_{c}"] = delta[k, mask].mean().item()
            rows.append(row)
        print(Y, "mean pi:", pi.mean(0).cpu().numpy().round(3), flush=True)
    out = RESULTS / "main" / "tables" / "eqcov_diagnostics_h5.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("written", out)
