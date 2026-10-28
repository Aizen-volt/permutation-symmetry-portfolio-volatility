"""Fourth series S1: gradient-boosting benchmark (GBM) on the features of the linear set benchmark.

For every test year: 200,000 (origin, portfolio) pairs are drawn from the training period exactly as for the networks
(N ~ U{2..20}, Dirichlet weights with alpha in {0.3, 1, 3}); HistGradientBoostingRegressor with the Poisson loss (log link,
minimiser = conditional mean, so QLIKE-consistent) is fitted to y / EWMA forecast; internal early stopping on 10% of the
sample. Fixed settings, no tuning. The forecast is the EWMA forecast times the prediction.

Usage: [PVOL_UNIVERSE=stocks] python scripts/27_gbm.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from pvol.config import HORIZONS, RESULTS, TEST_YEARS  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.dataset import PanelGPU, make_split, origins_in_year  # noqa: E402
from pvol.models import LinearSet  # noqa: E402
from pvol.portfolios import load  # noqa: E402

N_TRAIN, CHUNK = 200_000, 4096


@torch.no_grad()
def training_sample(panel, t_range, h, seed):
    gen = torch.Generator(device=panel.device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    F, Y = [], []
    for _ in range(N_TRAIN // CHUNK + 1):
        N = int(rng.integers(2, 21))
        t, S, w = panel.sample(CHUNK, N, t_range, gen, alpha=float(rng.choice([0.3, 1.0, 3.0])))
        inp, y = panel.batch(t, S, w, h=h)
        F.append(LinearSet().features(inp).cpu().numpy())
        Y.append((y.clamp_min(1e-12) / torch.exp(inp["base"])).cpu().numpy())
    return np.concatenate(F)[:N_TRAIN], np.concatenate(Y)[:N_TRAIN]


@torch.no_grad()
def predict(panel, gbm, portfolios, t_idx):
    out = np.full((len(t_idx), len(portfolios)), np.nan, dtype=np.float32)
    by_n = {}
    for j, p in enumerate(portfolios):
        by_n.setdefault(len(p["ids"]), []).append(j)
    t_all = torch.as_tensor(np.asarray(t_idx), device=panel.device)
    for N, cols in by_n.items():
        S = torch.as_tensor([portfolios[j]["ids"] for j in cols], device=panel.device)
        W = torch.as_tensor([portfolios[j]["weights"] for j in cols], device=panel.device, dtype=torch.float32)
        tt = t_all.repeat_interleave(len(cols))
        SS, WW = S.repeat(len(t_idx), 1), W.repeat(len(t_idx), 1)
        step = max(1, CHUNK * 100 // (N * N))
        res = []
        for a in range(0, len(tt), step):
            inp, _ = panel.batch(tt[a : a + step], SS[a : a + step], WW[a : a + step])
            ratio = gbm.predict(LinearSet().features(inp).cpu().numpy())
            res.append(inp["base"].cpu().numpy() + np.log(np.maximum(ratio, 1e-6)))
        out[:, cols] = np.concatenate(res).reshape(len(t_idx), len(cols))
    return out


if __name__ == "__main__":
    p = load_panel()
    panel = PanelGPU(p)
    ports = load(RESULTS / "main" / "portfolios.json")
    for h in HORIZONS:
        parts = []
        for y in TEST_YEARS:
            t0 = time.time()
            split = make_split(panel.dates, panel.first_valid, y, h)
            panel.fit_scalers(split.train, torch.Generator(device=panel.device).manual_seed(0))
            Xtr, Ytr = training_sample(panel, split.train, h, seed=y)
            gbm = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=500, max_leaf_nodes=31,
                                                min_samples_leaf=200, l2_regularization=1.0, early_stopping=True,
                                                validation_fraction=0.1, n_iter_no_change=20, random_state=0)
            gbm.fit(Xtr, Ytr)
            parts.append(predict(panel, gbm, ports, origins_in_year(panel.dates, y, h)))
            print(f"h={h} year={y}: {gbm.n_iter_} iterations, {time.time() - t0:.0f}s", flush=True)
        np.save(RESULTS / "main" / f"h{h}" / "preds" / "GBM.npy", np.concatenate(parts).astype(np.float32))
    print("done")
