"""Computational cost of the networks: parameters, training time per (seed, test year) run at h=5 (from the training
logs) and forecast time per 1,000 portfolios on the GPU for N = 10, 20 and the whole universe.

Writes results/main/tables/compute_cost.csv for the current universe (PVOL_UNIVERSE) and, when both CSVs exist,
paper/tables/compute_cost.tex.
"""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from pvol.config import RESULTS, UNIVERSE_NAME  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.dataset import PanelGPU, make_split  # noqa: E402
from pvol.evaluate import NEURAL  # noqa: E402
from pvol.experiment import load_tuned_hp  # noqa: E402
from pvol.models import SEQ_MODELS, SLOT_MODELS  # noqa: E402
from pvol.train import TrainCfg, build_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / ("train_main.log" if UNIVERSE_NAME == "etf" else "train_stocks.log")
B, REPS = 1000, 20
LINE = re.compile(r"\[main h=5\] (\S+) seed=\d+ year=\d+: val \S+ epochs (\d+) (\d+)s")


def train_times():
    rows = [LINE.match(s) for s in LOG.read_text(encoding="utf-8").splitlines()]
    d = pd.DataFrame([(m[1], int(m[2]), int(m[3])) for m in rows if m], columns=["model", "epochs", "sec"])
    return d.groupby("model").agg(epochs=("epochs", "mean"), train_sec=("sec", "mean"), runs=("sec", "size"))


@torch.no_grad()
def forecast_ms(model, name, panel, split, N, gen):
    """Milliseconds per 1,000 portfolios. Large portfolios are scored in chunks, as in train.predict_portfolios, because
    the pairwise tensors grow with N^2 (a single batch of 1,000 portfolios of 122 stocks would spill out of GPU memory)."""
    b = min(B, max(1, 8192 * 100 // (N * N))) if N > 10 else B
    t, S, w = panel.sample(b, N, split.val, gen)
    inp, _ = panel.batch(t, S, w, gru_len=TrainCfg().gru_len if name in SEQ_MODELS else 0)
    for _ in range(3):  # warm-up
        model(inp)
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        model(inp)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return 1000 * float(np.median(ts)) * B / b


if __name__ == "__main__":
    panel = PanelGPU(load_panel())
    split = make_split(panel.dates, panel.first_valid, 2026, 5)
    gen = torch.Generator(device=panel.device).manual_seed(0)
    panel.fit_scalers(split.train, torch.Generator(device=panel.device).manual_seed(0))
    hp = load_tuned_hp()
    tt = train_times()
    rows = []
    for name in NEURAL:
        model = build_model(name, hp.get(name)).to(panel.device).eval()
        r = dict(model=name, params=sum(p.numel() for p in model.parameters()))
        for N in (10, 20, panel.M):
            key = "ms_all" if N == panel.M else f"ms_{N}"
            r[key] = np.nan if name in SLOT_MODELS and N > 20 else forecast_ms(model, name, panel, split, N, gen)
        if name in tt.index:
            r.update(tt.loc[name].to_dict())
        rows.append(r)
        print(r, flush=True)
    out = RESULTS / "main" / "tables" / "compute_cost.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("written", out)
