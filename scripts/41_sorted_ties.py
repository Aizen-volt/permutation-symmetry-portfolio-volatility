"""Permutation consistency of MLP-sorted on equally weighted portfolios: sorting by weight leaves the
order of tied constituents to the input order, so MLP-sorted is exactly invariant only for distinct weights.

MLP-sorted is retrained in the main design (h = 5, seeds 0-2, every test year, tuned hyper-parameters) and applied to
100 equally weighted ten-asset portfolios of the evaluation set (group N10-EW), each listed in five random orders.
Reports, as in Table perm_main, the median and 99th percentile over portfolios and origins of the range of the
log-variance forecast across the orderings, averaged over seeds. Writes results*/main/tables/sorted_ties.csv.

Usage: [PVOL_UNIVERSE=stocks] python scripts/41_sorted_ties.py [MLP-tsort]
(the sixth series V1 passes MLP-tsort, sorted with ties broken by asset index; writes sorted_ties_MLP-tsort.csv)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from pvol.config import RESULTS, TEST_YEARS  # noqa: E402
from pvol.data import load_panel  # noqa: E402
from pvol.dataset import make_split, origins_in_year  # noqa: E402
from pvol.experiment import Experiment, load_tuned_hp  # noqa: E402
from pvol.train import predict_portfolios, train_model  # noqa: E402

H, SEEDS, N_PORTS, N_PERM = 5, (0, 1, 2), 100, 5
NAME = sys.argv[1] if len(sys.argv) > 1 else "MLP-sorted"

ex = Experiment("main", load_panel())
rng = np.random.default_rng(777)
base = [p for p in ex.portfolios if p["group"] == "N10-EW"][:N_PORTS]
perm = []
for k in range(N_PERM):
    for p in base:
        o = rng.permutation(len(p["ids"]))
        perm.append(dict(ids=[p["ids"][i] for i in o], weights=[p["weights"][i] for i in o]))
hp = load_tuned_hp()["MLP-sorted"]
rows = []
for seed in SEEDS:
    parts = []
    for y in TEST_YEARS:
        split = make_split(ex.panel.dates, ex.panel.first_valid, y, H)
        model, _ = train_model(NAME, ex.panel, split, H, seed, hp=hp)
        parts.append(predict_portfolios(model, NAME, ex.panel, perm, origins_in_year(ex.panel.dates, y, H)))
        del model
        torch.cuda.empty_cache()
    s = np.concatenate(parts).reshape(-1, N_PERM, N_PORTS)  # (T, perm, portfolio): log variance
    rng_ = s.max(1) - s.min(1)
    rows.append(dict(seed=seed, median=np.median(rng_), p99=np.quantile(rng_, 0.99)))
    print(rows[-1], flush=True)
out = pd.DataFrame(rows)
out.loc["mean"] = out.mean()
out.to_csv(RESULTS / "main" / "tables" / ("sorted_ties.csv" if NAME == "MLP-sorted" else f"sorted_ties_{NAME}.csv"))
print(out.round(4))
