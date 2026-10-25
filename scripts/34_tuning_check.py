"""Check of the tuning choice: validation loss of the hyper-parameters in use (tuned on the ETF universe at h = 5)
relative to the best configuration of the same grid re-tuned at another horizon or on the stocks (03_tune.py --tag).

Writes results/tuning/tuning_check.csv. Usage: python scripts/34_tuning_check.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHECKS = [("etf", "results", "h1"), ("etf", "results", "h22"), ("stocks", "results_stocks", "h5"), ("stocks", "results_stocks", "h1")]

if __name__ == "__main__":
    rows = []
    for universe, res, tag in CHECKS:
        used = json.loads((ROOT / res / "tuning/best_hp.json").read_text())
        log = [json.loads(s) for s in (ROOT / res / f"tuning/tuning_log_{tag}.jsonl").read_text().splitlines() if s]
        for m, hp in used.items():
            vals = [r["best_val"] for r in log if r["model"] == m]
            v_used = next(r["best_val"] for r in log if r["model"] == m and r["hp"] == hp)
            rows.append(dict(universe=universe, h=int(tag[1:]), model=m, same=v_used == min(vals),
                             gap_pct=100 * (v_used / min(vals) - 1), spread_pct=100 * (max(vals) / min(vals) - 1)))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/tuning/tuning_check.csv", index=False)
    print(df.groupby(["universe", "h"]).agg(same=("same", "sum"), max_gap=("gap_pct", "max")).to_string())
