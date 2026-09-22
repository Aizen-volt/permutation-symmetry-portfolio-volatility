"""Descriptive statistics of the ETF universe (paper Table 1) and the list of tickers by asset class."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvol.config import RESULTS  # noqa: E402
from pvol.data import load_panel  # noqa: E402

LABEL = {"equity_us": "US equity", "equity_sector": "US equity sectors", "equity_intl": "International equity",
         "bond": "Fixed income", "real_estate": "Real estate", "commodity": "Commodities", "fx": "Currencies",
         "tech": "Information technology", "comm": "Communication services", "cons_disc": "Consumer discretionary",
         "cons_staples": "Consumer staples", "energy": "Energy", "financials": "Financials", "health": "Health care",
         "industrials": "Industrials", "materials": "Materials", "utilities": "Utilities"}

if __name__ == "__main__":
    p = load_panel()
    R = p["R"][1:].astype(np.float64)
    cls = p["asset_class"]
    tick = p["tickers"]
    C = np.corrcoef(R.T)
    rows = []
    for c in [k for k in LABEL if k in set(cls)]:
        ids = np.where(cls == c)[0]
        r = R[:, ids]
        vol = r.std(0) * np.sqrt(252)
        within = C[np.ix_(ids, ids)][np.triu_indices(len(ids), 1)] if len(ids) > 1 else np.array([np.nan])
        other = C[np.ix_(ids, np.setdiff1d(np.arange(len(tick)), ids))]
        rows.append(dict(asset_class=LABEL[c], n=len(ids), tickers=", ".join(tick[ids]),
                         ann_ret=np.mean(r.mean(0) * 252), ann_vol=np.mean(vol), vol_min=vol.min(), vol_max=vol.max(),
                         skew=np.mean(pd.DataFrame(r).skew()), kurt=np.mean(pd.DataFrame(r).kurt()),
                         corr_within=np.nanmean(within), corr_other=other.mean()))
    df = pd.DataFrame(rows)
    out = RESULTS / "descriptives"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "universe_stats.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.drop(columns="tickers").round(3).to_string(index=False))
    print(df[["asset_class", "tickers"]].to_string(index=False))
