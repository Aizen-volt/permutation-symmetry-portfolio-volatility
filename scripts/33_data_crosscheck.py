"""Cross-check of the Yahoo Finance prices against a second source (Nasdaq historical quotes, unadjusted OHLC).

Nasdaq serves about ten years of daily history, so the check covers the common days of roughly 2016-2026, i.e. all
test years. Compared per instrument:
  ret_corr      correlation of daily close-to-close log returns
  big_ret_diff  share of days with returns differing by more than 0.5% (adjusted Yahoo vs unadjusted Nasdaq returns
                differ by the dividend on ex-dividend days, typically well below 0.5%)
  sq_ratio      ratio of the sums of squared daily returns (the variance proxy), Yahoo over Nasdaq
  range_mad     median absolute difference of the daily log range log(high/low) (scale-free, unaffected by adjustment)
Usage: [PVOL_UNIVERSE=stocks] python scripts/33_data_crosscheck.py
Output: results*/descriptives/data_crosscheck.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pvol.config import RAW, RESULTS, SAMPLE_END, UNIVERSE_NAME  # noqa: E402

URL = "https://api.nasdaq.com/api/quote/{}/historical?assetclass={}&fromdate=2015-01-01&todate={}&limit=10000"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def nasdaq(ticker, assetclass):
    r = requests.get(URL.format(ticker.replace("-", "."), assetclass, SAMPLE_END), headers=HEADERS, timeout=30)
    rows = ((r.json().get("data") or {}).get("tradesTable") or {}).get("rows") or []
    if not rows:
        return None
    d = pd.DataFrame(rows)
    num = lambda s: pd.to_numeric(s.str.replace(r"[$,]", "", regex=True), errors="coerce")
    return pd.DataFrame({k: num(d[k]) for k in ("open", "high", "low", "close")}
                        ).set_index(pd.to_datetime(d.date, format="%m/%d/%Y")).sort_index()


if __name__ == "__main__":
    raw = pd.read_parquet(RAW / "ohlcv_adjusted.parquet")
    cls = "etf" if UNIVERSE_NAME == "etf" else "stocks"
    rows = []
    for tk in raw["Close"].columns:
        try:
            n = nasdaq(tk, cls)
        except (requests.RequestException, ValueError) as e:
            print(tk, "failed:", e, flush=True)
            n = None
        time.sleep(0.3)
        if n is None:
            rows.append(dict(ticker=tk, days=0))
            continue
        y = pd.DataFrame({k: raw[k.capitalize()][tk] for k in ("high", "low", "close")}).dropna()
        common = y.index.intersection(n.index)
        y, n = y.loc[common], n.loc[common]
        ry, rn = np.log(y.close).diff().iloc[1:], np.log(n.close).diff().iloc[1:]
        rows.append(dict(ticker=tk, days=len(ry), first=common.min().date(), ret_corr=np.corrcoef(ry, rn)[0, 1],
                         big_ret_diff=float((np.abs(ry - rn) > 0.005).mean()), sq_ratio=float((ry**2).sum() / (rn**2).sum()),
                         range_mad=float(np.median(np.abs(np.log(y.high / y.low) - np.log(n.high / n.low))))))
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    (RESULTS / "descriptives").mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "descriptives" / "data_crosscheck.csv", index=False)
    ok = out[out.days > 0]
    print(f"{len(ok)}/{len(out)} instruments matched; median days {ok.days.median():.0f}")
    print(ok[["ret_corr", "big_ret_diff", "sq_ratio", "range_mad"]].describe(percentiles=[0.05, 0.5, 0.95]).round(4))
