"""Download daily OHLCV (dividend/split adjusted) for the ETF universe from Yahoo Finance."""
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pvol.config import RAW  # noqa: E402
from pvol.universe import UNIVERSE  # noqa: E402

RAW.mkdir(parents=True, exist_ok=True)


def main():
    tickers = list(UNIVERSE)
    df = yf.download(
        tickers,
        start="2000-01-01",
        end="2026-09-19",
        auto_adjust=True,
        actions=False,
        group_by="column",
        progress=False,
        threads=True,
    )
    df.to_parquet(RAW / "ohlcv_adjusted.parquet")
    close = df["Close"]
    summary = pd.DataFrame(
        {
            "asset_class": pd.Series(UNIVERSE),
            "first": close.apply(lambda s: s.first_valid_index()),
            "last": close.apply(lambda s: s.last_valid_index()),
            "n_obs": close.notna().sum(),
        }
    ).sort_values("first")
    summary.to_csv(RAW / "coverage.csv")
    print(summary.to_string())
    print("shape", df.shape)


if __name__ == "__main__":
    main()
