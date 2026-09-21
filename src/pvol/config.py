import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# PVOL_UNIVERSE selects the asset universe: "etf" (main study, default) or "stocks" (robustness study)
UNIVERSE_NAME = os.environ.get("PVOL_UNIVERSE", "etf")
_SUFFIX = "" if UNIVERSE_NAME == "etf" else f"_{UNIVERSE_NAME}"
RAW = ROOT / "data" / f"raw{_SUFFIX}"
PROCESSED = ROOT / "data" / f"processed{_SUFFIX}"
RESULTS = ROOT / f"results{_SUFFIX}"

SAMPLE_START = "2007-12-19"  # first day on which all 54 ETFs trade (EMB inception)
SAMPLE_END = "2026-09-18"

HAR_WINDOWS = (1, 5, 22, 66)  # day, week, month, quarter
CORR_WINDOWS = (22, 66)
EWMA_LAMBDA = 0.94  # RiskMetrics (1996) daily decay
MAX_LOOKBACK = max(HAR_WINDOWS)

HORIZONS = (1, 5, 22)
TEST_YEARS = tuple(range(2020, 2027))
VAR_FLOOR = 1e-8  # floor for daily variance proxies before taking logs (vol of 0.01%/day)
