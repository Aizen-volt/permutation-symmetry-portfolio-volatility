"""Investment universe: US-listed ETFs covering several asset classes.

All ETFs trade on US exchanges, so daily closes are synchronous across
asset classes (no stale-price / non-synchronous close problem that arises
when mixing e.g. European and US local-market indices).
"""

from .config import UNIVERSE_NAME

ETF_UNIVERSE = {
    # --- US equity (broad) ---
    "SPY": "equity_us",
    "QQQ": "equity_us",
    "IWM": "equity_us",
    "DIA": "equity_us",
    "MDY": "equity_us",
    # --- US equity sectors (Select Sector SPDRs, 1998) ---
    "XLB": "equity_sector",
    "XLE": "equity_sector",
    "XLF": "equity_sector",
    "XLI": "equity_sector",
    "XLK": "equity_sector",
    "XLP": "equity_sector",
    "XLU": "equity_sector",
    "XLV": "equity_sector",
    "XLY": "equity_sector",
    # --- International equity ---
    "EFA": "equity_intl",
    "EEM": "equity_intl",
    "VGK": "equity_intl",
    "EWJ": "equity_intl",
    "EWG": "equity_intl",
    "EWU": "equity_intl",
    "EWC": "equity_intl",
    "EWA": "equity_intl",
    "EWZ": "equity_intl",
    "FXI": "equity_intl",
    "EWY": "equity_intl",
    "EWT": "equity_intl",
    "EWW": "equity_intl",
    "EWH": "equity_intl",
    # --- Fixed income ---
    "SHY": "bond",
    "IEF": "bond",
    "TLT": "bond",
    "AGG": "bond",
    "TIP": "bond",
    "LQD": "bond",
    "HYG": "bond",
    "EMB": "bond",
    "MBB": "bond",
    "BWX": "bond",
    # --- Real estate ---
    "IYR": "real_estate",
    "VNQ": "real_estate",
    "RWX": "real_estate",
    # --- Commodities ---
    "GLD": "commodity",
    "SLV": "commodity",
    "USO": "commodity",
    "UNG": "commodity",
    "DBC": "commodity",
    "DBA": "commodity",
    "DBB": "commodity",
    # --- Currencies ---
    "FXE": "fx",
    "FXY": "fx",
    "FXB": "fx",
    "FXA": "fx",
    "FXF": "fx",
    "UUP": "fx",
}

# Robustness universe: large US stocks (S&P 100-type names and other large caps listed before 2007),
# labelled by GICS sector. Names without a complete price history since the sample start are dropped at download.
# Selection is based on today's large caps, so the sample is subject to survivorship bias (see paper).
_STOCKS_BY_SECTOR = {
    "tech": "AAPL MSFT NVDA ORCL CSCO ADBE CRM INTC IBM QCOM TXN AMD ACN INTU AMAT ADI MU ADP HPQ APH",
    "comm": "GOOGL CMCSA DIS T VZ NFLX TMUS",
    "cons_disc": "AMZN HD MCD NKE LOW SBUX TGT BKNG F",
    "cons_staples": "WMT PG KO PEP COST MO CL MDLZ KMB GIS",
    "energy": "XOM CVX COP EOG SLB OXY VLO",
    "financials": "JPM BAC WFC GS MS C AXP BLK SCHW USB PNC BK COF MET AIG CB MMC PGR TRV AFL ALL SPGI MA",
    "health": "UNH JNJ LLY PFE MRK ABT TMO DHR AMGN BMY GILD MDT CVS CI SYK ISRG REGN VRTX BDX HUM",
    "industrials": "GE CAT HON UNP UPS BA DE LMT GD MMM EMR FDX CSX NSC ITW NOC WM",
    "materials": "APD SHW ECL NEM",
    "real_estate": "AMT SPG PSA",
    "utilities": "NEE DUK SO D",
}
STOCK_UNIVERSE = {t: sec for sec, names in _STOCKS_BY_SECTOR.items() for t in names.split()}

UNIVERSE = STOCK_UNIVERSE if UNIVERSE_NAME in ("stocks", "stocks54") else ETF_UNIVERSE
SUBSAMPLE = 54 if UNIVERSE_NAME == "stocks54" else None  # experiment R5: stratified subsample of the complete stocks
ASSET_CLASSES = sorted(set(UNIVERSE.values()))
