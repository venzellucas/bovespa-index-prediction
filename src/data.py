"""Price data loading for the Bovespa index prediction project.

Downloads daily OHLCV for the Ibovespa and two explanatory markets (the S&P 500
and the USD/BRL rate) via yfinance, with a local parquet cache so repeated runs
and tests are reproducible and don't hammer the API.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Ticker -> short prefix used throughout the feature pipeline.
TICKERS: dict[str, str] = {
    "^BVSP": "BV",   # Ibovespa (target market)
    "^GSPC": "SP",   # S&P 500
    "BRL=X": "US",   # USD/BRL exchange rate
}

DEFAULT_START = "2006-01-01"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def _download(ticker: str, start: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        ticker, start=start, interval="1d",
        auto_adjust=True, progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker!r} (check network/TLS).")
    # yfinance returns a MultiIndex (field, ticker); flatten to single fields.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def load_prices(
    start: str = DEFAULT_START,
    *,
    use_cache: bool = True,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Return {prefix: OHLCV DataFrame} for every ticker in TICKERS.

    Results are cached to parquet under data/cache/. Pass refresh=True to
    re-download and overwrite the cache.
    """
    out: dict[str, pd.DataFrame] = {}
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for ticker, prefix in TICKERS.items():
        cache_file = CACHE_DIR / f"{prefix}_{start}.parquet"
        if use_cache and not refresh and cache_file.exists():
            out[prefix] = pd.read_parquet(cache_file)
            continue
        df = _download(ticker, start)
        if use_cache:
            df.to_parquet(cache_file)
        out[prefix] = df
    return out


if __name__ == "__main__":
    prices = load_prices()
    for prefix, df in prices.items():
        print(f"{prefix}: {len(df)} rows, {df.index.min().date()} -> {df.index.max().date()}")
