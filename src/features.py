"""Feature engineering and target construction.

Technical indicators are *causal* (each uses only a trailing window), so they
can be computed on the full price series without leaking future information.
Scaling is deliberately NOT done here: it must be fit on training data only,
inside each cross-validation fold (see src/evaluation.py). Doing it globally
was a leakage bug in the original 2021 notebook.
"""

from __future__ import annotations

import pandas as pd
from ta import momentum, trend

# Indicator columns produced by add_indicators(), in a stable order.
FEATURE_NAMES: list[str] = ["MACD", "RSI", "ADX", "Aroon", "CCI", "DPO", "MI", "TRIX"]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append the eight technical indicators to an OHLCV frame.

    Signatures follow the current `ta` API: AroonIndicator and MassIndex take
    (high, low) here, whereas the 2021 notebook passed `close` (which the new
    library rejects, and which fed Close as the high to MassIndex by mistake).
    """
    out = df.copy()
    high, low, close = out["High"], out["Low"], out["Close"]

    out["MACD"] = trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9).macd_signal()
    out["RSI"] = momentum.RSIIndicator(close=close, window=14).rsi()
    out["ADX"] = trend.ADXIndicator(high=high, low=low, close=close, window=14).adx()
    out["Aroon"] = trend.AroonIndicator(high=high, low=low, window=25).aroon_indicator()
    out["CCI"] = trend.CCIIndicator(high=high, low=low, close=close, window=20, constant=0.015).cci()
    out["DPO"] = trend.DPOIndicator(close=close, window=20).dpo()
    out["MI"] = trend.MassIndex(high=high, low=low, window_fast=9, window_slow=25).mass_index()
    out["TRIX"] = trend.TRIXIndicator(close=close, window=15).trix()
    return out


def forward_return(close: pd.Series, horizon: int = 5) -> pd.Series:
    """Forward return over `horizon` trading days: Close[t+h]/Close[t] - 1.

    Uses shift(-h) on the trading-day index (5 days ~= one week) rather than a
    calendar-week offset + ffill, which is cleaner and avoids the weekend/holiday
    artifacts of the original. The last `horizon` rows are NaN (no future yet).
    """
    return close.shift(-horizon) / close - 1.0


def build_dataset(
    prices: dict[str, pd.DataFrame],
    *,
    horizon: int = 5,
    target_market: str = "BV",
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the modelling table from {prefix: OHLCV}.

    Returns (X, y) where X holds prefixed indicator columns for every market and
    y is the forward return of the target market. Rows with any NaN (indicator
    warm-up at the start, target horizon at the end) are dropped jointly so X and
    y stay aligned.
    """
    frames = []
    for prefix, df in prices.items():
        feats = add_indicators(df)[FEATURE_NAMES]
        feats = feats.add_prefix(f"{prefix}_")
        frames.append(feats)

    X = pd.concat(frames, axis=1).sort_index()
    y = forward_return(prices[target_market]["Close"], horizon=horizon).rename("target")

    combined = X.join(y, how="inner").dropna()
    return combined[X.columns], combined["target"]


if __name__ == "__main__":
    from src.data import load_prices

    X, y = build_dataset(load_prices())
    print(f"X: {X.shape}, features: {list(X.columns)}")
    print(f"y: {y.shape}, range {y.index.min().date()} -> {y.index.max().date()}")
    print(f"target mean={y.mean():.4f} std={y.std():.4f}")
