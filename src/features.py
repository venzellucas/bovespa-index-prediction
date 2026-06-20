"""Feature engineering and target construction.

Technical indicators are *causal* (each uses only a trailing window), so they
can be computed on the full price series without leaking future information.
Scaling is deliberately NOT done here: it must be fit on training data only,
inside each cross-validation fold (see src/evaluation.py). Doing it globally
was a leakage bug in the original 2021 notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta import momentum, trend

# Trailing realized-volatility windows (HAR-RV style): week, month, quarter.
RV_WINDOWS: tuple[int, ...] = (5, 22, 66)

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


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def trailing_rv(close: pd.Series, window: int) -> pd.Series:
    """Realized volatility over the trailing `window` days (causal feature)."""
    return log_returns(close).rolling(window).std()


def forward_rv(close: pd.Series, horizon: int = 5) -> pd.Series:
    """Realized volatility over the *next* `horizon` days: std of r[t+1..t+h].

    This is the volatility target. rolling(h).std().shift(-h) places the
    forward-window std at time t; the last `horizon` rows are NaN.
    """
    return log_returns(close).rolling(horizon).std().shift(-horizon)


def add_rv_features(close: pd.Series, prefix: str) -> pd.DataFrame:
    """HAR-RV style trailing realized-vol features for one market."""
    cols = {f"{prefix}_RV{w}": trailing_rv(close, w) for w in RV_WINDOWS}
    return pd.DataFrame(cols)


def build_volatility_dataset(
    prices: dict[str, pd.DataFrame],
    *,
    horizon: int = 5,
    target_market: str = "BV",
    non_overlapping: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Modelling table for volatility forecasting.

    X = technical indicators (per market) + HAR-RV trailing realized-vol features
    (per market). y = forward realized volatility of the target market. Unlike
    the return target, volatility is genuinely autocorrelated, so the persistence
    baseline here is a strong and *legitimate* benchmark (the HAR-RV reference),
    not an overlap artifact.
    """
    frames = []
    for prefix, df in prices.items():
        feats = add_indicators(df)[FEATURE_NAMES].add_prefix(f"{prefix}_")
        frames.append(feats)
        frames.append(add_rv_features(df["Close"], prefix))

    X = pd.concat(frames, axis=1).sort_index()
    y = forward_rv(prices[target_market]["Close"], horizon=horizon).rename("target")

    combined = X.join(y, how="inner").dropna()
    if non_overlapping:
        combined = combined.iloc[::horizon]
    return combined[X.columns], combined["target"]


def build_dataset(
    prices: dict[str, pd.DataFrame],
    *,
    horizon: int = 5,
    target_market: str = "BV",
    non_overlapping: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the modelling table from {prefix: OHLCV}.

    Returns (X, y) where X holds prefixed indicator columns for every market and
    y is the forward return of the target market. Rows with any NaN (indicator
    warm-up at the start, target horizon at the end) are dropped jointly so X and
    y stay aligned.

    With non_overlapping=True the rows are subsampled every `horizon` days so
    consecutive targets cover disjoint windows. This removes the ~(horizon-1)/horizon
    autocorrelation that an h-day forward return sampled daily would otherwise have,
    and is the honest setup for evaluation (a daily sample makes naive persistence
    look skilful purely from window overlap).
    """
    frames = []
    for prefix, df in prices.items():
        feats = add_indicators(df)[FEATURE_NAMES]
        feats = feats.add_prefix(f"{prefix}_")
        frames.append(feats)

    X = pd.concat(frames, axis=1).sort_index()
    y = forward_return(prices[target_market]["Close"], horizon=horizon).rename("target")

    combined = X.join(y, how="inner").dropna()
    if non_overlapping:
        combined = combined.iloc[::horizon]
    return combined[X.columns], combined["target"]


if __name__ == "__main__":
    from src.data import load_prices

    X, y = build_dataset(load_prices())
    print(f"X: {X.shape}, features: {list(X.columns)}")
    print(f"y: {y.shape}, range {y.index.min().date()} -> {y.index.max().date()}")
    print(f"target mean={y.mean():.4f} std={y.std():.4f}")
