"""Reproduce the full comparison: return direction vs volatility forecasting.

Runs both tasks under identical, leakage-free, non-overlapping walk-forward
evaluation and prints the two result tables. This is the reproducible backbone
of the rebuilt project's results.

Usage:
    KERAS_BACKEND=torch python run_experiment.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from src.data import load_prices
from src.features import RV_WINDOWS, build_dataset, build_volatility_dataset
from src.evaluation import (
    arima_forward_vol,
    baseline_predictions,
    garch_forward_vol,
    regression_metrics,
    summarize,
    walk_forward_predict,
    walk_forward_predict_seq,
)
from src.models import make_gbr, make_har, make_lgbm, make_lstm, make_mlp


def _row(name: str, m: dict) -> str:
    """Directional row (for the return task)."""
    return (f"{name:<22} {m['dir_acc']:>8.3f} {m['roc_auc']:>8.3f} "
            f"{m['rmse']:>9.5f} {m['r2']:>9.4f}")


def _reg_row(name: str, preds) -> str:
    """Regression row (for the volatility task; direction is meaningless there)."""
    m = regression_metrics(preds["y_true"].to_numpy(), preds["y_pred"].to_numpy())
    return f"{name:<22} {m['r2']:>9.4f} {m['rmse']:>9.5f} {m['mae']:>9.5f}"


def return_task(prices) -> None:
    X, y = build_dataset(prices, non_overlapping=True)
    print(f"\n== TASK 1: 1-week return DIRECTION ({X.shape[0]} weekly samples) ==")
    print(f"{'predictor':<22} {'dir_acc':>8} {'roc_auc':>8} {'rmse':>9} {'r2':>9}")
    for kind in ("always_up", "persistence"):
        print(_row(f"baseline:{kind}", summarize(baseline_predictions(X, y, kind=kind))))
    for name, fac in (("GBR", make_gbr), ("MLP (ReLU)", make_mlp)):
        print(_row(name, summarize(walk_forward_predict(fac, X, y, scale=True))))
    print("  -> no model beats the ~0.54 up-week base rate: direction is not predictable.")


def volatility_task(prices) -> None:
    X, y = build_volatility_dataset(prices, non_overlapping=True)
    har_cols = [f"BV_RV{w}" for w in RV_WINDOWS]
    close = prices["BV"]["Close"]
    print(f"\n== TASK 2: 1-week realized VOLATILITY ({X.shape[0]} weekly samples, log-target) ==")
    print(f"{'predictor':<22} {'r2':>9} {'rmse':>9} {'mae':>9}")
    # naive baselines
    for kind in ("train_mean", "persistence"):
        print(_reg_row(f"baseline:{kind}", baseline_predictions(X, y, kind=kind)))
    # time-series-native benchmarks
    print(_reg_row("HAR-RV (OLS)",
                   walk_forward_predict(make_har, X[har_cols], y, scale=False, log_target=True)))
    print(_reg_row("ARIMA(2,0,1)", arima_forward_vol(y)))
    print(_reg_row("GARCH(1,1)", garch_forward_vol(close, y)))
    # gradient boosters on indicator + HAR features
    for name, fac in (("GBR (log)", make_gbr), ("LightGBM (log)", make_lgbm)):
        print(_reg_row(name, walk_forward_predict(fac, X, y, scale=True, log_target=True)))
    # neural nets (underperform on this small dataset)
    print(_reg_row("MLP ReLU (log)", walk_forward_predict(make_mlp, X, y, scale=True, log_target=True)))
    print(_reg_row("LSTM (lookback=4)", walk_forward_predict_seq(make_lstm, X, y, lookback=4)))
    print("  -> GARCH (R2 ~0.27) leads, then ARIMA (~0.22) and the boosters (~0.15);")
    print("     all beat naive persistence (~0.09). Neural nets trail on so few samples.")


if __name__ == "__main__":
    prices = load_prices()
    return_task(prices)
    volatility_task(prices)
