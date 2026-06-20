"""Walk-forward evaluation, leakage-free scaling, metrics and naive baselines.

This module fixes three weaknesses of the original notebook:

1. Scaling leakage -- StandardScaler was fit on the full series (including test
   years) before splitting. Here the scaler is fit on the training fold only and
   applied to validation/test, inside each fold.
2. No baseline -- return-prediction metrics are meaningless without a reference.
   We provide the naive baselines a model must beat: predict-zero, predict the
   training mean, persistence (last return), and always-up (directional).
3. Validation methodology is kept as the original's expanding year-by-year
   walk-forward, made explicit and reusable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# A model factory returns a fresh, unfitted estimator with .fit/.predict.
ModelFactory = Callable[[], object]


def walk_forward_splits(
    index: pd.DatetimeIndex,
    *,
    test_years: range = range(2016, 2026),
    train_window: int = 7,
) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (test_year, train_pos, valid_pos, test_pos) integer-position arrays.

    Mirrors the original scheme: for test year i, train on years [i-train_window-1,
    i-2], validate on i-1, test on i. Positions are relative to `index`.
    """
    years = index.year.to_numpy()
    pos = np.arange(len(index))
    for i in test_years:
        train = pos[(years >= i - train_window - 1) & (years < i - 1)]
        valid = pos[years == i - 1]
        test = pos[years == i]
        if len(train) == 0 or len(test) == 0:
            continue
        yield i, train, valid, test


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Directional metrics: did we predict the sign of the forward return?

    'up' is defined as a strictly positive forward return. ROC-AUC uses the raw
    predicted return as the score; the rest threshold the prediction at zero.
    """
    up_true = (y_true > 0).astype(int)
    up_pred = (y_pred > 0).astype(int)
    out = {
        "dir_acc": accuracy_score(up_true, up_pred),
        "precision": precision_score(up_true, up_pred, zero_division=0),
        "recall": recall_score(up_true, up_pred, zero_division=0),
        "f1": f1_score(up_true, up_pred, zero_division=0),
    }
    # ROC-AUC is undefined if only one class is present in the truth.
    out["roc_auc"] = roc_auc_score(up_true, y_pred) if len(np.unique(up_true)) > 1 else float("nan")
    return out


def walk_forward_predict(
    make_model: ModelFactory,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    scale: bool = True,
    log_target: bool = False,
    test_years: range = range(2016, 2026),
) -> pd.DataFrame:
    """Run a model across walk-forward folds, scaling per fold (no leakage).

    With log_target=True the model is fit on log(y) and predictions are
    exponentiated back -- the standard transform for strictly-positive,
    right-skewed targets like realized volatility. Returns a tidy frame indexed
    by date with columns [y_true, y_pred, year].
    """
    rows = []
    Xv, yv = X.to_numpy(dtype=float), y.to_numpy(dtype=float)
    for year, tr, _va, te in walk_forward_splits(X.index, test_years=test_years):
        X_tr, X_te = Xv[tr], Xv[te]
        if scale:
            scaler = StandardScaler().fit(X_tr)  # fit on TRAIN ONLY
            X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
        y_tr = np.log(yv[tr]) if log_target else yv[tr]
        model = make_model()
        model.fit(X_tr, y_tr)
        pred = np.asarray(model.predict(X_te)).ravel()
        if log_target:
            pred = np.exp(pred)
        rows.append(pd.DataFrame(
            {"y_true": yv[te], "y_pred": pred, "year": year},
            index=X.index[te],
        ))
    return pd.concat(rows)


def walk_forward_predict_seq(
    make_model: ModelFactory,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    lookback: int = 8,
    scale: bool = True,
    log_target: bool = False,
    test_years: range = range(2016, 2026),
) -> pd.DataFrame:
    """Walk-forward for sequence models (LSTM): each sample is the prior
    `lookback` rows of features. Scaling is fit on the training fold only; test
    sequences may reach back into earlier rows (past data only, no leakage).
    """
    Xv, yv = X.to_numpy(dtype=float), y.to_numpy(dtype=float)
    idx = X.index
    rows = []
    for year, tr, _va, te in walk_forward_splits(idx, test_years=test_years):
        Xs = StandardScaler().fit(Xv[tr]).transform(Xv) if scale else Xv

        def build(positions):
            keep = [p for p in positions if p - lookback + 1 >= 0]
            if not keep:
                return None, None, None
            seq = np.stack([Xs[p - lookback + 1: p + 1] for p in keep])
            return seq, yv[keep], keep

        X_tr, y_tr, _ = build(tr)
        X_te, y_te, pos_te = build(te)
        if X_tr is None or X_te is None:
            continue
        model = make_model()
        model.fit(X_tr, np.log(y_tr) if log_target else y_tr)
        pred = np.asarray(model.predict(X_te)).ravel()
        if log_target:
            pred = np.exp(pred)
        rows.append(pd.DataFrame(
            {"y_true": y_te, "y_pred": pred, "year": year}, index=idx[pos_te]))
    return pd.concat(rows)


# --- Naive baselines -------------------------------------------------------

def baseline_predictions(X: pd.DataFrame, y: pd.Series, *, kind: str,
                         test_years: range = range(2016, 2026)) -> pd.DataFrame:
    """Predictions for a naive baseline across the same walk-forward folds.

    kind: 'zero'        -> always predict 0 return
          'train_mean'  -> predict the training-fold mean return
          'persistence' -> predict the previous day's realized forward return
          'always_up'   -> predict a small positive constant (always 'up')
    """
    yv = y.to_numpy(dtype=float)
    rows = []
    for year, tr, _va, te in walk_forward_splits(X.index, test_years=test_years):
        if kind == "zero":
            pred = np.zeros(len(te))
        elif kind == "train_mean":
            pred = np.full(len(te), yv[tr].mean())
        elif kind == "always_up":
            pred = np.full(len(te), 1e-6)
        elif kind == "persistence":
            prev = np.concatenate(([yv[tr][-1]], yv[te][:-1]))
            pred = prev
        else:
            raise ValueError(f"unknown baseline {kind!r}")
        rows.append(pd.DataFrame(
            {"y_true": yv[te], "y_pred": pred, "year": year},
            index=X.index[te],
        ))
    return pd.concat(rows)


def garch_forward_vol(
    close: pd.Series,
    target: pd.Series,
    *,
    horizon: int = 5,
    test_years: range = range(2016, 2026),
) -> pd.DataFrame:
    """GARCH(1,1) forecast of forward realized volatility (econometric benchmark).

    For each evaluation date t (the index of `target`, restricted to test_years),
    fit a GARCH(1,1) on daily log returns up to t and forecast the next `horizon`
    daily variances. The prediction is sqrt(mean variance), matching the target's
    definition (std of the next-h daily returns). Returns can be scaled to percent
    for the fit and rescaled back, which arch recommends for numerical stability.
    """
    from arch import arch_model

    ret = (np.log(close).diff().dropna()) * 100.0
    dates = target.index[target.index.year.isin(list(test_years))]
    rows = []
    for t in dates:
        hist = ret.loc[:t]
        if len(hist) < 250:
            continue
        res = arch_model(hist, mean="Constant", vol="GARCH", p=1, q=1).fit(disp="off")
        var = res.forecast(horizon=horizon, reindex=False).variance.values[-1]
        pred = float(np.sqrt(var.mean())) / 100.0
        rows.append((t, target.loc[t], pred, t.year))
    out = pd.DataFrame(rows, columns=["date", "y_true", "y_pred", "year"]).set_index("date")
    return out


def arima_forward_vol(
    target: pd.Series,
    *,
    order: tuple[int, int, int] = (2, 0, 1),
    test_years: range = range(2016, 2026),
) -> pd.DataFrame:
    """ARIMA benchmark on the (log) realized-volatility series itself.

    A univariate autoregressive reference: at each evaluation date t, fit an
    ARIMA on log-volatility history strictly before t and forecast one step. This
    is the fair time-series counterpart to GARCH -- richer than persistence, but
    using only the past volatility series (no exogenous features).
    """
    from statsmodels.tsa.arima.model import ARIMA

    logv = np.log(target)
    dates = target.index[target.index.year.isin(list(test_years))]
    rows = []
    for t in dates:
        hist = logv.loc[:t].iloc[:-1]  # strictly before t
        if len(hist) < 60:
            continue
        try:
            res = ARIMA(hist.to_numpy(), order=order).fit()
            pred = float(np.exp(res.forecast(1)[0]))
        except Exception:
            continue
        rows.append((t, target.loc[t], pred, t.year))
    return pd.DataFrame(rows, columns=["date", "y_true", "y_pred", "year"]).set_index("date")


def summarize(preds: pd.DataFrame) -> dict[str, float]:
    """Combine regression + directional metrics for a predictions frame."""
    yt, yp = preds["y_true"].to_numpy(), preds["y_pred"].to_numpy()
    return {**regression_metrics(yt, yp), **classification_metrics(yt, yp)}


if __name__ == "__main__":
    from src.data import load_prices
    from src.features import build_dataset

    X, y = build_dataset(load_prices())
    print(f"Evaluating on {X.shape[0]} samples, test years 2016-2025\n")
    print(f"{'baseline':<14} {'dir_acc':>8} {'roc_auc':>8} {'rmse':>9} {'r2':>9}")
    for kind in ["zero", "train_mean", "always_up", "persistence"]:
        m = summarize(baseline_predictions(X, y, kind=kind))
        print(f"{kind:<14} {m['dir_acc']:>8.3f} {m['roc_auc']:>8.3f} {m['rmse']:>9.5f} {m['r2']:>9.4f}")
