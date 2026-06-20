"""Model factories with a uniform sklearn-style (fit/predict) interface.

These plug directly into evaluation.walk_forward_predict, which handles the
per-fold scaling and walk-forward splitting. Two families live here for now:

- Gradient Boosting (scikit-learn), with the modern loss names ('squared_error'
  etc. -- the original notebook's 'ls'/'lad' were removed in sklearn 1.0+).
- A Keras 3 MLP using ReLU hidden layers instead of the original all-sigmoid
  stack, which suffered vanishing gradients.

Set KERAS_BACKEND before importing this module if you want a backend other than
torch (the default chosen here for clean macOS/arm64 installs).
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


def make_har() -> LinearRegression:
    """HAR-RV benchmark: ordinary least squares on realized-vol features.

    The literature reference for volatility forecasting. Intended to be run on a
    feature matrix restricted to the trailing realized-vol columns (RV5/22/66).
    """
    return LinearRegression()


def make_gbr(random_state: int = 42) -> GradientBoostingRegressor:
    """Gradient Boosting Regressor with sensible, modern defaults."""
    return GradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        n_estimators=300,
        subsample=0.8,
        max_depth=3,
        min_samples_leaf=20,
        random_state=random_state,
    )


def make_lgbm(random_state: int = 42):
    """LightGBM regressor -- the modern gradient booster (post-2021 standard).

    A faster, histogram-based alternative to sklearn's GBR with leaf-wise growth
    and built-in regularization. Conservative settings to suit the small,
    low-signal weekly dataset.
    """
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=400,
        learning_rate=0.02,
        num_leaves=15,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_lambda=1.0,
        random_state=random_state,
        verbosity=-1,
    )


class KerasMLP:
    """Minimal sklearn-style wrapper around a Keras 3 MLP regressor (ReLU)."""

    def __init__(self, units=(32, 16), dropout=0.1, epochs=200,
                 batch_size=32, patience=25, random_state=42):
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.model_ = None

    def fit(self, X, y):
        import keras

        keras.utils.set_random_seed(self.random_state)
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")

        layers = [keras.layers.Input(shape=(X.shape[1],))]
        for u in self.units:
            layers.append(keras.layers.Dense(u, activation="relu"))
            layers.append(keras.layers.Dropout(self.dropout))
        layers.append(keras.layers.Dense(1, activation="linear"))
        self.model_ = keras.Sequential(layers)
        self.model_.compile(optimizer="adam", loss="mse")

        cb = [
            keras.callbacks.EarlyStopping(
                monitor="loss", patience=self.patience, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=self.patience // 2, min_lr=1e-5
            ),
        ]
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                        verbose=0, callbacks=cb)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype="float32")
        return self.model_.predict(X, verbose=0).ravel()


def make_mlp(random_state: int = 42) -> KerasMLP:
    return KerasMLP(random_state=random_state)


class KerasLSTM:
    """Minimal sklearn-style wrapper around a Keras 3 LSTM regressor.

    Expects 3D input of shape (samples, lookback, n_features). Built for the
    sequence walk-forward path in evaluation.walk_forward_predict_seq.
    """

    def __init__(self, units=16, dropout=0.1, epochs=200,
                 batch_size=16, patience=20, random_state=42):
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.model_ = None

    def fit(self, X, y):
        import keras

        keras.utils.set_random_seed(self.random_state)
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        self.model_ = keras.Sequential([
            keras.layers.Input(shape=(X.shape[1], X.shape[2])),
            keras.layers.LSTM(self.units),
            keras.layers.Dropout(self.dropout),
            keras.layers.Dense(1, activation="linear"),
        ])
        self.model_.compile(optimizer="adam", loss="mse")
        cb = [keras.callbacks.EarlyStopping(
            monitor="loss", patience=self.patience, restore_best_weights=True)]
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                        verbose=0, callbacks=cb)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype="float32")
        return self.model_.predict(X, verbose=0).ravel()


def make_lstm(random_state: int = 42) -> KerasLSTM:
    return KerasLSTM(random_state=random_state)
