# bovespa-index-prediction

Predicting the Bovespa index (Ibovespa) with technical indicators and machine
learning. Originally my undergraduate graduation thesis (UFABC, 2021); this is a
rebuilt, modernized, and methodologically corrected version.

The original notebook is preserved unchanged as [`TGII.ipynb`](TGII.ipynb).

## What changed in the rebuild

The 2021 thesis reported strong predictive results. Revisiting it revealed those
gains were largely **methodological artifacts**, not real signal. The rebuild
fixes them and re-asks the question honestly:

| Issue in the original | Fix |
|---|---|
| `StandardScaler` fit on the **whole series** (incl. test years) | Scaler fit on the **training fold only**, inside each walk-forward fold |
| 5-day forward return sampled **daily** → ~80% autocorrelated targets | **Non-overlapping** weekly sampling |
| No baseline → metrics uninterpretable | Naive baselines: zero, train-mean, always-up, persistence, HAR-RV |
| MLP with all-**sigmoid** hidden layers (vanishing gradients) | Keras 3 MLP with **ReLU** + early stopping |
| `pmdarima` / `tf.keras`, unpinned | **Keras 3** (torch backend), `statsforecast`, pinned `requirements.txt` |
| Aroon/MassIndex called with wrong args | Corrected to the current `ta` API |

## Key finding

- **Return direction is not predictable.** Once leakage and target-overlap are
  removed, no model beats the ~0.54 up-week base rate (R² ≈ 0).
- **Realized volatility *is* predictable**, and the model ranking is instructive:

  | model (1-week realized volatility) | R² | RMSE | MAE |
  |---|---:|---:|---:|
  | **GARCH(1,1)** | **0.272** | 0.00758 | 0.00476 |
  | ARIMA(2,0,1) | 0.224 | 0.00783 | 0.00422 |
  | Gradient Boosting (log) | 0.156 | 0.00816 | 0.00462 |
  | LightGBM (log) | 0.149 | 0.00820 | 0.00462 |
  | baseline: persistence | 0.088 | 0.00849 | 0.00532 |
  | HAR-RV (OLS) | −0.285 | 0.01007 | 0.00469 |
  | MLP (ReLU, log) | −0.736 | 0.01171 | 0.00679 |
  | LSTM (lookback 4) | −1.864 | 0.01504 | 0.01045 |

  The purpose-built econometric models (GARCH, ARIMA) lead; gradient boosters add
  modest value over naive persistence; and the neural nets *trail* — an honest
  reminder that ~960 weekly samples is far too little for deep sequence models.

Reproduce both tables with `python run_experiment.py`.

## Project layout

```
src/data.py         cached yfinance loader (Ibovespa, S&P 500, USD/BRL)
src/features.py     technical indicators, return + volatility targets, HAR-RV
src/evaluation.py   walk-forward (tabular + sequence), per-fold scaling, metrics,
                    baselines, GARCH and ARIMA volatility benchmarks
src/models.py       GBR, LightGBM, ReLU MLP, LSTM (Keras 3), HAR-RV OLS
run_experiment.py   reproduces the full comparison
TGII.ipynb          original 2021 thesis notebook (unchanged)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
KERAS_BACKEND=torch python run_experiment.py
```
