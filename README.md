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
- **Realized volatility *is* predictable.** Reframing the target to next-week
  realized volatility, Gradient Boosting on log-volatility reaches **R² ≈ 0.16**,
  beating the naive persistence baseline (**≈ 0.09**) — an honest positive result.

Reproduce both tables with `python run_experiment.py`.

## Project layout

```
src/data.py         cached yfinance loader (Ibovespa, S&P 500, USD/BRL)
src/features.py     technical indicators, return + volatility targets, HAR-RV
src/evaluation.py   walk-forward splits, per-fold scaling, metrics, baselines
src/models.py       GBR, ReLU MLP (Keras 3), HAR-RV OLS
run_experiment.py   reproduces the full comparison
TGII.ipynb          original 2021 thesis notebook (unchanged)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
KERAS_BACKEND=torch python run_experiment.py
```
