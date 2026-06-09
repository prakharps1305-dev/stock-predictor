# Stock Return Predictor

End-to-end machine learning pipeline for predicting next-day stock returns, evaluating model quality, and testing whether predictions survive realistic backtesting.

This project is designed for learning and research. It focuses on process quality (no leakage, robust evaluation, and realistic assumptions), not just model accuracy.

> Not financial advice. Educational use only.

---

## Table of contents

- [1) Project goals](#1-project-goals)
- [2) What this pipeline does](#2-what-this-pipeline-does)
- [3) System flow](#3-system-flow)
- [4) Tech stack](#4-tech-stack)
- [5) Project structure](#5-project-structure)
- [6) Quickstart](#6-quickstart)
- [7) Experiment configuration](#7-experiment-configuration)
- [8) Modeling and strategy design](#8-modeling-and-strategy-design)
- [9) Evaluation metrics](#9-evaluation-metrics)
- [10) Monte Carlo validation](#10-monte-carlo-validation)
- [11) Results template](#11-results-template)
- [12) Reproducibility checklist](#12-reproducibility-checklist)
- [13) Known limitations](#13-known-limitations)
- [14) Enhanced roadmap](#14-enhanced-roadmap)
- [15) References](#15-references)

---

## 1) Project goals

- Build a clean baseline for return prediction using tabular ML.
- Learn quant-research discipline:
  - strict time-based split
  - leakage prevention
  - risk-aware evaluation
- Compare linear, tree-based, and neural approaches on the same feature set.
- Validate edge using both backtest metrics and Monte Carlo stress testing.

---

## 2) What this pipeline does

1. Downloads OHLCV price data with `yfinance`.
2. Engineers about 20 features:
   - trend (moving averages, momentum)
   - mean-reversion (RSI, Bollinger distance)
   - risk/volatility (rolling std, drawdown-like behavior proxies)
   - autoregressive signals (lagged returns)
3. Trains three model families:
   - Linear Regression (interpretable baseline)
   - Random Forest (non-linear tabular model)
   - MLP in PyTorch (learned non-linear interactions)
4. Predicts next-day returns on a strictly out-of-sample test window.
5. Converts predictions into long/short strategy returns.
6. Reports both ML and trading performance metrics.
7. Runs 1000-path Monte Carlo resampling to assess result stability.

---

## 3) System flow

```mermaid
flowchart LR
    A["Raw market data (OHLCV)"] --> B["Feature engineering"]
    B --> C["Time-based split (train/test)"]
    C --> D["Model training"]
    D --> E["Return prediction"]
    E --> F["Signal generation"]
    F --> G["Backtest with transaction costs"]
    G --> H["Metrics: Sharpe, DD, CAGR, IC, Directional Accuracy"]
    H --> I["Monte Carlo robustness validation"]
```

---

## 4) Tech stack

- `yfinance` for historical market data
- `pandas`, `numpy` for data wrangling and feature construction
- `scikit-learn` for preprocessing and baseline/tree models
- `PyTorch` for neural network model (MLP)
- `scipy` for statistical evaluation (for example, Spearman IC)
- `matplotlib`, `seaborn` for plots and diagnostics

---

## 5) Project structure

```text
stock-predictor/
|-- data/                  # raw/intermediate price data (gitignored)
|-- notebooks/
|   |-- 01_data.ipynb      # ingestion and cleaning
|   |-- 02_features.ipynb  # feature engineering and target creation
|   |-- 03_models.ipynb    # model training and prediction
|   `-- 04_backtest.ipynb  # strategy simulation and evaluation
|-- results/
|   |-- equity_curve.png
|   |-- feature_importance.png
|   `-- monte_carlo.png
|-- requirements.txt
`-- README.md
```

---

## 6) Quickstart

### Prerequisites

- Python 3.10+ recommended
- `pip` and `venv`
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/yourusername/stock-predictor
cd stock-predictor
python -m venv .venv
```

Activate environment:

- macOS/Linux:

```bash
source .venv/bin/activate
```

- Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the pipeline

```bash
jupyter notebook
```

Execute notebooks in this order:

1. `01_data.ipynb`
2. `02_features.ipynb`
3. `03_models.ipynb`
4. `04_backtest.ipynb`

---

## 7) Experiment configuration

Document these values for each experiment run:

- ticker(s): example `AAPL`, `MSFT`, `SPY`
- train start and end dates
- test start and end dates
- target definition:
  - typically next-day log return
- transaction cost:
  - currently assumed `0.1%` per trade
- random seed(s) for reproducibility

Why this matters:
- Without explicit config tracking, results are hard to compare or reproduce.

---

## 8) Modeling and strategy design

### Data split policy

- Use chronological split only.
- No random shuffling.
- Any scaler or transformer is fit on train only, then applied to test.

### Target

- Predict next-day return, not next-day price.
- Returns are generally more stationary than raw prices.

### Model set

- **Linear Regression**: baseline and sanity check.
- **Random Forest**: captures non-linear effects without heavy tuning.
- **MLP (PyTorch)**: flexible function approximator with dropout/early stopping.

### Signal and backtest logic

- Convert predicted returns into position signals (long/short).
- Apply transaction costs on position changes.
- Compare strategy against buy-and-hold benchmark.

---

## 9) Evaluation metrics

Metrics are grouped by prediction quality and portfolio behavior.

### Prediction quality

- **IC (Information Coefficient)**:
  - Spearman rank correlation between predictions and realized returns.
- **Directional Accuracy**:
  - percent of times model got sign of return correct.

### Portfolio quality

- **Sharpe Ratio**:
  - risk-adjusted return (higher is better).
- **Max Drawdown**:
  - largest peak-to-trough portfolio decline (lower magnitude is better).
- **CAGR**:
  - annualized compounded growth rate.

Use multiple metrics together. A model can look strong on one metric and weak on another.

---

## 10) Monte Carlo validation

Purpose:
- test whether observed performance is likely robust or just noise.

Method (current design):

1. Resample return paths (bootstrap style) 1000 times.
2. Recompute performance distribution (for example, Sharpe).
3. Locate actual strategy metric inside that distribution.

Interpretation:
- If realized Sharpe is near center of random distribution, edge is likely weak.
- If it is consistently in upper tail, confidence improves.

---

## 11) Results template

Fill this table after running the full workflow.

| Model | Sharpe | Max DD | CAGR | Win Rate | IC | Dir Acc |
|---|---:|---:|---:|---:|---:|---:|
| Buy & Hold | 1.248 | 0.179 | 0.302 | 0.543 | N/A | N/A |
| Linear Regression | -1.026 | 0.414 | -0.236 | 0.465 | -0.008 | 0.471 |
| Random Forest | -0.685 | 0.298 | -0.172 | 0.472 | 0.015 | 0.474 |
| MLP (PyTorch) | TODO | TODO | TODO | TODO | TODO | TODO |

Also include run context:

- Date range: 2018-01-02 to 2023-12-28
- Ticker universe: AAPL
- Split date: 2022-11-02 (train) / 2022-11-03 (test)
- Transaction cost: 0.1% per trade
- Number of test observations: ~300 days

---

## 12) Reproducibility checklist

- [ ] Fixed random seeds (`numpy`, `torch`, and sklearn where needed)
- [ ] All library versions pinned in `requirements.txt`
- [ ] Data pull date recorded
- [ ] Train/test boundary documented
- [ ] Cost assumptions documented
- [ ] Results and plots exported to `results/`

---

## 13) Known limitations

- Limited to a small feature set and model set.
- Single-split evaluation can still be regime-sensitive.
- Backtest assumptions are simplified:
  - no slippage modeling beyond fixed cost
  - no liquidity or borrow constraints
- Predictive signals in markets decay quickly over time.

---

## 14) Enhanced roadmap

This is the upgraded direction for turning this into a stronger quant-research project.

### Phase 1: stronger validation

- Replace single split with walk-forward validation.
- Add expanding window retraining.
- Track performance by market regime (bull, bear, high-volatility).

### Phase 2: richer modeling

- Add gradient boosting (`XGBoost`/`LightGBM`) as strong tabular benchmarks.
- Try probabilistic outputs and confidence-calibrated position sizing.
- Add feature selection and SHAP-based interpretability checks.

### Phase 3: portfolio realism

- Move from single-asset to cross-sectional ranking across many stocks.
- Introduce risk controls:
  - max position size
  - sector neutrality
  - volatility targeting
- Add turnover, hit ratio by decile, and exposure diagnostics.

### Phase 4: research engineering

- Add experiment tracking (`MLflow` or lightweight CSV logs).
- Convert notebooks into modular scripts for repeatable runs.
- Add unit tests for feature logic and leakage guards.

---

## 15) References

- Gu, Kelly, Xiu (2020): [Empirical Asset Pricing via Machine Learning](https://academic.oup.com/rfs/article-abstract/33/5/2223/5758276)
- Lopez de Prado (2018): [The 10 Reasons Most ML Funds Fail](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3104816)
- Stefan Jansen: [Machine Learning for Trading](https://github.com/stefan-jansen/machine-learning-for-trading)
- QuantStart: [Forecasting Financial Time Series](https://www.quantstart.com/articles/Forecasting-Financial-Time-Series-Part-1/)

---

## License

Add your preferred license here (for example, MIT) before publishing publicly.
