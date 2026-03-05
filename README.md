# FinGaze

FinGaze is a Streamlit-based AI trading and portfolio analysis project that combines:

- Reinforcement learning models for single-stock trading simulation.
- Rule-based sector allocation models for multi-stock portfolio construction.
- Conservative Monte Carlo-style future projection workflows.
- Visual comparison tools to evaluate model behavior and reliability.

The app is designed as a multi-page dashboard for experimentation, not as live-trading software.

## What This Project Does

FinGaze provides three major workflows:

1. Single-stock RL trading simulation (`app.py`)
2. Sector allocation and custom stock selection (`pages/1_Sector_Allocation.py`)
3. Model comparison across RL and sector models (`pages/3_Model_Comparison.py`)

It also includes a dedicated forecast-results page:

- `pages/2_Future_Prediction.py`

And a training script to create model files:

- `finrl_train.py`

## Core Capabilities

- Trains and loads 5 Stable-Baselines3 algorithms:
  - PPO
  - A2C
  - DDPG
  - SAC
  - TD3
- Downloads historical market data using `yfinance`.
- Backtests with explicit transaction cost modeling:
  - Fee (basis points)
  - Slippage (basis points)
- Supports long-only or long/short position constraints.
- Adds Monte Carlo / bootstrap-based future forecasting with conservative adjustments.
- Lets users override model-recommended sector picks and instantly recompute results.
- Compares models using return, accuracy, drawdown, Sharpe-like indicators, and ranking metrics.

## Project Structure

```text
FinGaze/
├── app.py                             # Main RL stock simulation dashboard
├── finrl_train.py                     # RL model training script
├── models/
│   ├── a2c_model.zip
│   ├── ddpg_model.zip
│   ├── ppo_model.zip
│   ├── sac_model.zip
│   └── td3_model.zip
└── pages/
    ├── 1_Sector_Allocation.py         # Sector portfolio builder + custom selection
    ├── 2_Future_Prediction.py         # Forecast-focused deep-dive page
    └── 3_Model_Comparison.py          # RL and sector-model comparison lab
```

## How The Main Components Work

### 1) RL Training (`finrl_train.py`)

- Downloads ~2 years of AAPL close data.
- Converts prices to daily returns.
- Builds a custom Gymnasium environment:
  - Observation: latest return.
  - Action: continuous position in `[-1, 1]`.
  - Reward: action at time `t` multiplied by return at `t+1` (no look-ahead).
- Trains PPO/A2C/DDPG/SAC/TD3 for `20000` timesteps each.
- Saves models to the `models/` directory.

### 2) RL Backtesting + Forecast (`app.py`)

- User chooses ticker, date range, algorithm, costs, and risk settings.
- App downloads historical prices and computes returns.
- Loads selected trained model from `models/<algo>_model.zip`.
- Runs backtest with:
  - Position scaling (`signal_strength`)
  - Optional long-only clipping
  - Turnover-based cost deduction
- Produces:
  - Portfolio equity curve
  - Buy-and-hold comparison
  - Return/cost summary cards and charts

Forecast mode:

- Uses historical return sampling with:
  - Lookback window
  - Block bootstrap
  - Conservative drift/volatility adjustments
- Runs many simulations (Monte Carlo style).
- Reports expected, median, P10, P90, and probability of gain.

### 3) Sector Allocation (`pages/1_Sector_Allocation.py`)

- Uses predefined sector universes (Healthcare, Banking, IT, Energy, etc.).
- Evaluates stocks with 5 scoring models:
  - Model 1: Momentum
  - Model 2: Momentum + Low Vol
  - Model 3: Equal Risk (Low Vol preference)
  - Model 4: Quality Uptrend
  - Model 5: Mixed Score
- Picks one stock per selected sector and allocates capital by normalized positive scores.
- Lets user override each sector pick manually.
- Rebuilds portfolio equity and sector performance from chosen tickers.
- Creates conservative future projections per selected stock and stores results in session state.

### 4) Future Prediction (`pages/2_Future_Prediction.py`)

- Reads forecast outputs produced by Sector Allocation page.
- Shows:
  - Expected future profit and return
  - Risk flags (low gain probability / downside scenarios)
  - Forecasted return bars
  - Allocation pie chart
  - Per-stock expected/P10/P90 price and equity curves

### 5) Model Comparison (`pages/3_Model_Comparison.py`)

- Compares RL models on the app-selected ticker.
- Metrics include:
  - Return %
  - Accuracy %
  - Sharpe
  - Max Drawdown %
  - Composite score
- Separately compares sector models on sector-page stock universe using:
  - Direction accuracy
  - Rank IC
  - Top-picks average return

## Setup

### 1) Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install streamlit yfinance numpy pandas gymnasium stable-baselines3 plotly
```

## Run The App

From the project root:

```bash
streamlit run app.py
```

Streamlit will open the main page, and additional pages appear automatically in the left navigation.

## Train / Retrain Models

If model files are missing or you want fresh models:

```bash
python finrl_train.py
```

This generates/overwrites:

- `models/ppo_model.zip`
- `models/a2c_model.zip`
- `models/ddpg_model.zip`
- `models/sac_model.zip`
- `models/td3_model.zip`

## Notes And Limitations

- Educational/research tool only; not financial advice.
- Strategy logic is simplified and does not represent full market microstructure.
- Forecasting is scenario-based and conservative, but still uncertain.
- Data quality depends on `yfinance` availability and ticker coverage.
- Results are sensitive to:
  - date range
  - fee/slippage assumptions
  - simulation settings
  - selected sectors/stocks

## Recommended Improvements

- Add a `requirements.txt` or `pyproject.toml` for reproducible installs.
- Add unit tests for environment logic and portfolio math.
- Add experiment tracking for model versions and training metrics.
- Separate analytics engine from UI layer for cleaner architecture.
- Add CI checks (lint, tests, formatting).

## Git Quick Commands

```bash
git status
git add .
git commit -m "Add detailed README"
git push -u origin main
```

