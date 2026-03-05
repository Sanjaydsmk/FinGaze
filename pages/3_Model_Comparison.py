import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3

try:
    import plotly.express as px
except ImportError:
    px = None

st.set_page_config(page_title="Model Comparison", layout="wide")

st.markdown(
    """
<style>
.stApp {
    background: radial-gradient(circle at 10% 0%, #0b1220 0%, #0f172a 40%, #020617 100%);
    color: #e2e8f0;
}
.hero {
    background: linear-gradient(125deg, #111827, #1e3a8a);
    border-radius: 16px;
    padding: 18px 20px;
    color: #f8fafc;
    margin-bottom: 12px;
}
.card {
    background: linear-gradient(180deg, #0f172a, #111827);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 12px;
}
.winner {
    border: 1px solid #38bdf8;
    box-shadow: 0 8px 22px rgba(14, 165, 233, 0.18);
}
.rank-chip {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.8rem;
    color: #cbd5e1;
    background: #1f2937;
    margin-bottom: 6px;
}
</style>
""",
    unsafe_allow_html=True,
)


def performance_label(return_pct: float) -> str:
    if return_pct >= 15:
        return "Excellent"
    if return_pct >= 5:
        return "Good"
    if return_pct >= 0:
        return "Stable"
    return "Weak"


def accuracy_label(acc_pct: float) -> str:
    if acc_pct >= 65:
        return "High"
    if acc_pct >= 55:
        return "Medium"
    return "Low"


def style_fig(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="#020617", plot_bgcolor="#0f172a")
    return fig


st.markdown(
    """
<div class="hero">
  <h2>Model Comparison Lab</h2>
  <p>Compare RL trading models for stocks and sector-allocation models using original historical data.</p>
</div>
""",
    unsafe_allow_html=True,
)

MODEL_PATH = "models"
RL_MODELS = {
    "PPO": PPO,
    "A2C": A2C,
    "DDPG": DDPG,
    "SAC": SAC,
    "TD3": TD3,
}

SECTOR_TICKERS: Dict[str, List[str]] = {
    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "BMY", "MDT"],
    "Banking": ["JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "PNC", "TFC", "SCHW"],
    "Education": ["CHGG", "LOPE", "LRN", "TAL", "EDU", "COUR", "STRA", "ATGE", "LAUR"],
    "IT": ["MSFT", "AAPL", "GOOGL", "NVDA", "ADBE", "CRM", "ORCL", "CSCO", "AMD", "INTC"],
    "Telecommunication": ["VZ", "T", "TMUS", "CHTR", "CMCSA", "DISH", "ATUS", "LBRDK"],
    "Automobiles": ["TSLA", "GM", "F", "RIVN", "LCID", "NIO", "XPEV", "LI", "TM", "HMC"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "BKR"],
    "Consumer Staples": ["WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "CL", "KMB", "GIS"],
    "Consumer Discretionary": ["AMZN", "TSLA", "MCD", "SBUX", "NKE", "HD", "LOW", "BKNG", "MAR", "CMG"],
    "Industrials": ["BA", "CAT", "GE", "RTX", "HON", "LMT", "DE", "ETN", "UPS", "UNP"],
    "Materials": ["LIN", "APD", "SHW", "ECL", "NEM", "FCX", "DOW", "DD", "NUE", "CTVA"],
    "Utilities": ["NEE", "DUK", "SO", "AEP", "XEL", "D", "SRE", "PEG", "ED", "EIX"],
    "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "SPG", "O", "WELL", "DLR", "VICI", "AVB"],
    "Media & Entertainment": ["META", "NFLX", "DIS", "CMCSA", "WBD", "PARA", "FOXA", "ROKU", "SPOT", "TTWO"],
}

SECTOR_MODELS = [
    "Model 1 – Momentum",
    "Model 2 – Momentum + Low Vol",
    "Model 3 – Equal Risk (Low Vol)",
    "Model 4 – Quality Uptrend",
    "Model 5 – Mixed Score",
]


def to_scalar(action) -> float:
    return float(np.squeeze(action))


def load_rl_model(name: str):
    path = os.path.join(MODEL_PATH, f"{name.lower()}_model.zip")
    if not os.path.exists(path):
        return None
    return RL_MODELS[name].load(path)


def extract_close_series(data: pd.DataFrame) -> pd.Series:
    """Normalize yfinance output to a single close-price series."""
    if data is None or data.empty:
        return pd.Series(dtype=np.float64)

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            return pd.Series(dtype=np.float64)
        close_part = data["Close"]
        if isinstance(close_part, pd.DataFrame):
            if close_part.shape[1] < 1:
                return pd.Series(dtype=np.float64)
            return close_part.iloc[:, 0].dropna()
        return close_part.dropna()

    if "Close" not in data.columns:
        return pd.Series(dtype=np.float64)
    return data["Close"].dropna()


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    s = pd.Series(equity)
    peak = s.cummax()
    dd = (s - peak) / peak.replace(0, np.nan)
    return float(dd.min() * 100.0)


def annualized_sharpe(strategy_returns: np.ndarray) -> float:
    if len(strategy_returns) < 2:
        return 0.0
    mu = float(np.mean(strategy_returns))
    sigma = float(np.std(strategy_returns))
    if sigma <= 1e-8:
        return 0.0
    return (mu / sigma) * math.sqrt(252.0)


def backtest_with_accuracy(
    model,
    returns: np.ndarray,
    initial_capital: float,
    long_only: bool,
    signal_strength: float,
    fee_bps: float,
    slippage_bps: float,
) -> Dict[str, float]:
    capital = float(initial_capital)
    equity = [capital]
    current_position = 0.0
    correct = 0
    total = 0

    for i in range(1, len(returns)):
        obs = np.array([returns[i - 1]], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        raw_position = to_scalar(action) * signal_strength
        if long_only:
            position = float(np.clip(raw_position, 0.0, 1.0))
            pred_sign = 1 if position > 0.1 else 0
            act_sign = 1 if returns[i] > 0 else 0
        else:
            position = float(np.clip(raw_position, -1.0, 1.0))
            pred_sign = 1 if position > 0.05 else (-1 if position < -0.05 else 0)
            act_sign = 1 if returns[i] > 0 else (-1 if returns[i] < 0 else 0)

        correct += int(pred_sign == act_sign)
        total += 1

        turnover = abs(position - current_position)
        cost_rate = (fee_bps + slippage_bps) / 10000.0
        capital = max(0.0, capital - capital * turnover * cost_rate)
        current_position = position

        capital = max(0.0, capital * (1.0 + position * float(returns[i])))
        equity.append(capital)

    eq = np.asarray(equity, dtype=np.float64)
    ret_pct = ((eq[-1] / initial_capital) - 1.0) * 100.0 if initial_capital > 0 else 0.0
    strat_rets = pd.Series(eq).pct_change().dropna().to_numpy(dtype=np.float64)

    return {
        "Return %": ret_pct,
        "Sharpe": annualized_sharpe(strat_rets),
        "MaxDD %": max_drawdown(eq),
        "Accuracy %": (correct / total * 100.0) if total > 0 else 0.0,
    }


def annualized_stats(series: pd.Series) -> Tuple[float, float]:
    px = series.dropna()
    if len(px) < 30:
        return 0.0, 0.0
    rets = px.pct_change().dropna()
    if rets.empty:
        return 0.0, 0.0
    ann_ret = ((1.0 + float(rets.mean())) ** 252 - 1.0) * 100.0
    ann_vol = float(rets.std()) * math.sqrt(252.0) * 100.0
    return ann_ret, ann_vol


def build_score(model_name: str, ann_ret: float, ann_vol: float) -> float:
    if ann_vol <= 0:
        return -1e9
    sharpe_like = ann_ret / ann_vol
    if model_name == SECTOR_MODELS[0]:
        return ann_ret
    if model_name == SECTOR_MODELS[1]:
        return ann_ret - 0.5 * ann_vol
    if model_name == SECTOR_MODELS[2]:
        return -ann_vol
    if model_name == SECTOR_MODELS[3]:
        return ann_ret - 0.3 * ann_vol if ann_ret > 0 else -1e9
    if model_name == SECTOR_MODELS[4]:
        return 0.6 * ann_ret - 0.2 * ann_vol + 10.0 * sharpe_like
    return ann_ret


def compare_sector_models_on_stocks(
    prices: pd.DataFrame,
    stock_pool: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    stock_rows = []
    min_train_days = 80
    rebalance_step = 21
    forward_horizon = 21

    prices = prices.sort_index()
    n = len(prices.index)
    if n < (min_train_days + forward_horizon + 20):
        return pd.DataFrame(), pd.DataFrame()

    rebalance_points = list(range(min_train_days, n - forward_horizon, rebalance_step))
    if not rebalance_points:
        return pd.DataFrame(), pd.DataFrame()

    for model_name in SECTOR_MODELS:
        hits_dir = 0
        total = 0
        window_rank_ics = []
        top_returns = []
        all_returns = []

        for rebalance_i in rebalance_points:
            window_scores = []
            window_returns = []
            eval_count_window = 0

            for t in stock_pool:
                if t not in prices.columns:
                    continue

                s_train = prices.iloc[:rebalance_i][t].dropna()
                if len(s_train) < min_train_days:
                    continue

                s_forward = prices.iloc[rebalance_i : rebalance_i + forward_horizon + 1][t].dropna()
                if len(s_forward) < 2:
                    continue

                ann_ret, ann_vol = annualized_stats(s_train)
                score = build_score(model_name, ann_ret, ann_vol)
                fwd_ret = float(s_forward.iloc[-1] / s_forward.iloc[0] - 1.0) * 100.0

                pred_up = score > 0.0
                actual_up = fwd_ret > 0.0
                hits_dir += int(pred_up == actual_up)
                total += 1
                eval_count_window += 1

                window_scores.append(score)
                window_returns.append(fwd_ret)
                all_returns.append(fwd_ret)

                stock_rows.append(
                    {
                        "Sector Model": model_name,
                        "Stock": t,
                        "Rebalance Date": prices.index[rebalance_i],
                        "Model Score": score,
                        "Forward Return %": fwd_ret,
                        "Direction Hit": 1 if pred_up == actual_up else 0,
                    }
                )

            if eval_count_window < 2:
                continue

            s_score = pd.Series(window_scores, dtype=float)
            s_ret = pd.Series(window_returns, dtype=float)
            ic = float(s_score.rank().corr(s_ret.rank())) if len(s_score) >= 2 else np.nan
            if not np.isnan(ic):
                window_rank_ics.append(ic)

            top_n = max(1, int(math.ceil(eval_count_window * 0.3)))
            top_idx = s_score.sort_values(ascending=False).index[:top_n]
            top_avg_ret = float(s_ret.loc[top_idx].mean()) if len(top_idx) > 0 else 0.0
            top_returns.append(top_avg_ret)

        if total == 0:
            continue

        rank_ic = float(np.mean(window_rank_ics)) if window_rank_ics else 0.0
        top_avg_ret = float(np.mean(top_returns)) if top_returns else 0.0
        all_avg_ret = float(np.mean(all_returns)) if all_returns else 0.0

        rows.append(
            {
                "Sector Model": model_name,
                "Direction Accuracy %": (hits_dir / total) * 100.0,
                "Rank IC": rank_ic,
                "Top Picks Avg Return %": top_avg_ret,
                "All Stocks Avg Return %": all_avg_ret,
                "Predictions Evaluated": total,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(stock_rows)


with st.sidebar:
    st.header("Comparison Settings")
    today = pd.Timestamp.today().date()
    from_date = st.date_input("From Date", value=today - pd.Timedelta(days=365))
    to_date = st.date_input("To Date", value=today)
    initial_capital = st.number_input("Initial Capital ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
    long_only = st.checkbox("Long Only", value=True)
    signal_strength = st.slider("Signal Strength", 0.5, 3.0, 1.5, 0.1)
    fee_bps = st.number_input("Fee (bps)", min_value=0.0, max_value=100.0, value=2.0, step=1.0)
    slippage_bps = st.number_input("Slippage (bps)", min_value=0.0, max_value=100.0, value=2.0, step=1.0)
    selected_sectors = st.multiselect(
        "Sectors for sector-model comparison",
        options=list(SECTOR_TICKERS.keys()),
        default=["IT", "Healthcare", "Banking"],
    )

run = st.button("Run Model Comparison")

if run:
    if from_date >= to_date:
        st.error("`To Date` must be after `From Date`.")
        st.stop()

    app_ticker = st.session_state.get("app_selected_ticker")
    if not app_ticker:
        backtest_obj = st.session_state.get("app_backtest")
        if isinstance(backtest_obj, dict):
            app_ticker = backtest_obj.get("ticker")

    if not app_ticker:
        st.error("No app-page stock found. Set stock ticker in app page first.")
        st.stop()
    tickers = [str(app_ticker).strip().upper()]
    st.caption(f"Using app-page stock for RL comparison: **{tickers[0]}**")

    st.markdown("### 1) RL Model Comparison (App Models)")

    rl_rows = []

    for t in tickers:
        data = yf.download(t, start=from_date, end=to_date, auto_adjust=True, progress=False)
        close = extract_close_series(data)
        if len(close) < 60:
            continue
        returns = close.pct_change().dropna().to_numpy(dtype=np.float64).reshape(-1)
        if len(returns) < 30:
            continue

        for name in RL_MODELS.keys():
            model = load_rl_model(name)
            if model is None:
                continue
            metrics = backtest_with_accuracy(
                model=model,
                returns=returns,
                initial_capital=float(initial_capital),
                long_only=long_only,
                signal_strength=float(signal_strength),
                fee_bps=float(fee_bps),
                slippage_bps=float(slippage_bps),
            )
            score = metrics["Return %"] + 0.7 * metrics["Accuracy %"] + 3.0 * metrics["Sharpe"] + 0.5 * metrics["MaxDD %"]
            row = {"Stock": t, "Model": name, **metrics, "Composite Score": score}
            rl_rows.append(row)

    rl_df = pd.DataFrame(rl_rows)

    if rl_df.empty:
        st.warning("Could not evaluate RL models. Check model files or stock/date inputs.")
    else:
        st.markdown("#### Simple result")
        for stock in sorted(rl_df["Stock"].unique()):
            stock_df = rl_df[rl_df["Stock"] == stock].sort_values("Composite Score", ascending=False).reset_index(drop=True)
            winner = stock_df.iloc[0]
            st.markdown(
                f"""
<div class='card winner'>
  <div class='rank-chip'>{stock}</div>
  <h4 style='margin:2px 0 8px 0;'>Recommended Model: {winner['Model']}</h4>
  <div>Expected Performance: {performance_label(float(winner['Return %']))} ({winner['Return %']:.2f}%)</div>
  <div>Prediction Reliability: {accuracy_label(float(winner['Accuracy %']))} ({winner['Accuracy %']:.2f}%)</div>
</div>
""",
                unsafe_allow_html=True,
            )

            simple_df = stock_df[["Model", "Return %", "Accuracy %"]].copy()
            simple_df["Performance"] = simple_df["Return %"].apply(performance_label)
            simple_df["Reliability"] = simple_df["Accuracy %"].apply(accuracy_label)
            simple_df = simple_df[["Model", "Performance", "Reliability", "Return %", "Accuracy %"]]
            simple_df["Return %"] = simple_df["Return %"].map(lambda x: f"{x:.2f}%")
            simple_df["Accuracy %"] = simple_df["Accuracy %"].map(lambda x: f"{x:.2f}%")
            st.table(simple_df.reset_index(drop=True))

            if px is not None:
                fig_rl = px.bar(
                    stock_df,
                    x="Model",
                    y=["Return %", "Accuracy %"],
                    barmode="group",
                    title=f"{stock} - RL Model Comparison",
                )
                st.plotly_chart(style_fig(fig_rl), use_container_width=True)

    st.markdown("### 2) Sector Model Comparison (Sector Page Models)")
    if not selected_sectors:
        st.info("Select at least one sector to compare sector models.")
    else:
        sector_page_selected_df = st.session_state.get("sector_page_selected_stock_df")
        sector_page_model_df = st.session_state.get("sector_page_last_stock_df")

        stock_pool = []
        if isinstance(sector_page_selected_df, pd.DataFrame) and not sector_page_selected_df.empty and "Ticker" in sector_page_selected_df.columns:
            stock_pool.extend([str(x).upper() for x in sector_page_selected_df["Ticker"].dropna().unique().tolist()])
        if isinstance(sector_page_model_df, pd.DataFrame) and not sector_page_model_df.empty and "Ticker" in sector_page_model_df.columns:
            stock_pool.extend([str(x).upper() for x in sector_page_model_df["Ticker"].dropna().unique().tolist()])
        if not stock_pool:
            stock_pool = sorted({t for s in selected_sectors for t in SECTOR_TICKERS[s]})

        stock_pool = sorted(list(dict.fromkeys(stock_pool)))
        st.caption(f"Comparing sector models on sector-page stocks: {', '.join(stock_pool)}")

        prices = yf.download(stock_pool, start=from_date, end=to_date, auto_adjust=True, progress=False)

        if prices.empty:
            st.warning("No sector data found.")
        else:
            if isinstance(prices.columns, pd.MultiIndex):
                if "Close" in prices.columns.get_level_values(0):
                    prices = prices["Close"]
                else:
                    st.warning("Close prices missing for sector comparison.")
                    prices = pd.DataFrame()
            else:
                prices = prices[["Close"]]
                prices.columns = [stock_pool[0]]

            if not prices.empty:
                sector_df, _ = compare_sector_models_on_stocks(prices.dropna(how="all"), stock_pool)
                if sector_df.empty:
                    st.warning("Not enough data to evaluate sector models on sector-page stocks.")
                else:
                    best_sector = sector_df.sort_values(
                        ["Direction Accuracy %", "Rank IC", "Top Picks Avg Return %"],
                        ascending=False,
                    ).iloc[0]
                    st.markdown(
                        f"""
<div class='card winner'>
<b>Recommended Sector Model:</b> {best_sector['Sector Model']}<br>
<b>Expected Performance:</b> {performance_label(float(best_sector['Top Picks Avg Return %']))} ({best_sector['Top Picks Avg Return %']:.2f}%)<br>
<b>Prediction Reliability:</b> {accuracy_label(float(best_sector['Direction Accuracy %']))} ({best_sector['Direction Accuracy %']:.2f}%)
</div>
""",
                        unsafe_allow_html=True,
                    )
                    simple_sector_df = sector_df.sort_values(
                        ["Direction Accuracy %", "Top Picks Avg Return %"],
                        ascending=False,
                    ).copy()
                    simple_sector_df["Performance"] = simple_sector_df["Top Picks Avg Return %"].apply(performance_label)
                    simple_sector_df["Reliability"] = simple_sector_df["Direction Accuracy %"].apply(accuracy_label)
                    simple_sector_df = simple_sector_df[
                        [
                            "Sector Model",
                            "Performance",
                            "Reliability",
                            "Top Picks Avg Return %",
                            "Direction Accuracy %",
                        ]
                    ]
                    simple_sector_df["Top Picks Avg Return %"] = simple_sector_df["Top Picks Avg Return %"].map(lambda x: f"{x:.2f}%")
                    simple_sector_df["Direction Accuracy %"] = simple_sector_df["Direction Accuracy %"].map(lambda x: f"{x:.2f}%")
                    st.table(simple_sector_df.reset_index(drop=True))

                    if px is not None:
                        fig_sector = px.bar(
                            sector_df.sort_values(["Direction Accuracy %", "Top Picks Avg Return %"], ascending=False),
                            x="Sector Model",
                            y=["Direction Accuracy %", "Top Picks Avg Return %"],
                            barmode="group",
                            title="Sector Model Comparison",
                        )
                        st.plotly_chart(style_fig(fig_sector), use_container_width=True)
else:
    st.info("Set parameters and click **Run Model Comparison**.")
