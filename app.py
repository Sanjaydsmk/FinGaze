import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import os
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
try:
    import plotly.express as px
except ImportError:
    px = None

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="FinGaze : AI-Based Stock Portfolio Risk Classifier", layout="wide")

# =====================================
# CUSTOM CSS (🔥 UI MAGIC)
# =====================================
st.markdown("""
<style>
:root {
    --bg-0: #07111f;
    --bg-1: #0b1730;
    --text-main: #f8fafc;
    --text-soft: #cbd5e1;
}
.stApp {
    background:
        radial-gradient(circle at 15% 20%, rgba(34, 211, 238, 0.13), transparent 24%),
        radial-gradient(circle at 85% 18%, rgba(96, 165, 250, 0.16), transparent 28%),
        radial-gradient(circle at 60% 78%, rgba(74, 222, 128, 0.08), transparent 22%),
        linear-gradient(145deg, var(--bg-0) 0%, var(--bg-1) 44%, #050b16 100%);
    color: var(--text-main);
}
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 44px 44px;
    mask-image: radial-gradient(circle at center, black 48%, transparent 92%);
    opacity: 0.22;
}
.main .block-container {
    padding-top: 2.2rem;
    padding-bottom: 4rem;
    max-width: 1240px;
}
[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at top, rgba(96, 165, 250, 0.14), transparent 30%),
        linear-gradient(180deg, #08111f 0%, #0b1322 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.12);
}
.hero-shell {
    position: relative;
    overflow: hidden;
    padding: 32px 34px;
    border-radius: 28px;
    border: 1px solid rgba(125, 211, 252, 0.22);
    background: linear-gradient(120deg, rgba(9, 18, 34, 0.92), rgba(17, 36, 69, 0.78));
    box-shadow: 0 28px 80px rgba(2, 6, 23, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(18px);
    animation: heroRise 900ms ease-out;
}
.hero-shell::before {
    content: "";
    position: absolute;
    top: -80px;
    right: -50px;
    width: 260px;
    height: 260px;
    background: radial-gradient(circle, rgba(103, 232, 249, 0.22), transparent 70%);
}
.hero-shell::after {
    content: "";
    position: absolute;
    left: -50px;
    bottom: -90px;
    width: 280px;
    height: 280px;
    background: radial-gradient(circle, rgba(96, 165, 250, 0.16), transparent 70%);
}
.hero-grid {
    position: relative;
    z-index: 1;
    display: block;
}
.hero-title {
    margin: 0;
    font-size: clamp(2.4rem, 5vw, 4.3rem);
    line-height: 0.95;
    letter-spacing: -0.05em;
    font-weight: 800;
}
.hero-title span {
    display: block;
    font-size: 0.64em;
    line-height: 1.02;
    background: linear-gradient(90deg, #f8fafc 0%, #7dd3fc 48%, #4ade80 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-shell + div[data-testid="stButton"] {
    margin-top: 34px;
}
.stButton>button {
    background: linear-gradient(90deg, #0891b2, #2563eb 54%, #4f46e5 100%);
    color: white;
    border-radius: 14px;
    padding: 0.72em 1.6em;
    font-weight: 700;
    border: 1px solid rgba(125, 211, 252, 0.14);
    box-shadow: 0 12px 28px rgba(37, 99, 235, 0.24);
    transition: transform 220ms ease, box-shadow 220ms ease, filter 220ms ease;
}
.stButton>button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 18px 36px rgba(56, 189, 248, 0.28);
    filter: saturate(1.08);
}
.metric-card {
    position: relative;
    overflow: hidden;
    background: linear-gradient(180deg, rgba(9, 18, 36, 0.78), rgba(4, 10, 25, 0.9));
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(14px);
    text-align: center;
    min-height: 130px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    border: 1px solid rgba(148, 163, 184, 0.16);
    transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
    animation: fadeSlide 600ms ease-out;
}
.metric-card::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, rgba(103, 232, 249, 0.08), transparent 35%, rgba(74, 222, 128, 0.05));
    opacity: 0;
    transition: opacity 220ms ease;
}
.metric-card-large {
    padding: 34px 28px;
    border-radius: 24px;
    margin-bottom: 14px;
    min-height: 170px;
    text-align: left;
}
.metric-card:hover {
    transform: translateY(-7px) scale(1.01);
    border-color: rgba(56, 189, 248, 0.34);
    box-shadow: 0 18px 38px rgba(14, 165, 233, 0.16);
}
.metric-card:hover::before {
    opacity: 1;
}
.metric-card h4 {
    position: relative;
    margin: 0 0 10px 0;
    color: #cbd5e1;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.82rem;
}
.metric-card h2 {
    position: relative;
    margin: 0;
    font-size: clamp(1.8rem, 2.5vw, 2.7rem);
    line-height: 1;
    letter-spacing: -0.04em;
}
.metric-sub {
    position: relative;
    margin-top: 6px;
    font-size: 0.95rem;
    color: #93c5fd;
}
.metrics-row .metric-card h2 {
    font-size: clamp(1.55rem, 2vw, 2.15rem);
}
.metrics-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
    align-items: stretch;
    margin-bottom: 4px;
}
@media (max-width: 1100px) {
    .metrics-row {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
@media (max-width: 640px) {
    .metrics-row {
        grid-template-columns: 1fr;
    }
}
[data-testid="stPlotlyChart"] {
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 18px;
    padding: 8px;
    background: linear-gradient(180deg, rgba(8, 15, 30, 0.76), rgba(2, 6, 23, 0.94));
    transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
    animation: fadeSlide 650ms ease-out;
    box-shadow: 0 20px 40px rgba(2, 6, 23, 0.26);
}
[data-testid="stPlotlyChart"]:hover {
    transform: translateY(-4px) scale(1.006);
    border-color: rgba(56, 189, 248, 0.38);
    box-shadow: 0 22px 42px rgba(8, 47, 73, 0.45);
}
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes heroRise {
    from { opacity: 0; transform: translateY(22px) scale(0.985); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 12px;
    color: var(--text-main);
    letter-spacing: 0.01em;
}
.section-title::before {
    content: "";
    display: inline-block;
    width: 12px;
    height: 12px;
    margin-right: 10px;
    border-radius: 999px;
    background: linear-gradient(180deg, #67e8f9, #60a5fa);
    box-shadow: 0 0 14px rgba(103, 232, 249, 0.55);
}
@media (max-width: 640px) {
    .hero-shell {
        padding: 22px 20px;
        border-radius: 22px;
    }
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================
st.markdown(
    """
    <div class="hero-shell">
      <div class="hero-grid">
        <div>
          <div class="hero-title">FinGaze :<span>AI-Based Stock Portfolio Risk Classifier</span></div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================
# SIDEBAR
# =====================================
st.sidebar.header("⚙ Configuration")

ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
st.session_state["app_selected_ticker"] = ticker.strip().upper() if ticker else None

today = pd.Timestamp.today().date()
default_from = today - pd.Timedelta(days=365)
from_date = st.sidebar.date_input("From Date", value=default_from)
to_date = st.sidebar.date_input("To Date", value=today)
default_forecast_to = today + pd.Timedelta(days=60)
forecast_to_date = st.sidebar.date_input(
    "Forecast To Date (from today)",
    value=default_forecast_to,
    min_value=today + pd.Timedelta(days=1),
)

initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=1000,
    max_value=1000000,
    value=10000,
    step=1000
)

long_only = st.sidebar.checkbox("Long Only (for stocks)", value=True)
signal_strength = st.sidebar.slider("Signal Strength", 0.5, 3.0, 1.5, 0.1)
fee_bps = st.sidebar.number_input("Fee (bps)", min_value=0.0, max_value=100.0, value=2.0, step=1.0)
slippage_bps = st.sidebar.number_input("Slippage (bps)", min_value=0.0, max_value=100.0, value=2.0, step=1.0)

algo_choice = st.sidebar.selectbox(
    "Select Algorithm",
    ["PPO", "A2C", "DDPG", "SAC", "TD3"]
)

# =====================================
# MODEL LOADING
# =====================================
MODEL_PATH = "models"

model_map = {
    "PPO": PPO,
    "A2C": A2C,
    "DDPG": DDPG,
    "SAC": SAC,
    "TD3": TD3
}

if "model" not in st.session_state:
    st.session_state["model"] = None
if "last_return" not in st.session_state:
    st.session_state["last_return"] = None
if "recent_returns" not in st.session_state:
    st.session_state["recent_returns"] = None
if "app_backtest" not in st.session_state:
    st.session_state["app_backtest"] = None
if "app_forecast" not in st.session_state:
    st.session_state["app_forecast"] = None


def _to_scalar(action) -> float:
    return float(np.squeeze(action))


def load_model(algo):
    path = os.path.join(MODEL_PATH, f"{algo.lower()}_model.zip")
    if not os.path.exists(path):
        st.error("Model not found. Train first.")
        return None
    return model_map[algo].load(path)


def style_animated_plotly(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#020617",
        plot_bgcolor="#0f172a",
        transition={"duration": 900, "easing": "cubic-in-out"},
        hoverlabel={"bgcolor": "#0f172a", "font_size": 13, "font_color": "#f8fafc"},
        margin=dict(t=45, l=20, r=20, b=20),
    )
    return fig


def plot_line_chart(df_plot: pd.DataFrame, title: str, y_label: str):
    if px is None:
        st.line_chart(df_plot)
        return
    fig = px.line(
        df_plot.reset_index().rename(columns={"index": "Date"}),
        x="Date",
        y=list(df_plot.columns),
        title=title,
    )
    fig = style_animated_plotly(fig)
    fig.update_traces(line=dict(width=2.3))
    fig.update_layout(xaxis_title="Date", yaxis_title=y_label, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True, theme=None)

# =====================================
# BACKTEST
# =====================================
def backtest(
    model,
    returns,
    initial_capital,
    long_only=True,
    signal_strength=1.5,
    fee_bps=2.0,
    slippage_bps=2.0,
):

    capital = float(initial_capital)
    equity_curve = [capital]
    abs_position_sum = 0.0
    steps = 0
    current_position = 0.0
    total_cost = 0.0

    # No look-ahead bias:
    # action from return[t-1] is applied to return[t].
    for i in range(1, len(returns)):
        obs = np.array([returns[i - 1]], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        raw_position = _to_scalar(action) * signal_strength
        if long_only:
            position = float(np.clip(raw_position, 0.0, 1.0))
        else:
            position = float(np.clip(raw_position, -1.0, 1.0))

        turnover = abs(position - current_position)
        cost_rate = (fee_bps + slippage_bps) / 10000.0
        step_cost = capital * turnover * cost_rate
        total_cost += step_cost
        capital = max(0.0, capital - step_cost)
        current_position = position

        realized_return = float(returns[i])
        capital *= (1.0 + position * realized_return)
        capital = max(0.0, capital)
        equity_curve.append(capital)
        abs_position_sum += abs(position)
        steps += 1

    avg_abs_exposure = (abs_position_sum / steps) if steps else 0.0
    return equity_curve, avg_abs_exposure, total_cost


def forecast_future_returns(
    model,
    last_return,
    history_returns,
    initial_capital,
    horizon_days=30,
    n_sims=500,
    long_only=True,
    signal_strength=1.5,
    fee_bps=2.0,
    slippage_bps=2.0,
    lookback_days=252,
    block_size=5,
    conservatism=0.7,
):
    hist_full = np.asarray(history_returns, dtype=np.float64)
    if len(hist_full) > lookback_days:
        hist = hist_full[-lookback_days:]
    else:
        hist = hist_full

    if len(hist) < 20:
        raise ValueError("Need at least 20 return observations for forecasting.")

    low_q = float(np.quantile(hist, 0.025))
    high_q = float(np.quantile(hist, 0.975))
    hist_w = np.clip(hist, low_q, high_q)

    long_vol = max(float(np.std(hist_w)), 1e-6)
    recent_window = min(20, len(hist))
    recent_slice = hist_w[-recent_window:]
    recent_vol = max(float(np.std(recent_slice)), 1e-6)
    recent_mu = float(np.mean(recent_slice))
    long_mu = float(np.mean(hist_w))

    vol_scale = float(np.clip(recent_vol / long_vol, 0.7, 1.4))
    base_drift = 0.7 * long_mu + 0.3 * recent_mu
    risk_penalty = 0.5 * (long_vol ** 2)
    drift_target = base_drift - risk_penalty
    if drift_target > 0:
        drift_target *= (1.0 - conservatism)
    else:
        drift_target *= (1.0 - 0.25 * conservatism)

    all_paths = np.zeros((n_sims, horizon_days + 1), dtype=np.float64)
    all_paths[:, 0] = float(initial_capital)

    for sim_i in range(n_sims):
        capital = float(initial_capital)
        prev_r = float(last_return)
        current_position = 0.0

        sampled_returns = []
        while len(sampled_returns) < horizon_days:
            start_ix = np.random.randint(0, max(1, len(hist_w) - block_size + 1))
            sampled_returns.extend(hist_w[start_ix:start_ix + block_size].tolist())
        sampled_returns = np.asarray(sampled_returns[:horizon_days], dtype=np.float64)
        sampled_returns = sampled_returns * vol_scale
        sampled_returns += (drift_target - long_mu)
        sampled_returns = np.clip(sampled_returns, low_q, high_q)

        for day_i in range(horizon_days):
            obs = np.array([prev_r], dtype=np.float32)
            action, _ = model.predict(obs, deterministic=True)
            position_scale = 1.0 - (0.5 * conservatism)
            raw_position = _to_scalar(action) * signal_strength * position_scale
            if long_only:
                position = float(np.clip(raw_position, 0.0, 1.0))
            else:
                position = float(np.clip(raw_position, -1.0, 1.0))

            turnover = abs(position - current_position)
            cost_rate = (fee_bps + slippage_bps) / 10000.0
            step_cost = capital * turnover * cost_rate
            capital = max(0.0, capital - step_cost)
            current_position = position

            sampled_return = float(sampled_returns[day_i])
            capital *= (1.0 + position * sampled_return)
            capital = max(0.0, capital)
            prev_r = sampled_return
            all_paths[sim_i, day_i + 1] = capital

    return all_paths

# =====================================
# RUN SIMULATION
# =====================================
run_simulation = False

# =====================================
# PREDICTION
# =====================================
st.markdown("---")
run_simulation = st.button(" Run Trading Simulation")

if run_simulation:
    if from_date >= to_date:
        st.error("`To Date` must be after `From Date`.")
        st.stop()

    with st.spinner("Downloading market data..."):
        df = yf.download(ticker, start=from_date, end=to_date, progress=False)

    if df.empty:
        st.error("No data found.")
        st.stop()

    # yfinance can return either normal or MultiIndex columns; normalize to a single Close series.
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            st.error("Close price not found in downloaded data.")
            st.stop()
        close_part = df["Close"]
        if isinstance(close_part, pd.DataFrame):
            close_series = close_part.iloc[:, 0]
        else:
            close_series = close_part
    else:
        if "Close" not in df.columns:
            st.error("Close price not found in downloaded data.")
            st.stop()
        close_series = df["Close"]

    df = pd.DataFrame({"Close": close_series}).copy()
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)

    returns = df['Return'].values
    if len(returns) < 2:
        st.error("Not enough data points after preprocessing.")
        st.stop()

    model = load_model(algo_choice)
    if model is None:
        st.stop()

    equity, avg_abs_exposure, total_cost = backtest(
        model,
        returns,
        initial_capital,
        long_only=long_only,
        signal_strength=signal_strength,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )

    final_value = float(equity[-1])
    profit = final_value - initial_capital
    profit_pct = (profit / initial_capital) * 100
    fee_pct = (total_cost / initial_capital) * 100
    buy_hold_final = float(initial_capital * (df["Close"].iloc[-1] / df["Close"].iloc[0]))
    buy_hold_profit_pct = ((buy_hold_final - initial_capital) / initial_capital) * 100
    hold_profit = buy_hold_final - initial_capital

    pie_df = pd.DataFrame({
        "Metric": ["Fee %", "Net Return %", "Hold Return %"],
        "Amount": [abs(total_cost), abs(profit), abs(hold_profit)],
        "Pct": [abs(fee_pct), abs(profit_pct), abs(buy_hold_profit_pct)],
    })
    pie_df = pie_df[pie_df["Amount"] > 0]

    st.session_state["app_backtest"] = {
        "ticker": ticker,
        "from_date": from_date,
        "to_date": to_date,
        "final_value": final_value,
        "profit": profit,
        "profit_pct": profit_pct,
        "fee_pct": fee_pct,
        "total_cost": total_cost,
        "buy_hold_profit_pct": buy_hold_profit_pct,
        "hold_profit": hold_profit,
        "avg_abs_exposure": avg_abs_exposure,
        "equity_df": pd.DataFrame(equity, columns=["Portfolio Value"]),
        "price_df": df[["Close"]].copy(),
        "pie_df": pie_df.copy(),
    }
    st.session_state["app_forecast"] = None

    st.session_state["model"] = model
    st.session_state["last_return"] = float(returns[-1])
    st.session_state["recent_returns"] = returns
    st.success("Simulation completed. See structured sections below.")

backtest_result = st.session_state.get("app_backtest")
if backtest_result:
    st.markdown("<div class='section-title'>1) Backtest Summary</div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class="metric-card metric-card-large">
            <h4>Final Portfolio Value</h4>
            <h2>${backtest_result['final_value']:,.2f}</h2>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="metrics-row">
            <div class="metric-card">
                <h4>Fee %</h4>
                <h2>{backtest_result['fee_pct']:.2f}%</h2>
                <div class="metric-sub">${backtest_result['total_cost']:,.2f}</div>
            </div>
            <div class="metric-card">
                <h4>Net Return %</h4>
                <h2>{backtest_result['profit_pct']:.2f}%</h2>
                <div class="metric-sub">${backtest_result['profit']:,.2f}</div>
            </div>
            <div class="metric-card">
                <h4>Hold Return %</h4>
                <h2>{backtest_result['buy_hold_profit_pct']:.2f}%</h2>
                <div class="metric-sub">${backtest_result['hold_profit']:,.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.caption(
        f"Range: {backtest_result['from_date']} to {backtest_result['to_date']}. "
        f"Average model exposure: {backtest_result['avg_abs_exposure']:.2f}x. "
        f"Costs: fee {fee_bps:.1f} bps + slippage {slippage_bps:.1f} bps."
    )

    tabs = st.tabs(["Performance Mix", "Backtest Charts"])
    with tabs[0]:
        pie_df = backtest_result["pie_df"]
        if px is None:
            st.warning("Install plotly to enable interactive pie chart: `pip install plotly`")
            st.bar_chart(pie_df.set_index("Metric"))
        else:
            fig = px.pie(
                pie_df,
                names="Metric",
                values="Amount",
                hole=0.62,
                color="Metric",
                color_discrete_sequence=["#38bdf8", "#2dd4bf", "#f59e0b"],
            )
            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                insidetextorientation="radial",
                pull=[0.04] * len(pie_df),
                customdata=pie_df[["Pct"]].to_numpy(),
                marker=dict(line=dict(color="rgba(15, 23, 42, 0.95)", width=3)),
                hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<br>%{customdata[0]:.2f}%<br>%{percent}<extra></extra>",
            )
            fig.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(color="#e2e8f0", size=13),
                ),
                margin=dict(t=40, l=10, r=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", size=14),
            )
            fig = style_animated_plotly(fig)
            st.plotly_chart(fig, use_container_width=True, theme=None)
    with tabs[1]:
        plot_line_chart(backtest_result["equity_df"], "Portfolio Equity Curve", "USD")
        plot_line_chart(backtest_result["price_df"], "Stock Price", "Price")

st.markdown("<div class='section-title'>2) Future Return Forecast</div>", unsafe_allow_html=True)

simulations = st.slider("Monte Carlo Simulations", 200, 2000, 500, 100)
lookback_days = st.slider("Lookback Window (trading days)", 60, 756, 252, 21)
block_size = st.slider("Bootstrap Block Size", 1, 20, 5, 1)
conservatism_pct = st.slider("Conservatism Level (%)", 0, 95, 70, 5)
conservatism = conservatism_pct / 100.0

future_dates = pd.bdate_range(
    start=pd.Timestamp(today) + pd.Timedelta(days=1),
    end=pd.Timestamp(forecast_to_date),
)
forecast_days = len(future_dates)

if (
    st.session_state["model"] is not None
    and st.session_state["last_return"] is not None
    and st.session_state["recent_returns"] is not None
):
    st.caption(
        f"Forecast start: {today + pd.Timedelta(days=1)} | "
        f"Forecast end: {forecast_to_date} | "
        f"Trading days: {forecast_days}"
    )

    if st.button("Predict Future Returns"):
        if forecast_days < 1:
            st.error("Forecast end date must be after today.")
            st.stop()

        model = st.session_state["model"]
        last_r = st.session_state["last_return"]
        history_returns = st.session_state["recent_returns"]

        with st.spinner("Running future return simulations..."):
            try:
                projected_paths = forecast_future_returns(
                    model=model,
                    last_return=last_r,
                    history_returns=history_returns,
                    initial_capital=initial_capital,
                    horizon_days=forecast_days,
                    n_sims=simulations,
                    long_only=long_only,
                    signal_strength=signal_strength,
                    fee_bps=fee_bps,
                    slippage_bps=slippage_bps,
                    lookback_days=lookback_days,
                    block_size=block_size,
                    conservatism=conservatism,
                )
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

        projected_values = projected_paths[:, -1]

        expected_final = float(np.mean(projected_values))
        conservative_final = float(np.percentile(projected_values, 40))
        expected_return_pct = ((expected_final / initial_capital) - 1.0) * 100.0
        conservative_return_pct = ((conservative_final / initial_capital) - 1.0) * 100.0
        p10 = float(np.percentile(projected_values, 10))
        p50 = float(np.percentile(projected_values, 50))
        p90 = float(np.percentile(projected_values, 90))
        positive_prob = float(np.mean(projected_values > initial_capital) * 100.0)

        path_p10 = np.percentile(projected_paths, 10, axis=0)
        path_p50 = np.percentile(projected_paths, 50, axis=0)
        path_p90 = np.percentile(projected_paths, 90, axis=0)
        path_mean = np.mean(projected_paths, axis=0)

        forecast_index = [pd.Timestamp(today)] + list(future_dates)
        forecast_df = pd.DataFrame(
            {
                "Expected": path_mean,
                "Median": path_p50,
                "P10": path_p10,
                "P90": path_p90,
            },
            index=forecast_index,
        )
        st.session_state["app_forecast"] = {
            "forecast_days": forecast_days,
            "simulations": simulations,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "expected_final": expected_final,
            "positive_prob": positive_prob,
            "conservatism_pct": conservatism_pct,
            "conservative_return_pct": conservative_return_pct,
            "median_return_pct": ((p50 / initial_capital) - 1.0) * 100.0,
            "expected_return_pct": expected_return_pct,
            "forecast_df": forecast_df.copy(),
        }
        st.success("Forecast completed. See structured forecast summary below.")

else:
    st.info("Run simulation first.")

forecast_result = st.session_state.get("app_forecast")
if forecast_result:
    st.markdown(
        f"""
        <div class="metrics-row">
            <div class="metric-card">
                <h4>Conservative Return %</h4>
                <h2>{forecast_result['conservative_return_pct']:.2f}%</h2>
            </div>
            <div class="metric-card">
                <h4>Median Return %</h4>
                <h2>{forecast_result['median_return_pct']:.2f}%</h2>
            </div>
            <div class="metric-card">
                <h4>Expected Return %</h4>
                <h2>{forecast_result['expected_return_pct']:.2f}%</h2>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption(
        f"{forecast_result['forecast_days']}-day forecast using {forecast_result['simulations']} simulations. "
        f"P10: ${forecast_result['p10']:,.2f} | Median: ${forecast_result['p50']:,.2f} | "
        f"P90: ${forecast_result['p90']:,.2f} | Expected: ${forecast_result['expected_final']:,.2f} | "
        f"Profit Probability: {forecast_result['positive_prob']:.1f}% | "
        f"Conservatism: {forecast_result['conservatism_pct']}%"
    )
    plot_line_chart(forecast_result["forecast_df"], "Future Portfolio Projection", "USD")
