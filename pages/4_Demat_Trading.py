import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from theme import apply_shared_theme, render_page_hero
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None

st.set_page_config(page_title="Demat Trading", layout="wide")

st.markdown(
    """
<style>
.stApp {
    background: radial-gradient(circle at top left, #1c2433 0%, #0f172a 42%, #090f1a 100%);
    color: #e2e8f0;
}
.hero {
    background: linear-gradient(120deg, #101a2a, #1f2a3f);
    color: #f8fafc;
    border: 1px solid #3b475f;
    border-radius: 16px;
    padding: 18px 20px;
    margin-bottom: 12px;
}
.metric {
    background: linear-gradient(180deg, #0f172a, #0b1322);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 12px;
}
.glass-panel {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, rgba(10, 20, 45, 0.9), rgba(3, 10, 28, 0.82));
    border: 1px solid rgba(96, 165, 250, 0.26);
    border-radius: 24px;
    padding: 22px;
    margin: 12px 0 18px 0;
    box-shadow: 0 22px 60px rgba(2, 6, 23, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(18px);
}
.glass-panel::before {
    content: "";
    position: absolute;
    inset: -20% auto auto -10%;
    width: 280px;
    height: 280px;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.18), transparent 68%);
    pointer-events: none;
}
.glass-panel::after {
    content: "";
    position: absolute;
    right: -40px;
    bottom: -60px;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(34, 197, 94, 0.12), transparent 70%);
    pointer-events: none;
}
.glass-grid {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 16px;
}
.glass-tile {
    min-height: 132px;
    padding: 18px 18px 16px 18px;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.72), rgba(2, 6, 23, 0.6));
    border: 1px solid rgba(148, 163, 184, 0.16);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 12px;
    min-width: 0;
}
.glass-tile-label {
    font-size: 0.92rem;
    color: #bfdbfe;
    letter-spacing: 0.02em;
    margin-bottom: 10px;
}
.glass-tile-value {
    font-size: clamp(1.7rem, 2vw, 2.1rem);
    font-weight: 700;
    line-height: 1.05;
    color: #f8fafc;
    letter-spacing: -0.03em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-variant-numeric: tabular-nums lining-nums;
}
.glass-tile-value.compact {
    font-size: clamp(1.4rem, 1.6vw, 1.8rem);
}
.glass-tile-value.tight {
    font-size: clamp(1.15rem, 1.25vw, 1.45rem);
}
.glass-tile-delta {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 0.95rem;
    font-weight: 600;
    width: fit-content;
    max-width: 100%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.delta-pos {
    color: #4ade80;
    background: rgba(34, 197, 94, 0.18);
}
.delta-neg {
    color: #fda4af;
    background: rgba(244, 63, 94, 0.16);
}
.delta-flat {
    color: #cbd5e1;
    background: rgba(148, 163, 184, 0.14);
}
.signal-card {
    position: relative;
    z-index: 1;
    margin-top: 18px;
    padding: 22px;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(7, 18, 39, 0.88), rgba(2, 6, 23, 0.72));
    border: 1px solid rgba(96, 165, 250, 0.2);
}
.signal-card-buy {
    box-shadow: inset 4px 0 0 #22c55e;
}
.signal-card-sell {
    box-shadow: inset 4px 0 0 #ef4444;
}
.signal-card-hold {
    box-shadow: inset 4px 0 0 #f59e0b;
}
.signal-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 18px;
}
.signal-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f8fafc;
}
.signal-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
}
.signal-item {
    padding: 14px 16px;
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.54);
    border: 1px solid rgba(148, 163, 184, 0.12);
}
.signal-item-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.signal-item-value {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
}
.status-banner {
    background: linear-gradient(90deg, rgba(34, 197, 94, 0.22), rgba(34, 197, 94, 0.12));
    border: 1px solid rgba(74, 222, 128, 0.16);
    border-radius: 18px;
    color: #4ade80;
    padding: 16px 20px;
    margin: 8px 0 14px 0;
    font-weight: 700;
}
@media (max-width: 1100px) {
    .glass-grid,
    .signal-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
@media (max-width: 720px) {
    .glass-panel {
        padding: 16px;
        border-radius: 18px;
    }
    .glass-grid,
    .signal-grid {
        grid-template-columns: 1fr;
    }
    .glass-tile-value {
        font-size: 1.7rem;
    }
    .signal-head {
        align-items: flex-start;
        flex-direction: column;
    }
}
.terminal-card {
    background: linear-gradient(180deg, #0f172a, #0b1322);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 14px;
}
.order-card {
    background: linear-gradient(180deg, #111827, #0b1220);
    border: 1px solid #475569;
    border-top: 4px solid #ff6b00;
    border-radius: 14px;
    padding: 14px;
}
.chip {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.78rem;
    background: #1f2937;
    border: 1px solid #374151;
    color: #e5e7eb;
}
.signal-buy {
    border-left: 5px solid #22c55e;
}
.signal-sell {
    border-left: 5px solid #ef4444;
}
.signal-hold {
    border-left: 5px solid #f59e0b;
}
</style>
""",
    unsafe_allow_html=True,
)
apply_shared_theme()

render_page_hero(
    "Demat Trading",
    "Broker-style order ticket with live quotes and model suggestion.",
)

MODEL_PATH = "models"
MODEL_MAP = {"PPO": PPO, "A2C": A2C, "DDPG": DDPG, "SAC": SAC, "TD3": TD3}
INITIAL_DEMAT_CASH = 100000.0
RANGE_MAP = {
    "1D": ("1d", "1m", "1"),
    "5D": ("5d", "5m", "5"),
    "1M": ("1mo", "30m", "30"),
    "3M": ("3mo", "60m", "60"),
    "YTD": ("ytd", "1d", "D"),
    "1Y": ("1y", "1d", "D"),
}


def _to_scalar(action) -> float:
    return float(np.squeeze(action))


def _extract_close(data: pd.DataFrame) -> pd.Series:
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


def _extract_ohlc(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    if isinstance(data.columns, pd.MultiIndex):
        cols = {}
        for name in ["Open", "High", "Low", "Close"]:
            if name in data.columns.get_level_values(0):
                part = data[name]
                cols[name] = part.iloc[:, 0] if isinstance(part, pd.DataFrame) else part
        if len(cols) == 4:
            return pd.DataFrame(cols).dropna()
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    req = ["Open", "High", "Low", "Close"]
    if any(c not in data.columns for c in req):
        return pd.DataFrame(columns=req)
    return data[req].dropna().copy()


@st.cache_resource
def load_model(algo: str):
    path = os.path.join(MODEL_PATH, f"{algo.lower()}_model.zip")
    if not os.path.exists(path):
        return None
    return MODEL_MAP[algo].load(path)


@st.cache_data(ttl=15)
def fetch_live_quote(ticker: str) -> Tuple[float, pd.DataFrame, Dict[str, str]]:
    data = yf.download(
        ticker,
        period="1d",
        interval="1m",
        auto_adjust=True,
        prepost=True,
        progress=False,
    )
    close = _extract_close(data)
    intraday = pd.DataFrame({"Close": close}) if not close.empty else pd.DataFrame(columns=["Close"])

    price_candidates = []
    source = "intraday_1m"
    previous_close = np.nan

    try:
        tk = yf.Ticker(ticker)
        fast = tk.fast_info
        if fast is not None:
            for key in ("lastPrice", "last_price", "regularMarketPrice"):
                v = fast.get(key)
                if v is not None and np.isfinite(float(v)) and float(v) > 0:
                    price_candidates.append(float(v))
                    source = f"fast_info:{key}"
                    break
            pc = fast.get("previousClose")
            if pc is not None and np.isfinite(float(pc)) and float(pc) > 0:
                previous_close = float(pc)
    except Exception:
        pass

    if not close.empty and np.isfinite(float(close.iloc[-1])) and float(close.iloc[-1]) > 0:
        price_candidates.append(float(close.iloc[-1]))

    live_price = price_candidates[0] if price_candidates else np.nan
    if not np.isfinite(previous_close):
        # Robust fallback for day-change math: use daily history previous close, not intraday first tick.
        daily = yf.download(ticker, period="7d", interval="1d", auto_adjust=True, progress=False)
        daily_close = _extract_close(daily)
        if len(daily_close) >= 2:
            previous_close = float(daily_close.iloc[-2])
        elif len(daily_close) == 1:
            previous_close = float(daily_close.iloc[-1])
        elif np.isfinite(live_price):
            previous_close = float(live_price)

    day_change_pct = np.nan
    if np.isfinite(live_price) and np.isfinite(previous_close) and previous_close > 0:
        day_change_pct = float((live_price / previous_close - 1.0) * 100.0)

    quote_ts = ""
    if not close.empty:
        quote_ts = str(close.index[-1])

    meta = {
        "source": source,
        "quote_time": quote_ts,
        "day_change_pct": f"{day_change_pct:.4f}" if np.isfinite(day_change_pct) else "",
        "previous_close": f"{previous_close:.4f}" if np.isfinite(previous_close) else "",
    }
    return live_price, intraday, meta


@st.cache_data(ttl=20)
def fetch_chart_ohlc(ticker: str, range_key: str) -> pd.DataFrame:
    period, interval, _ = RANGE_MAP.get(range_key, RANGE_MAP["1D"])
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        prepost=True,
        progress=False,
    )
    return _extract_ohlc(data)


def tradingview_symbol(ticker: str) -> str:
    t = ticker.upper().strip()
    if not t:
        return "NASDAQ:AAPL"
    # Simple default mapping for US symbols.
    return f"NASDAQ:{t}"


def tradingview_widget_html(symbol: str, interval: str, height_px: int) -> str:
    inner_h = max(420, int(height_px) - 20)
    return f"""
    <div class="tradingview-widget-container" style="height:{inner_h}px;width:100%">
      <div id="tv_chart" style="height:{inner_h}px;width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": false,
        "width": "100%",
        "height": {inner_h},
        "symbol": "{symbol}",
        "interval": "{interval}",
        "timezone": "Asia/Kolkata",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "hide_side_toolbar": false,
        "allow_symbol_change": false,
        "withdateranges": true,
        "container_id": "tv_chart"
      }});
      </script>
    </div>
    """


@st.cache_data(ttl=120)
def fetch_signal_frame(ticker: str) -> pd.DataFrame:
    data = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)
    close = _extract_close(data)
    if close.empty:
        return pd.DataFrame(columns=["Close", "Return"])
    out = pd.DataFrame({"Close": close})
    out["Return"] = out["Close"].pct_change()
    return out.dropna().copy()


def _position_from_action(action_val: float, long_only: bool, strength: float) -> float:
    raw = action_val * strength
    if long_only:
        return float(np.clip(raw, 0.0, 1.0))
    return float(np.clip(raw, -1.0, 1.0))


def model_signal(
    model,
    signal_df: pd.DataFrame,
    long_only: bool,
    strength: float,
) -> Tuple[str, float, float]:
    if model is None or signal_df.empty or len(signal_df) < 80:
        return "HOLD", 0.0, 0.0

    returns = signal_df["Return"].to_numpy(dtype=np.float64)
    closes = signal_df["Close"]

    eval_window = min(40, len(returns) - 1)
    pos_hist = []
    correct = 0
    total = 0
    start_idx = len(returns) - eval_window

    for i in range(start_idx, len(returns)):
        obs = np.array([returns[i - 1]], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        pos = _position_from_action(_to_scalar(action), long_only=long_only, strength=strength)
        pos_hist.append(pos)

        if long_only:
            pred = 1 if pos > 0.15 else 0
            actual = 1 if returns[i] > 0 else 0
        else:
            pred = 1 if pos > 0.12 else (-1 if pos < -0.12 else 0)
            actual = 1 if returns[i] > 0 else (-1 if returns[i] < 0 else 0)
        correct += int(pred == actual)
        total += 1

    rl_edge = float(np.mean(pos_hist[-8:])) if pos_hist else 0.0
    hit_rate = (correct / total) if total > 0 else 0.0

    ma_fast = closes.rolling(10).mean().iloc[-1]
    ma_slow = closes.rolling(30).mean().iloc[-1]
    trend_spread = 0.0 if (pd.isna(ma_fast) or pd.isna(ma_slow) or ma_slow == 0) else float(ma_fast / ma_slow - 1.0)
    momentum_5d = float(closes.pct_change(5).iloc[-1]) if len(closes) > 6 else 0.0
    trend_score = np.tanh(trend_spread * 25.0) + 0.6 * np.tanh(momentum_5d * 12.0)

    vol20 = float(signal_df["Return"].tail(20).std()) if len(signal_df) >= 20 else 0.0
    vol60 = float(signal_df["Return"].tail(60).std()) if len(signal_df) >= 60 else vol20
    vol_penalty = float(np.clip((vol20 / max(vol60, 1e-8)) - 1.0, 0.0, 1.0))

    combined = (0.68 * rl_edge) + (0.32 * trend_score)
    if long_only:
        combined = max(0.0, combined)

    quality_boost = max(0.0, hit_rate - 0.5) * 1.2
    confidence = float(np.clip((abs(combined) * 0.75) + quality_boost - (0.25 * vol_penalty), 0.05, 0.99))

    if long_only:
        if combined > 0.16:
            signal = "BUY"
        else:
            signal = "HOLD"
    else:
        if combined > 0.12:
            signal = "BUY"
        elif combined < -0.12:
            signal = "SELL"
        else:
            signal = "HOLD"

    return signal, confidence, hit_rate


def _delta_class(value: float) -> str:
    if not np.isfinite(value):
        return "delta-flat"
    if value > 0:
        return "delta-pos"
    if value < 0:
        return "delta-neg"
    return "delta-flat"


def _value_size_class(value: str) -> str:
    length = len(value)
    if length >= 14:
        return "tight"
    if length >= 10:
        return "compact"
    return ""


def init_state():
    if "demat_cash" not in st.session_state:
        st.session_state["demat_cash"] = INITIAL_DEMAT_CASH
    if "demat_positions" not in st.session_state:
        st.session_state["demat_positions"] = {}
    if "demat_trades" not in st.session_state:
        st.session_state["demat_trades"] = []
    if "demat_realized_pnl" not in st.session_state:
        st.session_state["demat_realized_pnl"] = 0.0


def reset_demat_account():
    st.session_state["demat_cash"] = INITIAL_DEMAT_CASH
    st.session_state["demat_positions"] = {}
    st.session_state["demat_trades"] = []
    st.session_state["demat_realized_pnl"] = 0.0
    st.session_state["demat_order_qty"] = 1
    st.session_state.pop("demat_order_notice", None)


def execute_order(
    side: str,
    ticker: str,
    qty: int,
    price: float,
    fee_bps: float,
    slippage_bps: float,
    signal: str,
    confidence: float,
):
    if qty <= 0:
        st.warning("Quantity must be greater than 0.")
        return
    if not np.isfinite(price) or price <= 0:
        st.error("Invalid live price. Refresh quote and try again.")
        return

    positions: Dict[str, Dict[str, float]] = st.session_state["demat_positions"]
    cash = float(st.session_state["demat_cash"])
    realized_pnl = 0.0

    notional = qty * price
    cost_rate = (fee_bps + slippage_bps) / 10000.0
    cost = notional * cost_rate

    if side == "BUY":
        total = notional + cost
        if total > cash:
            st.error("Not enough cash for this buy order.")
            return
        current = positions.get(ticker, {"qty": 0, "avg_price": 0.0})
        new_qty = int(current["qty"]) + qty
        prev_notional = float(current["qty"]) * float(current["avg_price"])
        avg_price = (prev_notional + total) / new_qty
        positions[ticker] = {"qty": new_qty, "avg_price": avg_price}
        st.session_state["demat_cash"] = cash - total
    else:
        current = positions.get(ticker, {"qty": 0, "avg_price": 0.0})
        holding = int(current["qty"])
        if qty > holding:
            st.error("Not enough quantity to sell.")
            return
        proceeds = notional - cost
        avg_price = float(current["avg_price"])
        realized_pnl = proceeds - (avg_price * qty)
        new_qty = holding - qty
        if new_qty == 0:
            positions.pop(ticker, None)
        else:
            positions[ticker]["qty"] = new_qty
        st.session_state["demat_cash"] = cash + proceeds
        st.session_state["demat_realized_pnl"] = float(st.session_state["demat_realized_pnl"]) + realized_pnl

    st.session_state["demat_trades"].append(
        {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Ticker": ticker,
            "Side": side,
            "Qty": qty,
            "Price": round(price, 4),
            "Notional": round(notional, 2),
            "Cost": round(cost, 2),
            "Realized PnL": round(realized_pnl, 2),
            "Model Signal": signal,
            "Confidence": round(confidence, 3),
        }
    )
    notice = f"{side} order executed: {qty} {ticker} @ ${price:.2f}"
    if side == "SELL":
        notice += f" | Realized PnL ${realized_pnl:,.2f}"
    st.session_state["demat_order_notice"] = notice
    st.rerun()


init_state()

st.sidebar.header("Trade Setup")
default_ticker = st.session_state.get("app_selected_ticker", "AAPL")
ticker = st.sidebar.text_input("Ticker", value=(default_ticker or "AAPL")).strip().upper()
st.session_state["app_selected_ticker"] = ticker or "AAPL"
algo = st.sidebar.selectbox("RL Model", list(MODEL_MAP.keys()))
long_only = st.sidebar.checkbox("Long Only", value=True)
signal_strength = st.sidebar.slider("Signal Strength", 0.5, 3.0, 1.5, 0.1)
fee_bps = st.sidebar.number_input("Fee (bps)", min_value=0.0, max_value=100.0, value=2.0, step=1.0)
slippage_bps = st.sidebar.number_input("Slippage (bps)", min_value=0.0, max_value=100.0, value=2.0, step=1.0)

if st.sidebar.button("Refresh Live Data"):
    fetch_live_quote.clear()
    fetch_signal_frame.clear()

if st.sidebar.button("Reset Demat Account"):
    reset_demat_account()
    fetch_live_quote.clear()
    fetch_signal_frame.clear()
    fetch_chart_ohlc.clear()
    st.rerun()

model = load_model(algo)
if model is None:
    st.error(f"Model file missing: {MODEL_PATH}/{algo.lower()}_model.zip")
    st.stop()

live_price, intraday_df, quote_meta = fetch_live_quote(ticker)
signal_df = fetch_signal_frame(ticker)
signal, confidence, hit_rate = model_signal(model, signal_df, long_only=long_only, strength=signal_strength)

order_notice = st.session_state.pop("demat_order_notice", None)
if order_notice:
    st.markdown(f'<div class="status-banner">{order_notice}</div>', unsafe_allow_html=True)

cash = float(st.session_state["demat_cash"])
positions = st.session_state["demat_positions"]
holding = positions.get(ticker, {"qty": 0, "avg_price": 0.0})
holding_qty = int(holding["qty"])
holding_value = holding_qty * live_price if np.isfinite(live_price) else 0.0
holding_live_pnl = ((live_price - float(holding["avg_price"])) * holding_qty) if np.isfinite(live_price) else 0.0
suggested_qty = 0
if np.isfinite(live_price) and live_price > 0:
    budget = cash * min(0.25, max(0.03, confidence / 4))
    suggested_qty = max(1, int(budget / live_price))

total_holdings_value = 0.0
total_unrealized_pnl = 0.0
for tk, p in positions.items():
    px_now, _, _ = fetch_live_quote(tk)
    if np.isfinite(px_now):
        total_holdings_value += int(p["qty"]) * float(px_now)
        total_unrealized_pnl += (float(px_now) - float(p["avg_price"])) * int(p["qty"])
portfolio_value = cash + total_holdings_value
realized_pnl = float(st.session_state["demat_realized_pnl"])
total_pnl = portfolio_value - INITIAL_DEMAT_CASH

signal_class = "signal-hold"
if signal == "BUY":
    signal_class = "signal-buy"
elif signal == "SELL":
    signal_class = "signal-sell"

signal_card_class = "signal-card-hold"
if signal == "BUY":
    signal_card_class = "signal-card-buy"
elif signal == "SELL":
    signal_card_class = "signal-card-sell"

live_price_text = f"${live_price:,.2f}" if np.isfinite(live_price) else "N/A"
cash_text = f"${cash:,.2f}"
holdings_text = f"${total_holdings_value:,.2f}"
portfolio_text = f"${portfolio_value:,.2f}"
live_price_size = _value_size_class(live_price_text)
cash_size = _value_size_class(cash_text)
holdings_size = _value_size_class(holdings_text)
portfolio_size = _value_size_class(portfolio_text)
unrealized_size = _value_size_class(f"{total_unrealized_pnl:+,.2f}")
realized_size = _value_size_class(f"{realized_pnl:+,.2f}")
total_pnl_size = _value_size_class(f"{total_pnl:+,.2f}")
holding_size = _value_size_class(f"{holding_qty} shares")

stats_html = f"""
<div class="glass-panel">
  <div class="glass-grid">
    <div class="glass-tile">
      <div class="glass-tile-label">Live Price</div>
      <div class="glass-tile-value {live_price_size}">{live_price_text}</div>
    </div>
    <div class="glass-tile">
      <div class="glass-tile-label">Cash</div>
      <div class="glass-tile-value {cash_size}">{cash_text}</div>
    </div>
    <div class="glass-tile">
      <div class="glass-tile-label">Holdings Value</div>
      <div class="glass-tile-value {holdings_size}">{holdings_text}</div>
      <div class="glass-tile-delta {_delta_class(total_unrealized_pnl)}">{total_unrealized_pnl:+,.2f} U-PnL</div>
    </div>
    <div class="glass-tile">
      <div class="glass-tile-label">Portfolio Value</div>
      <div class="glass-tile-value {portfolio_size}">{portfolio_text}</div>
      <div class="glass-tile-delta {_delta_class(total_pnl)}">{total_pnl:+,.2f} Total PnL</div>
    </div>
    <div class="glass-tile">
      <div class="glass-tile-label">Unrealized PnL</div>
      <div class="glass-tile-value {unrealized_size}">{total_unrealized_pnl:+,.2f}</div>
    </div>
    <div class="glass-tile">
      <div class="glass-tile-label">Realized PnL</div>
      <div class="glass-tile-value {realized_size}">{realized_pnl:+,.2f}</div>
    </div>
    <div class="glass-tile">
      <div class="glass-tile-label">Net PnL</div>
      <div class="glass-tile-value {total_pnl_size}">{total_pnl:+,.2f}</div>
    </div>
    <div class="glass-tile">
      <div class="glass-tile-label">Current Holding</div>
      <div class="glass-tile-value {holding_size}">{holding_qty} shares</div>
      <div class="glass-tile-delta {_delta_class(holding_live_pnl)}">{holding_live_pnl:+,.2f} {ticker} Live</div>
    </div>
  </div>

  <div class="signal-card {signal_card_class}">
    <div class="signal-head">
      <div class="signal-title">Model Suggestion ({algo})</div>
      <span class="chip">{ticker}</span>
    </div>
    <div class="signal-grid">
      <div class="signal-item">
        <div class="signal-item-label">Action</div>
        <div class="signal-item-value">{signal}</div>
      </div>
      <div class="signal-item">
        <div class="signal-item-label">Confidence</div>
        <div class="signal-item-value">{confidence:.2f}</div>
      </div>
      <div class="signal-item">
        <div class="signal-item-label">Recent Hit Rate</div>
        <div class="signal-item-value">{hit_rate * 100.0:.1f}%</div>
      </div>
      <div class="signal-item">
        <div class="signal-item-label">Suggested Quantity</div>
        <div class="signal-item-value">{suggested_qty}</div>
      </div>
      <div class="signal-item">
        <div class="signal-item-label">Average Price</div>
        <div class="signal-item-value">${holding['avg_price']:.2f}</div>
      </div>
      <div class="signal-item">
        <div class="signal-item-label">Live PnL</div>
        <div class="signal-item-value">{holding_live_pnl:+,.2f}</div>
      </div>
    </div>
  </div>
</div>
"""
st.markdown(stats_html, unsafe_allow_html=True)
if quote_meta.get("quote_time"):
    day_change_text = quote_meta.get("day_change_pct", "")
    day_change_line = f" | Day change: {float(day_change_text):.2f}%" if day_change_text else ""
    st.caption(
        f"Live quote source: {quote_meta.get('source', 'unknown')} | Last tick: {quote_meta.get('quote_time')}{day_change_line}"
    )

if st.session_state.get("demat_order_ticker") != ticker:
    st.session_state["demat_order_ticker"] = ticker
    st.session_state["demat_order_qty"] = max(1, suggested_qty)
if "demat_order_qty" not in st.session_state:
    st.session_state["demat_order_qty"] = 1

st.markdown("### 1) Live Chart")
day_change = np.nan
if quote_meta.get("day_change_pct"):
    day_change = float(quote_meta["day_change_pct"])
day_change_abs = np.nan
if quote_meta.get("previous_close") and np.isfinite(live_price):
    prev_close = float(quote_meta["previous_close"])
    if np.isfinite(prev_close):
        day_change_abs = live_price - prev_close
st.markdown(
    f"## ${live_price:,.2f}  \n"
    f"<span style='color:{'#22c55e' if (np.isfinite(day_change) and day_change >= 0) else '#ef4444'};'>"
    f"{day_change_abs:+.2f} ({day_change:+.2f}%)</span>",
    unsafe_allow_html=True,
)

chart_ctrl_1, chart_ctrl_2 = st.columns([1.2, 1.0])
with chart_ctrl_1:
    selected_range = st.radio(
        "Range",
        list(RANGE_MAP.keys()),
        index=0,
        horizontal=True,
        key="demat_chart_range",
        label_visibility="collapsed",
    )
with chart_ctrl_2:
    use_tv = st.toggle("Use TradingView live widget", value=True)
chart_height = st.slider("Chart Height", min_value=520, max_value=980, value=760, step=40)

_, _, tv_interval = RANGE_MAP[selected_range]
if use_tv:
    components.html(
        tradingview_widget_html(tradingview_symbol(ticker), tv_interval, chart_height),
        height=chart_height + 24,
    )
else:
    chart_df = fetch_chart_ohlc(ticker, selected_range)
    if chart_df.empty:
        st.warning("No OHLC data available for selected range.")
    elif go is None:
        st.line_chart(chart_df[["Close"]])
    else:
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=chart_df.index,
                    open=chart_df["Open"],
                    high=chart_df["High"],
                    low=chart_df["Low"],
                    close=chart_df["Close"],
                    increasing_line_color="#14b8a6",
                    decreasing_line_color="#f43f5e",
                )
            ]
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#020617",
            plot_bgcolor="#020617",
            xaxis_rangeslider_visible=False,
            height=chart_height,
            margin=dict(t=10, l=10, r=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
st.caption("Original data from Yahoo Finance. TradingView widget requires internet access in your browser.")

st.markdown("### 2) Trade Desk")
desk_left, desk_right = st.columns([1.0, 1.4])

with desk_left:
    st.markdown('<div class="order-card">', unsafe_allow_html=True)
    st.markdown("#### Order Ticket")
    st.markdown(f"**Symbol:** `{ticker}`  |  **LTP:** `{f'${live_price:.2f}' if np.isfinite(live_price) else 'N/A'}`")
    st.number_input("Order Quantity", min_value=1, step=1, key="demat_order_qty")
    order_qty = int(st.session_state["demat_order_qty"])
    order_cols = st.columns(2)
    if order_cols[0].button("BUY", use_container_width=True):
        execute_order("BUY", ticker, order_qty, live_price, fee_bps, slippage_bps, signal, confidence)
    if order_cols[1].button("SELL", use_container_width=True):
        execute_order("SELL", ticker, order_qty, live_price, fee_bps, slippage_bps, signal, confidence)
    quick_cols = st.columns(2)
    if quick_cols[0].button("Buy Suggested", use_container_width=True):
        execute_order("BUY", ticker, suggested_qty, live_price, fee_bps, slippage_bps, signal, confidence)
    if quick_cols[1].button("Sell Suggested", use_container_width=True):
        execute_order("SELL", ticker, min(suggested_qty, holding_qty), live_price, fee_bps, slippage_bps, signal, confidence)
    st.caption("Quantity is fully user-controlled in the order ticket.")
    st.markdown("</div>", unsafe_allow_html=True)

with desk_right:
    st.markdown("#### Market Watch")
    watchlist = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL"]
    for tk in positions.keys():
        if tk not in watchlist:
            watchlist.append(tk)
    rows = []
    for tk in watchlist:
        tk_price, _, tk_meta = fetch_live_quote(tk)
        tk_change = float(tk_meta["day_change_pct"]) if tk_meta.get("day_change_pct") else np.nan
        prev_close = float(tk_meta["previous_close"]) if tk_meta.get("previous_close") else np.nan
        day_abs = float(tk_price - prev_close) if np.isfinite(tk_price) and np.isfinite(prev_close) else np.nan
        rows.append(
            {
                "Ticker": tk,
                "LTP": round(float(tk_price), 2) if np.isfinite(tk_price) else np.nan,
                "Prev Close": round(float(prev_close), 2) if np.isfinite(prev_close) else np.nan,
                "Day Chg": round(float(day_abs), 2) if np.isfinite(day_abs) else np.nan,
                "Day %": round(float(tk_change), 2) if np.isfinite(tk_change) else np.nan,
                "Qty": int(positions.get(tk, {"qty": 0})["qty"]),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("### 3) Portfolio")
portfolio_tab1, portfolio_tab2 = st.tabs(["Open Positions", "Trade Book"])
with portfolio_tab1:
    if positions:
        rows = []
        for tk, p in positions.items():
            px_now, _, _ = fetch_live_quote(tk)
            qty = int(p["qty"])
            avg = float(p["avg_price"])
            mkt_val = qty * px_now if np.isfinite(px_now) else np.nan
            pnl = (px_now - avg) * qty if np.isfinite(px_now) else np.nan
            rows.append(
                {
                    "Ticker": tk,
                    "Qty": qty,
                    "Avg Price": round(avg, 4),
                    "Live Price": round(float(px_now), 4) if np.isfinite(px_now) else np.nan,
                    "Market Value": round(float(mkt_val), 2) if np.isfinite(mkt_val) else np.nan,
                    "Unrealized PnL": round(float(pnl), 2) if np.isfinite(pnl) else np.nan,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No open positions.")

with portfolio_tab2:
    if st.session_state["demat_trades"]:
        trades_df = pd.DataFrame(st.session_state["demat_trades"]).iloc[::-1].reset_index(drop=True)
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
    else:
        st.info("No trades executed yet.")
