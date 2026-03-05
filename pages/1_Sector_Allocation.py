import math
from datetime import timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
try:
    import plotly.express as px_plotly
except ImportError:
    px_plotly = None


st.set_page_config(page_title="Sector Allocation", layout="wide")
st.markdown(
    """
<style>
.stApp {
    background: radial-gradient(circle at 10% 0%, #0b1220 0%, #0f172a 40%, #020617 100%);
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #0b1220);
}
[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}
.hero-wrap {
    background: linear-gradient(125deg, #111827, #1e3a8a);
    color: #ffffff;
    border-radius: 18px;
    padding: 22px 24px;
    margin-bottom: 16px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.22);
    animation: fadeSlide 550ms ease-out;
}
.hero-wrap h2, .hero-wrap p {
    margin: 0;
}
.hero-wrap p {
    margin-top: 8px;
    color: #bfdbfe;
}
.card {
    background: linear-gradient(180deg, #0f172a, #111827);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 14px;
    min-height: 150px;
    box-shadow: 0 8px 18px rgba(2, 6, 23, 0.45);
    transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
    animation: fadeSlide 500ms ease-out, softFloat 5.5s ease-in-out infinite;
}
.card:hover {
    transform: translateY(-5px) scale(1.01);
    border-color: #38bdf8;
    box-shadow: 0 16px 26px rgba(14, 165, 233, 0.22);
}
.card-title {
    font-size: 1rem;
    font-weight: 700;
    color: #f8fafc;
}
.card-sub {
    margin-top: 4px;
    color: #93c5fd;
    font-size: 0.92rem;
}
.card-metric {
    margin-top: 8px;
    color: #cbd5e1;
    font-size: 0.89rem;
}
.pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.78rem;
    background: #1e293b;
    color: #7dd3fc;
    margin-bottom: 8px;
}
[data-testid="stPlotlyChart"] {
    border: 1px solid rgba(56, 189, 248, 0.18);
    border-radius: 14px;
    padding: 6px;
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.75), rgba(2, 6, 23, 0.9));
    transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
    animation: fadeSlide 650ms ease-out;
}
[data-testid="stPlotlyChart"]:hover {
    transform: translateY(-4px) scale(1.006);
    border-color: rgba(56, 189, 248, 0.55);
    box-shadow: 0 14px 28px rgba(8, 47, 73, 0.45);
}
.stButton > button {
    border-radius: 12px;
    border: 1px solid #334155;
    background: linear-gradient(90deg, #0ea5e9, #2563eb);
    color: #e0f2fe;
    transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 10px 18px rgba(14, 165, 233, 0.35);
    filter: brightness(1.05);
}
[data-testid="stExpander"] {
    border-radius: 12px;
    border: 1px solid #1f2937;
    overflow: hidden;
    animation: fadeSlide 450ms ease-out;
}
[data-testid="stExpander"]:hover {
    border-color: #334155;
}
[data-baseweb="select"] > div {
    transition: box-shadow 200ms ease, border-color 200ms ease, transform 200ms ease;
}
[data-baseweb="select"] > div:hover {
    box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.5);
    transform: translateY(-1px);
}
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes softFloat {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-2px); }
    100% { transform: translateY(0px); }
}
</style>
""",
    unsafe_allow_html=True,
)


# ==========================
# SECTOR UNIVERSE
# ==========================

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

MODEL_CHOICES = [
    "Model 1 – Momentum",
    "Model 2 – Momentum + Low Vol",
    "Model 3 – Equal Risk (Low Vol)",
    "Model 4 – Quality Uptrend",
    "Model 5 – Mixed Score",
]


def fmt_money(v: float) -> str:
    return f"${float(v):,.2f}"


def fmt_pct(v: float) -> str:
    return f"{float(v):.2f}%"


def render_stock_cards(
    df: pd.DataFrame,
    title: str,
    show_model_ticker: bool = False,
) -> None:
    st.markdown(f"### {title}")
    if df.empty:
        st.info("No items to display.")
        return
    sorted_df = df.sort_values("Sector").reset_index(drop=True)
    cols = st.columns(3)
    for idx, row in sorted_df.iterrows():
        col = cols[idx % 3]
        model_line = ""
        if show_model_ticker and "Model Ticker" in row:
            model_line = f"<div class='card-metric'><b>Model:</b> {row['Model Ticker']}</div>"
        with col:
            st.markdown(
                f"""
<div class="card">
  <div class="pill">{row['Sector']}</div>
  <div class="card-title">{row['Ticker']}</div>
  <div class="card-sub">Allocation {fmt_money(row['Allocation $'])}</div>
  {model_line}
  <div class="card-metric"><b>Profit:</b> {fmt_money(row['Profit $'])} ({fmt_pct(row['Profit %'])})</div>
  <div class="card-metric"><b>Ann Ret:</b> {fmt_pct(row['Ann. Return %'])} | <b>Vol:</b> {fmt_pct(row['Ann. Vol %'])}</div>
</div>
""",
                unsafe_allow_html=True,
            )


def plot_interactive_lines(df: pd.DataFrame, title: str, y_label: str) -> None:
    if px_plotly is not None:
        plot_df = df.reset_index().rename(columns={"index": "Date"})
        fig = px_plotly.line(plot_df, x="Date", y=list(df.columns), title=title, template="plotly_dark")
        fig.update_layout(
            height=320,
            paper_bgcolor="#020617",
            plot_bgcolor="#0f172a",
            legend_title_text="",
            xaxis_title="Date",
            yaxis_title=y_label,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        fig.update_traces(line=dict(width=2.2))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df)


def style_animated_plotly(fig, chart_kind: str = "bar"):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#020617",
        plot_bgcolor="#0f172a",
        transition={"duration": 850, "easing": "cubic-in-out"},
        hoverlabel={"bgcolor": "#111827", "font_size": 13, "font_color": "#e2e8f0"},
    )
    if chart_kind == "bar":
        fig.update_traces(
            marker_line_color="#38bdf8",
            marker_line_width=1.0,
            opacity=0.94,
            selector=dict(type="bar"),
        )
    elif chart_kind == "pie":
        fig.update_traces(
            pull=[0.02] * len(fig.data[0]["labels"]) if fig.data else None,
            marker=dict(line=dict(color="#0f172a", width=2)),
            selector=dict(type="pie"),
        )
    return fig


@dataclass
class StockStat:
    sector: str
    ticker: str
    ann_ret: float
    ann_vol: float
    score: float
    price: pd.Series


def download_prices(tickers: List[str], start, end) -> pd.DataFrame:
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        # (field, ticker)
        field = "Close" if "Close" in data.columns.get_level_values(0) else "Adj Close"
        close = data[field]
    else:
        close = data[["Close"]] if "Close" in data.columns else data.to_frame(name="Close")
        close.columns = [tickers[0]]

    close = close.dropna(how="all")
    return close


def annualized_stats(series: pd.Series) -> Tuple[float, float]:
    px = series.dropna()
    if len(px) < 50:
        return 0.0, 0.0
    rets = px.pct_change().dropna()
    if rets.empty:
        return 0.0, 0.0
    mean_daily = float(rets.mean())
    vol_daily = float(rets.std())
    ann_ret = ((1.0 + mean_daily) ** 252 - 1.0) * 100.0
    ann_vol = vol_daily * math.sqrt(252.0) * 100.0
    return ann_ret, ann_vol


def build_score(model_name: str, ann_ret: float, ann_vol: float) -> float:
    if ann_vol <= 0:
        return -1e9
    sharpe_like = ann_ret / ann_vol

    if model_name == MODEL_CHOICES[0]:  # pure momentum
        return ann_ret

    if model_name == MODEL_CHOICES[1]:  # momentum + low vol
        return ann_ret - 0.5 * ann_vol

    if model_name == MODEL_CHOICES[2]:  # equal risk (prefer low vol)
        return -ann_vol

    if model_name == MODEL_CHOICES[3]:  # quality: positive, stable
        return ann_ret - 0.3 * ann_vol if ann_ret > 0 else -1e9

    if model_name == MODEL_CHOICES[4]:  # mixed
        return 0.6 * ann_ret - 0.2 * ann_vol + 10.0 * sharpe_like

    return ann_ret


def normalize(d: Dict[str, float]) -> Dict[str, float]:
    vals = {k: max(0.0, float(v)) for k, v in d.items()}
    s = sum(vals.values())
    if s <= 0:
        n = len(vals)
        return {k: 1.0 / n for k in vals} if n else {}
    return {k: v / s for k, v in vals.items()}


def build_portfolio(
    initial_capital: float,
    from_date,
    to_date,
    model_name: str,
    selected_sectors: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sectors = [s for s in selected_sectors if s in SECTOR_TICKERS]
    all_tickers = sorted({t for sec in sectors for t in SECTOR_TICKERS[sec]})

    if not sectors or not all_tickers:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    prices = download_prices(all_tickers, start=from_date, end=to_date)
    if prices.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    stats: List[StockStat] = []

    for sector in sectors:
        for ticker in SECTOR_TICKERS[sector]:
            if ticker not in prices.columns:
                continue
            series = prices[ticker].dropna()
            ann_ret, ann_vol = annualized_stats(series)
            score = build_score(model_name, ann_ret, ann_vol)
            stats.append(
                StockStat(
                    sector=sector,
                    ticker=ticker,
                    ann_ret=ann_ret,
                    ann_vol=ann_vol,
                    score=score,
                    price=series,
                )
            )

    if not stats:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # pick best stock per sector
    best_per_sector: Dict[str, StockStat] = {}
    for s in stats:
        current = best_per_sector.get(s.sector)
        if current is None or s.score > current.score:
            best_per_sector[s.sector] = s

    # sector weights proportional to best stock scores (shifted positive)
    raw_scores = {sec: max(0.0, st.score) for sec, st in best_per_sector.items()}
    if sum(raw_scores.values()) <= 0:
        sector_weights = {sec: 1.0 / len(best_per_sector) for sec in best_per_sector}
    else:
        sector_weights = normalize(raw_scores)

    sector_alloc = {sec: initial_capital * w for sec, w in sector_weights.items()}

    # equity curves and P/L per stock
    equity_cols: Dict[str, pd.Series] = {}
    stock_rows = []

    for sec, st_obj in best_per_sector.items():
        alloc = sector_alloc[sec]
        px = st_obj.price.reindex(prices.index).ffill().bfill()
        if px.empty or alloc <= 0:
            continue
        p0 = float(px.iloc[0])
        if p0 <= 0:
            continue
        shares = alloc / p0
        equity = shares * px
        equity_cols[st_obj.ticker] = equity

        profit = float(equity.iloc[-1] - alloc)
        stock_rows.append(
            {
                "Sector": sec,
                "Ticker": st_obj.ticker,
                "Ann. Return %": st_obj.ann_ret,
                "Ann. Vol %": st_obj.ann_vol,
                "Score": st_obj.score,
                "Allocation $": alloc,
                "Profit $": profit,
                "Profit %": (profit / alloc) * 100.0 if alloc > 0 else 0.0,
            }
        )

    if not equity_cols:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    equity_df = pd.DataFrame(equity_cols).sort_index()

    # per-sector summary
    stock_df = pd.DataFrame(stock_rows)
    sector_summary = (
        stock_df.groupby("Sector", as_index=False)[["Allocation $", "Profit $"]]
        .sum()
        .assign(
            Profit_pct=lambda df: np.where(
                df["Allocation $"] > 0,
                df["Profit $"] / df["Allocation $"] * 100.0,
                0.0,
            )
        )
        .rename(columns={"Profit_pct": "Profit %"})
    )

    # store in session_state for reuse
    st.session_state["sector_page_equity_df"] = equity_df
    st.session_state["sector_page_prices_df"] = prices
    st.session_state["sector_page_stock_df"] = stock_df
    st.session_state["sector_page_sector_df"] = sector_summary
    st.session_state["sector_page_initial_capital"] = initial_capital
    st.session_state["sector_page_range"] = (from_date, to_date)
    st.session_state["sector_page_model"] = model_name

    return sector_summary, stock_df, equity_df


def rebuild_selected_portfolio(
    prices_df: pd.DataFrame,
    model_stock_df: pd.DataFrame,
    model_name: str,
    user_selection: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_rows = []
    equity_cols: Dict[str, pd.Series] = {}

    for _, row in model_stock_df.iterrows():
        sector = row["Sector"]
        model_ticker = row["Ticker"]
        selected_ticker = user_selection.get(sector, model_ticker)
        alloc = float(row["Allocation $"])

        if selected_ticker not in prices_df.columns:
            continue
        series = prices_df[selected_ticker].reindex(prices_df.index).ffill().bfill().dropna()
        if series.empty or alloc <= 0:
            continue
        start_price = float(series.iloc[0])
        if start_price <= 0:
            continue

        shares = alloc / start_price
        equity = shares * series
        equity_cols[f"{sector}::{selected_ticker}"] = equity

        ann_ret, ann_vol = annualized_stats(series)
        score = build_score(model_name, ann_ret, ann_vol)
        profit = float(equity.iloc[-1] - alloc)
        selected_rows.append(
            {
                "Sector": sector,
                "Model Ticker": model_ticker,
                "Ticker": selected_ticker,
                "Ann. Return %": ann_ret,
                "Ann. Vol %": ann_vol,
                "Score": score,
                "Allocation $": alloc,
                "Profit $": profit,
                "Profit %": (profit / alloc) * 100.0 if alloc > 0 else 0.0,
            }
        )

    selected_stock_df = pd.DataFrame(selected_rows)
    if selected_stock_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    selected_equity_df = pd.DataFrame(equity_cols).sort_index()
    selected_sector_df = (
        selected_stock_df.groupby("Sector", as_index=False)[["Allocation $", "Profit $"]]
        .sum()
        .assign(
            Profit_pct=lambda df: np.where(
                df["Allocation $"] > 0,
                df["Profit $"] / df["Allocation $"] * 100.0,
                0.0,
            )
        )
        .rename(columns={"Profit_pct": "Profit %"})
    )
    return selected_sector_df, selected_stock_df, selected_equity_df


def simulate_future_for_stock(
    price_series: pd.Series,
    alloc: float,
    horizon_days: int,
    n_sims: int,
) -> Tuple[float, float, float, float, float, float]:
    profile, _ = simulate_future_profile(price_series, alloc, horizon_days, n_sims)
    return (
        profile["profit"],
        profile["expected_final"],
        profile["p10_final"],
        profile["p90_final"],
        profile["prob_gain"],
        profile["start_value"],
    )


def _bootstrap_future_log_returns(
    price_series: pd.Series,
    horizon_days: int,
    n_sims: int,
) -> Tuple[np.ndarray, float]:
    px = price_series.dropna()
    if len(px) < 60:
        return np.array([]), 0.0

    rets = np.log(px / px.shift(1)).dropna()
    if rets.empty:
        return np.array([]), 0.0

    low_q = float(rets.quantile(0.01))
    high_q = float(rets.quantile(0.99))
    rets = rets.clip(lower=low_q, upper=high_q)
    returns_array = rets.to_numpy(dtype=np.float64)
    if len(returns_array) < max(90, horizon_days + 10):
        return np.array([]), 0.0

    # Historical scenario simulation: sample contiguous windows from real history.
    n_windows = len(returns_array) - horizon_days + 1
    if n_windows <= 0:
        return np.array([]), 0.0

    windows = np.array([returns_array[i : i + horizon_days] for i in range(n_windows)], dtype=np.float64)
    recent_vol = max(float(np.std(returns_array[-min(40, len(returns_array)) :])), 1e-8)
    win_vol = np.std(windows, axis=1)

    recency_rank = np.arange(n_windows, dtype=np.float64) + 1.0
    recency_w = recency_rank / recency_rank.sum()
    vol_diff = np.abs(win_vol - recent_vol)
    vol_w = np.exp(-vol_diff / recent_vol)
    weights = recency_w * vol_w
    weights = weights / max(weights.sum(), 1e-12)

    sampled_idx = np.random.choice(np.arange(n_windows), size=n_sims, p=weights, replace=True)
    boot = windows[sampled_idx].copy()

    # Conservative assumptions: transaction/friction drag + occasional stress hit.
    boot -= 0.00025
    stress_mask = np.random.rand(n_sims) < 0.22
    for i in np.where(stress_mask)[0]:
        shock_start = np.random.randint(0, max(1, horizon_days - 3))
        shock_len = min(np.random.randint(2, 5), horizon_days - shock_start)
        boot[i, shock_start : shock_start + shock_len] -= np.random.uniform(0.006, 0.018)

    boot = np.clip(boot, low_q, high_q)

    start_price = float(px.iloc[-1])
    if start_price <= 0:
        return np.array([]), 0.0

    return boot, start_price


def simulate_future_profile(
    price_series: pd.Series,
    alloc: float,
    horizon_days: int,
    n_sims: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    empty_profile = {
        "profit": 0.0,
        "expected_final": 0.0,
        "p10_final": 0.0,
        "p90_final": 0.0,
        "prob_gain": 0.0,
        "start_value": 0.0,
    }
    if alloc <= 0:
        return empty_profile, pd.DataFrame()

    boot, start_price = _bootstrap_future_log_returns(price_series, horizon_days, n_sims)
    if boot.size == 0 or start_price <= 0:
        return empty_profile, pd.DataFrame()

    start_equity = alloc
    cum_log = np.cumsum(boot, axis=1)
    growth_paths = np.exp(np.hstack([np.zeros((n_sims, 1)), cum_log]))
    price_paths = start_price * growth_paths
    equity_paths = start_equity * growth_paths

    finals = equity_paths[:, -1]
    mean_final = float(finals.mean())
    med_final = float(np.median(finals))
    q35_final = float(np.quantile(finals, 0.35))
    p10_final = float(np.quantile(finals, 0.10))
    p90_final = float(np.quantile(finals, 0.90))
    prob_gain = float(np.mean(finals > start_equity) * 100.0)

    # Conservative base-case forecast:
    # 1) start from lower quantile (q35), not mean
    # 2) cap by annualized return ceiling over the horizon
    # 3) shrink upside when uncertainty band is wide
    hist_log = np.log(price_series.dropna() / price_series.dropna().shift(1)).dropna()
    hist_ann = float(np.expm1(np.median(hist_log) * 252.0)) if not hist_log.empty else 0.0
    ann_cap = float(np.clip(0.05 + 0.35 * max(hist_ann, 0.0), 0.06, 0.14))
    horizon_cap_mult = (1.0 + ann_cap) ** (horizon_days / 252.0)
    horizon_cap_final = start_equity * horizon_cap_mult

    base_final = min(q35_final, horizon_cap_final, med_final)
    upside = max(0.0, base_final / start_equity - 1.0)
    band_width = max(0.0, (p90_final - p10_final) / start_equity)
    uncertainty_shrink = float(np.clip(1.0 - 0.8 * band_width, 0.1, 1.0))
    expected_final = start_equity * (1.0 + upside * uncertainty_shrink)
    expected_final = min(expected_final, base_final)

    profit = expected_final - start_equity

    curve_df = pd.DataFrame(
        {
            "Expected Price": price_paths.mean(axis=0),
            "P10 Price": np.quantile(price_paths, 0.10, axis=0),
            "P90 Price": np.quantile(price_paths, 0.90, axis=0),
            "Expected Equity ($)": equity_paths.mean(axis=0),
            "P10 Equity ($)": np.quantile(equity_paths, 0.10, axis=0),
            "P90 Equity ($)": np.quantile(equity_paths, 0.90, axis=0),
        }
    )

    profile = {
        "profit": profit,
        "expected_final": expected_final,
        "p10_final": p10_final,
        "p90_final": p90_final,
        "prob_gain": prob_gain,
        "start_value": start_equity,
    }
    return profile, curve_df


# ==========================
# PAGE LAYOUT
# ==========================

st.markdown(
    """
<div class="hero-wrap">
  <h2>Sector Allocation Lab</h2>
  <p>Model suggests one stock per selected sector. You can override picks and instantly view updated equity, price, and forecast visuals.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Sector Allocation Settings")
    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=1000.0,
        max_value=5_000_000.0,
        value=10_000.0,
        step=1000.0,
    )
    from_date = st.date_input("From Date")
    to_date = st.date_input("To Date")
    model_name = st.selectbox("Model", MODEL_CHOICES)
    selected_sectors = st.multiselect(
        "Select Sectors (1 or more)",
        options=list(SECTOR_TICKERS.keys()),
        default=["IT", "Healthcare", "Banking"],
    )
    default_future_date = to_date + timedelta(days=90)
    forecast_to_date = st.date_input(
        "Forecast End Date",
        value=default_future_date,
        min_value=to_date + timedelta(days=1),
    )
    sims = st.slider("Future simulations", 200, 2000, 500, 100)

build_clicked = st.button("Build Sector Portfolio")

if build_clicked:
    if not selected_sectors:
        st.error("Select at least one sector.")
        st.stop()
    if from_date >= to_date:
        st.error("`To Date` must be after `From Date`.")
        st.stop()
    if forecast_to_date <= to_date:
        st.error("`Forecast End Date` must be after `To Date`.")
        st.stop()

    horizon_days = len(
        pd.bdate_range(
            start=pd.Timestamp(to_date) + pd.Timedelta(days=1),
            end=pd.Timestamp(forecast_to_date),
        )
    )
    horizon_days = max(horizon_days, 5)

    with st.spinner("Building sector allocation..."):
        sector_df, stock_df, equity_df = build_portfolio(
            initial_capital=initial_capital,
            from_date=from_date,
            to_date=to_date,
            model_name=model_name,
            selected_sectors=selected_sectors,
        )

    if sector_df.empty or stock_df.empty or equity_df.empty:
        st.error("Could not build sector allocation. Try another date range.")
        st.stop()

    st.session_state["sector_page_build_done"] = True
    st.session_state["sector_page_last_stock_df"] = stock_df.copy()
    st.session_state["sector_page_last_prices_df"] = st.session_state["sector_page_prices_df"].copy()
    st.session_state["sector_page_last_model"] = model_name
    st.session_state["sector_page_last_selected_sectors"] = list(selected_sectors)
    st.session_state["sector_page_last_to_date"] = to_date
    st.session_state["sector_page_last_forecast_to_date"] = forecast_to_date
    st.session_state["sector_page_last_horizon_days"] = horizon_days
    st.session_state["sector_page_last_initial_capital"] = float(initial_capital)
    for _, row in stock_df.iterrows():
        sec = row["Sector"]
        st.session_state[f"selected_stock_{sec}"] = row["Ticker"]

if not st.session_state.get("sector_page_build_done", False):
    st.info("Choose settings and click **Build Sector Portfolio** to see allocation and graphs.")
    st.stop()

stock_df = st.session_state["sector_page_last_stock_df"].copy()
prices_df = st.session_state["sector_page_last_prices_df"].copy()
model_name = st.session_state["sector_page_last_model"]
selected_sectors = st.session_state.get("sector_page_last_selected_sectors", [])
to_date = st.session_state["sector_page_last_to_date"]
forecast_to_date = st.session_state["sector_page_last_forecast_to_date"]
horizon_days = int(st.session_state["sector_page_last_horizon_days"])
initial_capital = float(st.session_state["sector_page_last_initial_capital"])

if selected_sectors:
    st.markdown(f"`Sectors used:` {', '.join(selected_sectors)}")
render_stock_cards(stock_df, "Model Suggested Stocks")

if px_plotly is not None and not stock_df.empty:
    model_alloc_fig = px_plotly.bar(
        stock_df.sort_values("Allocation $", ascending=False),
        x="Sector",
        y="Allocation $",
        color="Profit %",
        hover_data=["Ticker", "Profit $", "Ann. Return %", "Ann. Vol %"],
        title="Model Allocation by Sector",
    )
    model_alloc_fig.update_layout(height=360)
    model_alloc_fig = style_animated_plotly(model_alloc_fig, chart_kind="bar")
    st.plotly_chart(model_alloc_fig, use_container_width=True)

st.markdown("### Personal Stock Selection by Sector")
st.caption("If you don't like a model stock, change it here. All graphs and forecasts update automatically.")
user_selection: Dict[str, str] = {}
for _, row in stock_df.sort_values("Sector").iterrows():
    sec = row["Sector"]
    model_ticker = row["Ticker"]
    options = list(dict.fromkeys(SECTOR_TICKERS.get(sec, [])))
    if not options:
        continue
    if model_ticker not in options:
        options = [model_ticker] + options
    default_idx = options.index(model_ticker) if model_ticker in options else 0
    user_selection[sec] = st.selectbox(
        f"{sec} stock",
        options=options,
        index=default_idx,
        key=f"selected_stock_{sec}",
    )
    chosen = user_selection[sec]
    if chosen not in prices_df.columns:
        st.warning(
            f"{sec}: `{chosen}` has no price data in the current date range. "
            "Pick another stock or rebuild with a wider range."
        )

selected_sector_df, selected_stock_df, selected_equity_df = rebuild_selected_portfolio(
    prices_df=prices_df,
    model_stock_df=stock_df,
    model_name=model_name,
    user_selection=user_selection,
)
if selected_stock_df.empty or selected_equity_df.empty:
    st.error("Could not build portfolio from selected stocks.")
    st.stop()

st.session_state["sector_page_selected_sector_df"] = selected_sector_df.copy()
st.session_state["sector_page_selected_stock_df"] = selected_stock_df.copy()
st.session_state["sector_page_selected_equity_df"] = selected_equity_df.copy()
st.session_state["sector_page_last_sims"] = int(sims)

render_stock_cards(selected_stock_df, "Final Portfolio (After Your Selection)", show_model_ticker=True)

if px_plotly is not None and not selected_sector_df.empty:
    sector_perf_fig = px_plotly.bar(
        selected_sector_df.sort_values("Profit %", ascending=False),
        x="Sector",
        y="Profit %",
        color="Profit %",
        title="Selected Sector Performance (%)",
        hover_data=["Allocation $", "Profit $"],
    )
    sector_perf_fig.update_layout(height=360)
    sector_perf_fig = style_animated_plotly(sector_perf_fig, chart_kind="bar")
    st.plotly_chart(sector_perf_fig, use_container_width=True)

st.markdown("### Equity Curve & Price for Selected Stocks")
for _, row in selected_stock_df.sort_values("Sector").iterrows():
    sec = row["Sector"]
    ticker = row["Ticker"]
    st.markdown(f"**{sec} – {ticker}**")

    eq_key = f"{sec}::{ticker}"
    eq = selected_equity_df.get(eq_key)
    price_series = prices_df.get(ticker)
    if eq is None or price_series is None:
        st.write("No data available.")
        continue
    eq = eq.dropna()
    price_series = price_series.dropna()
    if eq.empty or price_series.empty:
        st.write("Not enough data.")
        continue

    with st.expander(f"{sec} - {ticker} charts", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Equity curve ($)")
            plot_interactive_lines(
                pd.DataFrame({"Equity ($)": eq}, index=eq.index),
                f"{sec} {ticker} Equity",
                "USD",
            )
        with c2:
            st.caption("Stock price")
            plot_interactive_lines(
                pd.DataFrame({"Price": price_series}, index=price_series.index),
                f"{sec} {ticker} Price",
                "Price",
            )

future_rows = []
total_expected_profit = 0.0
future_curve_map: Dict[str, pd.DataFrame] = {}
for _, row in selected_stock_df.sort_values("Sector").iterrows():
    ticker = row["Ticker"]
    sec = row["Sector"]
    alloc = float(row["Allocation $"])
    px = prices_df.get(ticker)
    if px is None:
        continue

    profile, curve_df = simulate_future_profile(px, alloc, horizon_days, sims)
    if curve_df.empty:
        continue

    fut_profit = profile["profit"]
    fut_final = profile["expected_final"]
    p10_final = profile["p10_final"]
    p90_final = profile["p90_final"]
    prob_gain = profile["prob_gain"]
    start_val = profile["start_value"]
    total_expected_profit += fut_profit

    future_rows.append(
        {
            "Sector": sec,
            "Ticker": ticker,
            "Allocation $": alloc,
            "Expected Future Value $": fut_final,
            "Expected Future Profit $": fut_profit,
            "Expected Return %": (fut_profit / start_val) * 100.0 if start_val > 0 else 0.0,
            "P10 Value $": p10_final,
            "P90 Value $": p90_final,
            "Prob. Gain %": prob_gain,
        }
    )

    curve_index = pd.bdate_range(start=pd.Timestamp(to_date), periods=len(curve_df))
    curve_df = curve_df.copy()
    curve_df.index = curve_index
    future_curve_map[f"{sec}::{ticker}"] = curve_df

if future_rows:
    future_df = pd.DataFrame(future_rows)
    st.session_state["sector_page_future_df"] = future_df.copy()
    st.session_state["sector_page_future_total_profit"] = float(total_expected_profit)
    st.session_state["sector_page_future_curve_map"] = future_curve_map
else:
    st.session_state["sector_page_future_df"] = pd.DataFrame()
    st.session_state["sector_page_future_total_profit"] = 0.0
    st.session_state["sector_page_future_curve_map"] = {}

st.info("Future prediction details moved to the **Future Prediction** page.")

st.markdown("### Model Predicted Sector Allocation % (Pie Chart)")
pie_df = stock_df[["Sector", "Ticker", "Allocation $"]].copy()
if px_plotly is not None and not pie_df.empty:
    fig = px_plotly.pie(
        pie_df,
        names="Sector",
        values="Allocation $",
        custom_data=["Ticker"],
        hole=0.35,
        title="Model Suggested Stocks and Sector Weights",
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Model stock: %{customdata[0]}<br>"
        "Allocation: $%{value:,.2f}<br>Weight: %{percent}<extra></extra>",
    )
    fig = style_animated_plotly(fig, chart_kind="pie")
    st.plotly_chart(fig, use_container_width=True)
else:
    pie_fallback = pie_df.set_index("Sector")[["Allocation $"]]
    st.caption("Plotly is unavailable, showing allocation bar chart instead.")
    st.bar_chart(pie_fallback)
