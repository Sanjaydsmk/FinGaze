from datetime import timedelta
from typing import Dict

import pandas as pd
import streamlit as st
from theme import apply_shared_theme, render_page_hero

try:
    import plotly.express as px_plotly
except ImportError:
    px_plotly = None

st.set_page_config(page_title="Future Prediction", layout="wide")

st.markdown(
    """
<style>
.stApp {
    background: radial-gradient(circle at 10% 0%, #0b1220 0%, #0f172a 40%, #020617 100%);
    color: #e2e8f0;
}
.hero {
    background: linear-gradient(125deg, #111827, #1e3a8a);
    color: #fff;
    border-radius: 16px;
    padding: 20px 22px;
    margin-bottom: 14px;
}
.kpi {
    background: linear-gradient(180deg, #0f172a, #111827);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 12px;
}
.kpi-label {
    color: #cbd5e1;
    font-size: 0.95rem;
    margin-bottom: 8px;
}
.kpi-value {
    color: #f8fafc;
    font-size: 1.05rem;
    font-weight: 600;
    line-height: 1.35;
    white-space: normal;
    word-break: break-word;
}
.snap {
    background: linear-gradient(180deg, #0f172a, #111827);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)
apply_shared_theme()

render_page_hero(
    "Future Prediction Details",
    "Dedicated view for forecast outputs from Sector Allocation page.",
)

future_df: pd.DataFrame = st.session_state.get("sector_page_future_df", pd.DataFrame())
curve_map: Dict[str, pd.DataFrame] = st.session_state.get("sector_page_future_curve_map", {})
total_profit = float(st.session_state.get("sector_page_future_total_profit", 0.0))
initial_capital = float(st.session_state.get("sector_page_last_initial_capital", 0.0))
to_date = st.session_state.get("sector_page_last_to_date")
forecast_to_date = st.session_state.get("sector_page_last_forecast_to_date")
horizon_days = int(st.session_state.get("sector_page_last_horizon_days", 0))
sims = int(st.session_state.get("sector_page_last_sims", 0))

if future_df.empty:
    st.info("No forecast data found. Build portfolio first in Sector Allocation page.")
    st.stop()

period_text = ""
if to_date is not None and forecast_to_date is not None:
    start_text = pd.Timestamp(to_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end_text = pd.Timestamp(forecast_to_date).strftime("%Y-%m-%d")
    period_text = (
        f"{start_text} to {end_text} "
        f"({horizon_days} trading days, {sims} simulations)"
    )

c1, c2 = st.columns([1, 1])
with c1:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.metric(
        "Conservative base-case future profit",
        f"${total_profit:,.2f}",
        f"{(total_profit / initial_capital) * 100.0:.2f}%" if initial_capital > 0 else "",
    )
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">Forecast window</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">{period_text if period_text else "N/A"}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.caption("Forecast uses conservative base-case logic with stress/cost adjustments and capped upside.")

high_risk = future_df[(future_df["Prob. Gain %"] < 55.0) | (future_df["P10 Value $"] < 0.9 * future_df["Allocation $"])]
if not high_risk.empty:
    st.warning(
        "Risk flag: Some picks have low gain probability or large P10 downside. "
        "Prefer smaller position sizing for these names."
    )

st.markdown("### Future Projection Snapshot")
cols = st.columns(3)
for i, (_, row) in enumerate(future_df.sort_values("Expected Return %", ascending=False).iterrows()):
    with cols[i % 3]:
        st.markdown(
            f"""
<div class="snap">
  <div><b>{row['Sector']}</b> - {row['Ticker']}</div>
  <div style="margin-top:6px;">Allocation: ${row['Allocation $']:,.2f}</div>
  <div style="margin-top:4px;">Expected Profit: ${row['Expected Future Profit $']:,.2f}</div>
  <div style="margin-top:4px;">Expected Return: {row['Expected Return %']:.2f}%</div>
  <div style="margin-top:4px;">Prob Gain: {row['Prob. Gain %']:.1f}%</div>
</div>
""",
            unsafe_allow_html=True,
        )

if px_plotly is not None:
    bar = px_plotly.bar(
        future_df.sort_values("Expected Return %", ascending=False),
        x="Sector",
        y="Expected Return %",
        color="Prob. Gain %",
        hover_data=["Ticker", "Expected Future Profit $", "P10 Value $", "P90 Value $"],
        title="Forecasted Return by Sector/Stock",
        template="plotly_dark",
    )
    bar.update_layout(height=400, paper_bgcolor="#020617", plot_bgcolor="#0f172a")
    st.plotly_chart(bar, use_container_width=True)

if px_plotly is not None:
    pie = px_plotly.pie(
        future_df,
        names="Sector",
        values="Allocation $",
        color="Sector",
        custom_data=["Ticker"],
        hole=0.38,
        title="Selected Portfolio Allocation for Forecast",
        template="plotly_dark",
    )
    pie.update_traces(
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Ticker: %{customdata[0]}<br>Allocation: $%{value:,.2f}<extra></extra>",
    )
    pie.update_layout(height=420, paper_bgcolor="#020617", plot_bgcolor="#0f172a")
    st.plotly_chart(pie, use_container_width=True)

st.markdown("### Per-Stock Forecast Curves")
curve_options = []
for _, row in future_df.sort_values("Expected Return %", ascending=False).iterrows():
    sec = row["Sector"]
    ticker = row["Ticker"]
    key = f"{sec}::{ticker}"
    curve_df = curve_map.get(key)
    if curve_df is None or curve_df.empty:
        continue
    curve_options.append(f"{sec} - {ticker}")

if curve_options:
    st.caption("View selected sector")
    selected_curve_label = st.selectbox(
        "View selected sector forecast",
        options=curve_options,
        key="future_curve_focus",
        label_visibility="collapsed",
    )
    selected_sec, selected_ticker = selected_curve_label.split(" - ", 1)
    selected_key = f"{selected_sec}::{selected_ticker}"
    curve_df = curve_map.get(selected_key)

    if curve_df is not None and not curve_df.empty:
        st.markdown(f"**{selected_sec} - {selected_ticker}**")
        c1, c2 = st.columns(2)
        with c1:
            if px_plotly is not None:
                fig_eq = px_plotly.line(
                    curve_df.reset_index().rename(columns={"index": "Date"}),
                    x="Date",
                    y=["Expected Equity ($)", "P10 Equity ($)", "P90 Equity ($)"],
                    template="plotly_dark",
                    title=f"{selected_ticker} Forecast Equity",
                )
                fig_eq.update_layout(height=320, paper_bgcolor="#020617", plot_bgcolor="#0f172a")
                st.plotly_chart(fig_eq, use_container_width=True)
            else:
                st.line_chart(curve_df[["Expected Equity ($)", "P10 Equity ($)", "P90 Equity ($)"]])

        with c2:
            if px_plotly is not None:
                fig_px = px_plotly.line(
                    curve_df.reset_index().rename(columns={"index": "Date"}),
                    x="Date",
                    y=["Expected Price", "P10 Price", "P90 Price"],
                    template="plotly_dark",
                    title=f"{selected_ticker} Forecast Price",
                )
                fig_px.update_layout(height=320, paper_bgcolor="#020617", plot_bgcolor="#0f172a")
                st.plotly_chart(fig_px, use_container_width=True)
            else:
                st.line_chart(curve_df[["Expected Price", "P10 Price", "P90 Price"]])
