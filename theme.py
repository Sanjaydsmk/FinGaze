import streamlit as st


def apply_shared_theme() -> None:
    st.markdown(
        """
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
    padding: 28px 30px;
    border-radius: 28px;
    border: 1px solid rgba(125, 211, 252, 0.22);
    background: linear-gradient(120deg, rgba(9, 18, 34, 0.92), rgba(17, 36, 69, 0.78));
    box-shadow: 0 28px 80px rgba(2, 6, 23, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(18px);
    animation: heroRise 900ms ease-out;
    margin-bottom: 14px;
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
.hero-title {
    position: relative;
    z-index: 1;
    margin: 0;
    font-size: clamp(2rem, 4vw, 3.2rem);
    line-height: 1;
    letter-spacing: -0.05em;
    font-weight: 800;
    color: #f8fafc;
}
.hero-title span {
    background: linear-gradient(90deg, #f8fafc 0%, #7dd3fc 48%, #4ade80 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    position: relative;
    z-index: 1;
    margin: 12px 0 0 0;
    max-width: 820px;
    color: var(--text-soft);
    font-size: 1rem;
    line-height: 1.7;
}
.stButton > button {
    background: linear-gradient(90deg, #0891b2, #2563eb 54%, #4f46e5 100%);
    color: white;
    border-radius: 14px;
    padding: 0.72em 1.6em;
    font-weight: 700;
    border: 1px solid rgba(125, 211, 252, 0.14);
    box-shadow: 0 12px 28px rgba(37, 99, 235, 0.24);
    transition: transform 220ms ease, box-shadow 220ms ease, filter 220ms ease;
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 18px 36px rgba(56, 189, 248, 0.28);
    filter: saturate(1.08);
}
[data-testid="stPlotlyChart"] {
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 18px;
    padding: 8px;
    background: linear-gradient(180deg, rgba(8, 15, 30, 0.76), rgba(2, 6, 23, 0.94));
    box-shadow: 0 20px 40px rgba(2, 6, 23, 0.26);
}
@keyframes heroRise {
    from { opacity: 0; transform: translateY(22px) scale(0.985); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}
@media (max-width: 640px) {
    .hero-shell {
        padding: 22px 20px;
        border-radius: 22px;
    }
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_page_hero(title: str, subtitle: str = "") -> None:
    subtitle_html = f'<p class="hero-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
<div class="hero-shell">
  <h1 class="hero-title">{title}</h1>
  {subtitle_html}
</div>
""",
        unsafe_allow_html=True,
    )
