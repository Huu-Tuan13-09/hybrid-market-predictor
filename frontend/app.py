"""
frontend/app.py
================
Streamlit Frontend — Main dashboard for the Hybrid AI Market Predictor.

Pages:
  📊 Dashboard    — VN-Index price chart + technical indicators
  🤖 AI Analysis  — Run full prediction pipeline, display agent reports
  📰 News Feed    — Latest crawled financial news
  🗄️  RAG Manager  — Manage ChromaDB collections, run direct queries
  ⚙️  Settings     — Configuration panel

Architecture:
  All data comes from the FastAPI backend via HTTP requests.
  This file contains ZERO ML or AI logic — pure presentation layer.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Page Config — must be FIRST Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title     = "Hybrid AI Market Predictor",
    page_icon      = "📈",
    layout         = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Premium Dark Theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1929 50%, #0a1628 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1929 0%, #111827 100%);
    border-right: 1px solid rgba(99, 179, 237, 0.15);
}

/* Sidebar Navigation Items (Radio buttons) */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
    color: #ffffff !important;
    font-size: 1.05rem;
    font-weight: 500;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(17, 24, 39, 0.8);
    border: 1px solid rgba(99, 179, 237, 0.2);
    border-radius: 12px;
    padding: 1rem;
    backdrop-filter: blur(10px);
}

/* Prediction box */
.prediction-box {
    background: rgba(17, 24, 39, 0.9);
    border-radius: 16px;
    padding: 2rem;
    border: 1px solid rgba(99, 179, 237, 0.3);
    text-align: center;
    backdrop-filter: blur(20px);
}

.direction-up   { color: #48bb78; font-size: 3rem; font-weight: 700; }
.direction-down { color: #fc8181; font-size: 3rem; font-weight: 700; }
.direction-flat { color: #f6e05e; font-size: 3rem; font-weight: 700; }

/* Agent report cards */
.agent-card {
    background: rgba(26, 32, 44, 0.8);
    border-radius: 12px;
    padding: 1.2rem;
    margin: 0.5rem 0;
    border-left: 4px solid;
    backdrop-filter: blur(10px);
}
.agent-quant     { border-left-color: #63b3ed; }
.agent-sentiment { border-left-color: #68d391; }
.agent-economist { border-left-color: #f6ad55; }
.agent-cio       { border-left-color: #b794f4; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2b6cb0, #3182ce);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(66, 153, 225, 0.4);
}

/* Header gradient text */
.gradient-title {
    background: linear-gradient(135deg, #63b3ed, #9f7aea, #ed64a6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

/* --- FIX FADED TEXT --- */
/* Làm sáng nhãn của các input (Ticker, Ngày phân tích, Lookback days) */
[data-testid="stWidgetLabel"] p, label p {
    color: #ffffff !important;
    font-weight: 500 !important;
}

/* Làm sáng chữ ở st.caption (Last update, Cập nhật từ CafeF, ...) */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
[data-testid="stMarkdownContainer"] small {
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
}

/* Làm sáng chữ của st.metric (Đóng cửa, RSI, MACD, Quant Agent...) */
[data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] div {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

/* Làm sáng số liệu của st.metric (như 1,907.30, 70.6...) */
[data-testid="stMetricValue"] div {
    color: #ffffff !important;
}

/* Làm sáng chữ ở các Tab (Semantic Search, Index Text) */
button[data-baseweb="tab"] p, button[data-baseweb="tab"] span {
    color: #ffffff !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Config & Session State
# ---------------------------------------------------------------------------

import os
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "news_data" not in st.session_state:
    st.session_state.news_data = None


# ---------------------------------------------------------------------------
# API Helpers
# ---------------------------------------------------------------------------

def api_get(endpoint: str, params: dict = None) -> dict | None:
    try:
        resp = httpx.get(f"{BACKEND_URL}{endpoint}", params=params, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        st.error(f"API Error {exc.response.status_code}: {exc.response.text[:200]}")
    except Exception as exc:
        st.error(f"Connection error: {exc}")
    return None


def api_post(endpoint: str, payload: dict) -> dict | None:
    try:
        resp = httpx.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        st.error(f"API Error {exc.response.status_code}: {exc.response.text[:300]}")
    except Exception as exc:
        st.error(f"Connection error: {exc}")
    return None


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def direction_color(direction: str) -> str:
    return {"TĂNG": "#48bb78", "GIẢM": "#fc8181", "ĐI NGANG": "#f6e05e"}.get(direction, "#a0aec0")

def direction_icon(direction: str) -> str:
    return {"TĂNG": "📈", "GIẢM": "📉", "ĐI NGANG": "➡️"}.get(direction, "❓")

def css_class(direction: str) -> str:
    return {"TĂNG": "direction-up", "GIẢM": "direction-down", "ĐI NGANG": "direction-flat"}.get(direction, "")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="gradient-title">📈 HybridAI</div>', unsafe_allow_html=True)
    st.caption("VN-Index Market Intelligence")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "🤖 AI Analysis", "📰 News Feed", "🗄️ RAG Manager", "⚙️ Settings"],
        label_visibility="collapsed",
    )

    st.divider()
    # Health check indicator
    health = api_get("/health")
    if health and health.get("status") == "ok":
        model_icon = "✅" if health.get("model_loaded") else "⚠️"
        st.success(f"Backend Online {model_icon}")
        counts = health.get("chromadb_counts", {})
        for coll, n in counts.items():
            st.caption(f"  📦 {coll}: {n} docs")
    else:
        st.error("Backend Offline")
    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")


# ===========================================================================
# PAGE: Dashboard
# ===========================================================================

if page == "📊 Dashboard":
    st.markdown('<div class="gradient-title">📊 VN-Index Dashboard</div>', unsafe_allow_html=True)
    st.caption(f"Market data powered by Yahoo Finance | Last update: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    st.divider()

    col_ticker, col_days, col_btn = st.columns([2, 2, 1])
    with col_ticker:
        ticker = st.text_input("Ticker", value="^VNINDEX", label_visibility="collapsed")
    with col_days:
        days = st.selectbox("Khoảng thời gian", [30, 60, 90, 180], label_visibility="collapsed")
    with col_btn:
        load_btn = st.button("🔄 Load", use_container_width=True)

    if load_btn or True:  # Auto-load on page visit
        with st.spinner("Fetching market data…"):
            mdata = api_get("/market-data", params={"ticker": ticker, "days": days})

        if mdata:
            # ── KPI Row ─────────────────────────────────────────────────
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            ind = mdata.get("indicators", {})
            r1d = mdata.get("return_1d", 0)
            kpi1.metric("Đóng cửa", f"{mdata['close']:,.2f}", f"{r1d:+.2f}%")
            kpi2.metric("RSI(14)",  f"{ind.get('rsi_14', 0):.1f}",
                        "Overbought" if ind.get("rsi_14", 50) > 70 else ("Oversold" if ind.get("rsi_14", 50) < 30 else "Normal"))
            kpi3.metric("MACD",    f"{ind.get('macd', 0):.2f}")
            kpi4.metric("ADX",     f"{ind.get('adx', 0):.1f}",
                        "Strong trend" if ind.get("adx", 0) > 25 else "Weak trend")
            kpi5.metric("Vol Ratio", f"{ind.get('volume_ratio', 0):.2f}x")

            st.divider()

            # ── OHLCV Chart ─────────────────────────────────────────────
            history = mdata.get("ohlcv_history", [])
            if history:
                df_chart = pd.DataFrame(history)
                for col_name in ["index", "Date", "time", "date"]:
                    if col_name in df_chart.columns and col_name != "date":
                        df_chart = df_chart.rename(columns={col_name: "date"})
                        break
                
                if "date" in df_chart.columns:
                    date_series = pd.to_datetime(df_chart["date"]).dt.strftime('%d/%m/%Y')
                    df_chart["hover_text"] = (
                        date_series + "<br>" +
                        "open: " + df_chart["open"].apply(lambda x: f"{x:,.2f}") + "<br>" +
                        "high: " + df_chart["high"].apply(lambda x: f"{x:,.2f}") + "<br>" +
                        "low: " + df_chart["low"].apply(lambda x: f"{x:,.2f}") + "<br>" +
                        "close: " + df_chart["close"].apply(lambda x: f"{x:,.2f}")
                    )
                else:
                    df_chart["hover_text"] = ""

                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    row_heights=[0.75, 0.25],
                    vertical_spacing=0.03,
                )

                fig.add_trace(go.Candlestick(
                    x     = df_chart.get("date", df_chart.index),
                    open  = df_chart["open"],
                    high  = df_chart["high"],
                    low   = df_chart["low"],
                    close = df_chart["close"],
                    name  = ticker,
                    text  = df_chart.get("hover_text", ""),
                    hoverinfo = "text",
                    increasing_line_color = "#48bb78",
                    decreasing_line_color = "#fc8181",
                ), row=1, col=1)

                fig.add_trace(go.Bar(
                    x     = df_chart.get("date", df_chart.index),
                    y     = df_chart["volume"],
                    name  = "Volume",
                    marker_color = "#63b3ed",
                    opacity = 0.6,
                ), row=2, col=1)

                fig.update_layout(
                    paper_bgcolor = "rgba(0,0,0,0)",
                    plot_bgcolor  = "rgba(13,25,41,0.8)",
                    font          = {"color": "#e2e8f0", "family": "Inter"},
                    xaxis_rangeslider_visible = False,
                    height        = 500,
                    showlegend    = False,
                    margin        = {"l": 0, "r": 0, "t": 10, "b": 0},
                )
                fig.update_xaxes(gridcolor="rgba(99,179,237,0.1)")
                fig.update_yaxes(gridcolor="rgba(99,179,237,0.1)")
                st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE: AI Analysis
# ===========================================================================

elif page == "🤖 AI Analysis":
    st.markdown('<div class="gradient-title">🤖 Hybrid AI Analysis</div>', unsafe_allow_html=True)
    st.caption("XGBoost ML + LangGraph Multi-Agent (Llama 3.3 via Groq)")
    st.divider()

    col_t, col_d, col_run = st.columns([2, 2, 1])
    with col_t:
        ticker_ai = st.text_input("Ticker", value="^VNINDEX", key="ai_ticker")
    with col_d:
        analysis_date = st.date_input("Ngày phân tích", value=date.today(), key="ai_date")
    with col_run:
        run_btn = st.button("🚀 Chạy Phân tích", use_container_width=True, type="primary")

    if run_btn:
        with st.spinner("🔄 Đang chạy pipeline... (30-90 giây)"):
            payload = {
                "ticker":        ticker_ai,
                "analysis_date": str(analysis_date),
                "lookback_days": 504,
            }
            result = api_post("/predict", payload)

        if result:
            st.session_state.prediction_result = result

    result = st.session_state.prediction_result
    if result:
        direction = result.get("direction", "ĐI NGANG")
        conf      = result.get("confidence_score", 0)
        action    = result.get("action", "CHỜ ĐỢI")
        regime    = result.get("weight_regime", "BALANCED")

        # ── Hero Prediction Card ──────────────────────────────────────
        st.markdown(f"""
        <div class="prediction-box">
            <div style="color: #a0aec0; font-size: 0.9rem; margin-bottom: 0.5rem;">
                VN-Index Dự báo Ngày Mai — {result.get('prediction_date', '')}
            </div>
            <div class="{css_class(direction)}">{direction_icon(direction)} {direction}</div>
            <div style="color: #a0aec0; margin-top: 0.5rem;">
                Độ tin cậy: <strong style="color:{direction_color(direction)}">{conf:.0%}</strong>
                &nbsp;|&nbsp; Quyết định: <strong style="color: #b794f4">{action}</strong>
                &nbsp;|&nbsp; Chế độ trọng số: <strong style="color: #63b3ed">{regime}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")  # spacer

        # ── ML vs Weight Score ────────────────────────────────────────
        col_ml, col_ws, col_proc = st.columns(3)
        ml_p = result.get("ml_probability", {})
        col_ml.metric(
            "XGBoost P(Tăng)", f"{ml_p.get('p_up', 0):.1%}",
            f"ML: {result.get('ml_direction','')} [{result.get('ml_confidence','')}]",
        )
        col_ws.metric(
            "Weighted Score", f"{result.get('weighted_score', 0):+.4f}",
            result.get("consensus", "MIXED"),
        )
        col_proc.metric(
            "Processing Time", f"{result.get('processing_time_ms', 0):,} ms",
            f"{result.get('articles_scraped', 0)} articles | {result.get('rag_context_chars', 0):,} chars RAG",
        )

        st.divider()

        # ── Dynamic Weights ───────────────────────────────────────────
        st.subheader("⚖️ Trọng số Động (CIO)")
        weights = result.get("weights_used", {})
        w_col1, w_col2, w_col3 = st.columns(3)
        w_col1.metric("Quant Agent",     f"{weights.get('quant', 0):.0%}")
        w_col2.metric("Sentiment Agent", f"{weights.get('sentiment', 0):.0%}")
        w_col3.metric("Macro Economist", f"{weights.get('macro', 0):.0%}")

        # ── CIO Reasoning ────────────────────────────────────────────
        with st.expander("🧠 CIO Lý luận Tổng hợp", expanded=True):
            st.markdown(result.get("cio_reasoning", ""))
            if result.get("key_signals"):
                st.subheader("📌 Tín hiệu Chính")
                for sig in result["key_signals"]:
                    st.markdown(f"• {sig}")
            if result.get("risk_factors"):
                st.subheader("⚠️ Rủi ro")
                for risk in result["risk_factors"]:
                    st.markdown(f"• {risk}")
            if result.get("stop_loss_note"):
                st.info(f"🛑 Stop-loss note: {result['stop_loss_note']}")

        # ── Agent Reports ─────────────────────────────────────────────
        st.divider()
        st.subheader("🤖 Báo cáo Chi tiết từng Agent")
        agent_colors = {
            "Quant Agent":      ("agent-quant",     "📊"),
            "Sentiment Agent":  ("agent-sentiment",  "💬"),
            "Economist Agent":  ("agent-economist",  "🏛️"),
        }
        for rep in result.get("agent_reports", []):
            name   = rep["agent_name"]
            css, icon = agent_colors.get(name, ("agent-cio", "🔮"))
            conf_c = rep["confidence"]
            color  = "#48bb78" if conf_c >= 65 else ("#f6e05e" if conf_c >= 40 else "#fc8181")
            st.markdown(f"""
            <div class="agent-card {css}">
                <strong>{icon} {name}</strong>
                &nbsp;&nbsp;<span style="color:{color}">Confidence: {conf_c}%</span>
                &nbsp;&nbsp;<em style="color:#a0aec0">{rep['recommendation']}</em>
                <div style="margin-top: 0.5rem; color: #cbd5e0">{rep['summary']}</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"  Raw JSON — {name}"):
                st.json(rep.get("raw_report", {}))

        # ── Errors ────────────────────────────────────────────────────
        if result.get("error_messages"):
            with st.expander("⚠️ Non-fatal Errors"):
                for err in result["error_messages"]:
                    st.warning(err)


# ===========================================================================
# PAGE: News Feed
# ===========================================================================

elif page == "📰 News Feed":
    st.markdown('<div class="gradient-title">📰 Tin tức Tài chính</div>', unsafe_allow_html=True)
    st.caption("Cập nhật từ CafeF, Tinnhanh Chứng khoán, VnEconomy")
    st.divider()

    col_lim, col_btn = st.columns([3, 1])
    with col_lim:
        limit = st.slider("Số bài báo", 5, 30, 15)
    with col_btn:
        refresh_btn = st.button("🔄 Tải tin mới", use_container_width=True)

    if refresh_btn or st.session_state.news_data is None:
        with st.spinner("Đang crawl tin tức…"):
            news_resp = api_get("/news", params={"limit": limit})
        if news_resp:
            st.session_state.news_data = news_resp

    news_data = st.session_state.news_data
    if news_data:
        st.caption(f"Tổng cộng: {news_data.get('total', 0)} bài | Scraped at: {news_data.get('scraped_at', 'N/A')[:19]}")
        for art in news_data.get("articles", []):
            with st.container():
                col_src, col_date = st.columns([1, 3])
                col_src.caption(f"📌 {art.get('source', '').upper()}")
                col_date.caption(art.get("published_at", ""))
                st.markdown(f"**[{art.get('title', 'No title')}]({art.get('url', '#')})**")
                if art.get("summary"):
                    st.markdown(f"<small>{art['summary']}</small>", unsafe_allow_html=True)
                st.divider()


# ===========================================================================
# PAGE: RAG Manager
# ===========================================================================

elif page == "🗄️ RAG Manager":
    st.markdown('<div class="gradient-title">🗄️ RAG Manager</div>', unsafe_allow_html=True)
    st.caption("Quản lý ChromaDB vector store — Index tài liệu vĩ mô")
    st.divider()

    tab1, tab2 = st.tabs(["🔍 Semantic Search", "➕ Index Text"])

    with tab1:
        q_col, c_col = st.columns([3, 1])
        with q_col:
            query = st.text_input("Nhập câu truy vấn", placeholder="e.g. Tác động lãi suất FED đến VN-Index")
        with c_col:
            coll = st.selectbox("Collection", ["macro_reports", "market_news", "company_filings"])
        if st.button("🔍 Tìm kiếm", use_container_width=True) and query:
            rag_res = api_get("/rag/query", params={"q": query, "collection": coll, "n_results": 5})
            if rag_res:
                st.caption(f"Tìm thấy {rag_res.get('total', 0)} kết quả")
                for r in rag_res.get("results", []):
                    with st.expander(f"📄 {r.get('source', 'unknown')} | sim={r.get('similarity', 0):.2f}"):
                        st.markdown(r.get("text", ""))

    with tab2:
        idx_text = st.text_area("Văn bản cần index", height=200)
        idx_src  = st.text_input("Nguồn (source identifier)")
        idx_coll = st.selectbox("Collection đích", ["macro_reports", "company_filings", "market_news"], key="idx_coll")
        if st.button("➕ Index vào ChromaDB") and idx_text and idx_src:
            result = api_post("/rag/index", {
                "text": idx_text, "source": idx_src, "collection": idx_coll
            })
            if result:
                st.success(f"✅ Đã index {result.get('chunks_indexed', 0)} chunks vào '{idx_coll}'")


# ===========================================================================
# PAGE: Settings
# ===========================================================================

elif page == "⚙️ Settings":
    st.markdown('<div class="gradient-title">⚙️ Settings</div>', unsafe_allow_html=True)
    st.divider()

    st.subheader("🔗 Backend Connection")
    st.info(f"Backend URL: `{BACKEND_URL}`")

    health = api_get("/health")
    if health:
        st.json(health)

    st.divider()
    st.subheader("🏋️ Re-train Model")
    st.warning("Re-training sẽ mất 1-3 phút. Backend sẽ tự động dùng model mới sau khi train xong.")
    train_ticker = st.text_input("Ticker", value="^VNINDEX", key="train_ticker")
    train_days   = st.number_input("Lookback days", value=504, min_value=100, max_value=1260)
    if st.button("🔄 Bắt đầu Training", type="primary"):
        with st.spinner("Training XGBoost model…"):
            train_res = api_post("/train", {"ticker": train_ticker, "lookback_days": train_days})
        if train_res:
            st.success(f"✅ Training hoàn thành! Accuracy={train_res.get('accuracy', 0):.1%} | AUC={train_res.get('auc_roc', 0):.4f}")
            st.json(train_res)
