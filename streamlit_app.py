from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent import StockAnalysisAgent, parse_tickers  # noqa: E402

# ─────────────────────────────────────────────
# CẤU HÌNH TRANG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VnStock AI – Phân tích chứng khoán thông minh",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not os.getenv("ANTHROPIC_API_KEY") and "ANTHROPIC_API_KEY" in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = str(st.secrets["ANTHROPIC_API_KEY"])


# ─────────────────────────────────────────────
# CSS TÙNG CHỈNH
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Font & nền chung ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a3a6b 0%, #2453B5 60%, #3b78d4 100%);
    border-radius: 16px;
    padding: 36px 40px 32px 40px;
    margin-bottom: 28px;
    color: #fff;
    box-shadow: 0 4px 24px rgba(36,83,181,0.18);
}
.hero-banner h1 {
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.hero-banner p {
    font-size: 1.05rem;
    opacity: 0.88;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.30);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 14px;
    color: #e8f0ff;
}

/* ── Thẻ kết quả phân tích ── */
.analysis-card {
    background: #ffffff;
    border-radius: 14px;
    border: 1px solid #e2e8f4;
    padding: 28px 28px 24px 28px;
    box-shadow: 0 2px 12px rgba(36,83,181,0.07);
    margin-bottom: 16px;
}
.analysis-card h3 {
    color: #1a3a6b;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 2px solid #eaf0ff;
}

/* ── Thẻ số liệu (metric card) ── */
.metric-row {
    display: flex;
    gap: 14px;
    margin-bottom: 18px;
    flex-wrap: wrap;
}
.metric-card {
    background: #f7faff;
    border: 1px solid #dce8ff;
    border-radius: 12px;
    padding: 14px 20px;
    flex: 1;
    min-width: 110px;
    text-align: center;
}
.metric-card .label {
    font-size: 0.73rem;
    color: #6b7a99;
    font-weight: 600;
    letter-spacing: 0.3px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.metric-card .value {
    font-size: 1.22rem;
    font-weight: 700;
    color: #1a3a6b;
}
.metric-card .value.up   { color: #0e9c65; }
.metric-card .value.down { color: #e02424; }

/* ── Nút bấm ── */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2453B5, #3b78d4);
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.97rem;
    padding: 10px 0;
    letter-spacing: 0.2px;
    transition: opacity .2s;
}
div.stButton > button[kind="primary"]:hover { opacity: 0.88; }

div.stButton > button:not([kind="primary"]) {
    border-radius: 10px;
    font-weight: 500;
}

/* ── Ô nhập liệu ── */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    border-radius: 10px;
    border: 1.5px solid #d0dbf0;
    font-size: 1rem;
    transition: border-color .2s;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
    border-color: #2453B5;
    box-shadow: 0 0 0 3px rgba(36,83,181,0.12);
}

/* ── Tab ── */
div[data-testid="stTabs"] button[role="tab"] {
    font-weight: 600;
    font-size: 0.97rem;
    padding: 8px 24px;
    border-radius: 8px 8px 0 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #f0f4fc;
}
.sidebar-logo {
    text-align: center;
    padding: 16px 0 10px 0;
}
.sidebar-logo .brand {
    font-size: 1.5rem;
    font-weight: 800;
    color: #2453B5;
    letter-spacing: -0.5px;
}
.sidebar-logo .tagline {
    font-size: 0.78rem;
    color: #7a8fb5;
    margin-top: 2px;
}
.sidebar-section {
    background: #ffffff;
    border-radius: 12px;
    padding: 16px;
    margin-top: 14px;
    border: 1px solid #dce8ff;
}
.sidebar-section h4 {
    font-size: 0.82rem;
    font-weight: 700;
    color: #6b7a99;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin: 0 0 10px 0;
}
.sidebar-item {
    font-size: 0.88rem;
    color: #2e3d5a;
    padding: 5px 0;
    display: flex;
    align-items: flex-start;
    gap: 8px;
    line-height: 1.4;
}

/* ── Follow-up box ── */
.followup-box {
    background: #f0f4fc;
    border-radius: 12px;
    border: 1px solid #dce8ff;
    padding: 20px 24px;
    margin-top: 8px;
}

/* ── Divider ── */
hr { border-color: #e8edf5; margin: 18px 0; }

/* ── Thẻ biểu tượng tính năng ── */
.feature-grid {
    display: flex;
    gap: 14px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.feature-item {
    background: #f7faff;
    border: 1px solid #dce8ff;
    border-radius: 12px;
    padding: 14px 18px;
    flex: 1;
    min-width: 160px;
    font-size: 0.88rem;
    color: #2e3d5a;
    line-height: 1.5;
}
.feature-item .icon { font-size: 1.5rem; margin-bottom: 6px; display: block; }
.feature-item strong { color: #1a3a6b; font-weight: 700; }

/* ── Expander ── */
div[data-testid="stExpander"] {
    border-radius: 12px;
    border: 1px solid #e2e8f4;
    overflow: hidden;
}

/* ── Download button ── */
div[data-testid="stDownloadButton"] > button {
    border-radius: 10px;
    border: 1.5px solid #2453B5;
    color: #2453B5;
    font-weight: 600;
    background: white;
    width: 100%;
    font-size: 0.93rem;
    padding: 8px 0;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: #eaf0ff;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HÀM TIỆN ÍCH
# ─────────────────────────────────────────────
def get_agent() -> StockAnalysisAgent:
    if "agent" not in st.session_state:
        st.session_state.agent = StockAnalysisAgent()
    return st.session_state.agent


def build_price_table(symbol: str) -> pd.DataFrame:
    agent = get_agent()
    df = agent.fetcher.get_ohlcv(symbol)
    df = agent.fetcher.calculate_indicators(df)
    return df.tail(20).copy()


def build_intraday_data(symbol: str) -> tuple:
    """Trả về (intraday_bars_df, current_price_dict) khi trong phiên giao dịch."""
    fetcher = get_agent().fetcher
    if not fetcher.is_trading_session():
        return pd.DataFrame(), {}
    bars = fetcher.get_intraday_bars(symbol)
    current = fetcher.get_current_price(symbol)
    return bars, current


def analyze_single(symbol: str) -> str:
    agent = get_agent()
    agent.reset()
    return agent.analyze(symbol)


async def analyze_batch(symbols: List[str]) -> Dict[str, str]:
    agent = get_agent()
    return await agent.analyze_many(symbols)


def _pdf_download_single(symbol: str, result_text: str) -> None:
    pdf_bytes = st.session_state.get("single_pdf_bytes")
    pdf_name = st.session_state.get("single_pdf_name")
    if pdf_bytes is None:
        agent = get_agent()
        pdf_path = Path(agent.export_single_report(symbol, result_text))
        pdf_bytes = pdf_path.read_bytes()
        pdf_name = pdf_path.name
        st.session_state.single_pdf_bytes = pdf_bytes
        st.session_state.single_pdf_name = pdf_name
    st.download_button(
        label="⬇  Tải báo cáo PDF",
        data=pdf_bytes,
        file_name=pdf_name,
        mime="application/pdf",
        use_container_width=True,
    )


def _pdf_download_batch(results: Dict[str, str]) -> None:
    pdf_bytes = st.session_state.get("batch_pdf_bytes")
    pdf_name = st.session_state.get("batch_pdf_name")
    if pdf_bytes is None:
        agent = get_agent()
        pdf_path = Path(agent.export_batch_report(results))
        pdf_bytes = pdf_path.read_bytes()
        pdf_name = pdf_path.name
        st.session_state.batch_pdf_bytes = pdf_bytes
        st.session_state.batch_pdf_name = pdf_name
    st.download_button(
        label="⬇  Tải báo cáo PDF tổng hợp",
        data=pdf_bytes,
        file_name=pdf_name,
        mime="application/pdf",
        use_container_width=True,
    )


def _metric_html(label: str, value: str, cls: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value {cls}">{value}</div>
    </div>"""


def render_disclaimer() -> None:
    st.warning(
        "Miễn trừ trách nhiệm: Nội dung phân tích chỉ mang tính tham khảo, không phải khuyến nghị đầu tư bắt buộc. "
        "Người dùng tự chịu trách nhiệm với mọi quyết định giao dịch và nên kết hợp thêm tư vấn tài chính độc lập.",
        icon="⚖️",
    )


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <div class="brand">📊 VnStock AI</div>
            <div class="tagline">Phân tích chứng khoán thông minh</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("""
        <div class="sidebar-section">
            <h4>⚡ Tính năng</h4>
            <div class="sidebar-item">🔍 <span>Phân tích đơn lẻ hoặc nhiều mã cùng lúc</span></div>
            <div class="sidebar-item">🤖 <span>AI Claude phân tích kỹ thuật & cơ bản</span></div>
            <div class="sidebar-item">📈 <span>Biểu đồ giá & chỉ báo kỹ thuật realtime</span></div>
            <div class="sidebar-item">📄 <span>Xuất báo cáo PDF chuyên nghiệp</span></div>
            <div class="sidebar-item">💬 <span>Hỏi đáp follow-up với AI</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-section" style="margin-top:12px">
            <h4>📌 Hướng dẫn</h4>
            <div class="sidebar-item">1️⃣ <span>Nhập mã cổ phiếu (VD: STB, VNM)</span></div>
            <div class="sidebar-item">2️⃣ <span>Nhấn <b>Phân tích ngay</b></span></div>
            <div class="sidebar-item">3️⃣ <span>Đọc kết quả & tải PDF</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Đặt lại hội thoại", use_container_width=True):
            if "agent" in st.session_state:
                st.session_state.agent.reset()
            for key in [
                "single_result", "single_symbol_name", "single_df",
                "single_intraday", "single_current_price", "single_in_session",
                "single_follow_up", "follow_up_answer",
                "single_pdf_bytes", "single_pdf_name",
                "batch_results", "batch_symbols",
                "batch_pdf_bytes", "batch_pdf_name",
            ]:
                st.session_state.pop(key, None)
            st.success("✅ Đã đặt lại thành công!")

        st.markdown("""
        <div class="sidebar-section" style="margin-top:12px;background:#fff8e8;border-color:#f3d28f;">
            <h4>⚖️ Miễn trừ trách nhiệm</h4>
            <div class="sidebar-item">Nội dung AI chỉ mang tính tham khảo.</div>
            <div class="sidebar-item">Không phải lời khuyên mua/bán chứng khoán.</div>
            <div class="sidebar-item">Vui lòng tự quản trị rủi ro trước khi giao dịch.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("© 2026 VnStock AI · Dữ liệu từ TCBS/VCI · Powered by Claude AI")


# ─────────────────────────────────────────────
# HEADER CHÍNH
# ─────────────────────────────────────────────
def render_header() -> None:
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">🚀 AI-Powered · Thị trường Việt Nam</div>
        <h1>Phân tích Chứng khoán thông minh</h1>
        <p>Phân tích kỹ thuật & cơ bản chuyên sâu bằng AI · Xem biểu đồ · Xuất báo cáo PDF ngay trên trình duyệt</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature highlights
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-item">
            <span class="icon">🤖</span>
            <strong>AI Claude Phân tích</strong><br>
            Nhận xét chuyên sâu về xu hướng, hỗ trợ, kháng cự
        </div>
        <div class="feature-item">
            <span class="icon">📊</span>
            <strong>Chỉ báo Kỹ thuật</strong><br>
            MA5/20/50, RSI, MACD, Bollinger Bands
        </div>
        <div class="feature-item">
            <span class="icon">📈</span>
            <strong>Đa mã Đồng thời</strong><br>
            Phân tích tối đa 5 mã song song
        </div>
        <div class="feature-item">
            <span class="icon">📄</span>
            <strong>Báo cáo PDF</strong><br>
            Xuất báo cáo chuyên nghiệp ngay lập tức
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB: PHÂN TÍCH 1 MÃ
# ─────────────────────────────────────────────
def render_single_tab() -> None:
    with st.container():
        c1, c2 = st.columns([2.5, 1])
        with c1:
            single_symbol = st.text_input(
                "Nhập mã cổ phiếu",
                placeholder="Ví dụ: STB, VNM, HPG, MWG...",
                key="single_symbol",
                label_visibility="visible",
            )
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_single = st.button("🔍 Phân tích ngay", type="primary", use_container_width=True)

    if run_single:
        symbol = single_symbol.strip().upper()
        if not symbol:
            st.warning("⚠️ Vui lòng nhập mã cổ phiếu trước khi phân tích.")
            return

        fetcher = get_agent().fetcher
        in_session = fetcher.is_trading_session()

        spinner_msg = (
            f"🔴 Đang trong phiên giao dịch — AI phân tích realtime mã **{symbol}**..."
            if in_session else
            f"🤖 AI đang phân tích mã **{symbol}**, vui lòng đợi..."
        )
        with st.spinner(spinner_msg):
            try:
                result   = analyze_single(symbol)
                price_df = build_price_table(symbol)
                intraday_bars, current_price = build_intraday_data(symbol)
            except Exception as exc:
                st.error(f"❌ Không thể phân tích {symbol}: {exc}")
                return

        st.session_state.single_result        = result
        st.session_state.single_symbol_name   = symbol
        st.session_state.single_df            = price_df
        st.session_state.single_intraday      = intraday_bars
        st.session_state.single_current_price = current_price
        st.session_state.single_in_session    = in_session
        st.session_state.pop("single_pdf_bytes", None)
        st.session_state.pop("single_pdf_name",  None)
        st.session_state.pop("follow_up_answer",  None)

    result        = st.session_state.get("single_result")
    symbol        = st.session_state.get("single_symbol_name")
    price_df      = st.session_state.get("single_df")
    intraday_bars = st.session_state.get("single_intraday", pd.DataFrame())
    current_price = st.session_state.get("single_current_price", {})
    in_session    = st.session_state.get("single_in_session", False)

    if result and symbol:
        # ── Badge phiên giao dịch ──
        if in_session:
            now_str = get_agent().fetcher.get_vn_now_str()
            st.markdown(f"""
            <div style="background:#e6f9ef;border:1.5px solid #0e9c65;border-radius:10px;
                        padding:10px 18px;margin-bottom:14px;display:flex;align-items:center;gap:10px;">
                <span style="font-size:1.3rem;">🔴</span>
                <span style="color:#0a7a50;font-weight:700;font-size:0.97rem;">
                    ĐANG TRONG PHIÊN GIAO DỊCH — Dữ liệu realtime · {now_str}
                </span>
                <span style="margin-left:auto;font-size:0.82rem;color:#3aaa7a;">
                    HOSE 9:00 – 14:45 | HNX 9:00 – 15:00
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        col_left, col_right = st.columns([1.3, 1])

        # ── Cột trái: Kết quả phân tích AI ──
        with col_left:
            st.markdown(f"""
            <div class="analysis-card">
                <h3>🤖 Kết quả phân tích AI — {symbol}</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(result)
            st.markdown("<br>", unsafe_allow_html=True)
            _pdf_download_single(symbol, result)

        # ── Cột phải: Dữ liệu giá & biểu đồ ──
        with col_right:
            # Metric realtime khi trong phiên
            if in_session and current_price:
                rt_price  = current_price.get("price", 0)
                rt_high   = current_price.get("high", 0)
                rt_low    = current_price.get("low", 0)
                rt_vol    = current_price.get("volume", 0)
                rt_time   = current_price.get("time", "")
                st.markdown(f"""
                <div style="background:#fff8e1;border:1.5px solid #ffc107;border-radius:12px;
                            padding:12px 16px;margin-bottom:12px;">
                    <div style="font-size:0.75rem;font-weight:700;color:#b8860b;
                                letter-spacing:.5px;text-transform:uppercase;margin-bottom:6px;">
                        ⚡ Giá Realtime (lúc {rt_time})
                    </div>
                    <div class="metric-row" style="margin-bottom:0">
                        {_metric_html("Giá khớp", f"{rt_price:,}")}
                        {_metric_html("Cao nhất", f"{rt_high:,}")}
                        {_metric_html("Thấp nhất", f"{rt_low:,}")}
                        {_metric_html("KL khớp", f"{rt_vol/1e3:.0f}K")}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            elif isinstance(price_df, pd.DataFrame) and not price_df.empty:
                latest     = price_df.iloc[-1]
                prev       = price_df.iloc[-2] if len(price_df) > 1 else latest
                close_now  = latest.get("close", 0)
                close_prev = prev.get("close", 0)
                change     = close_now - close_prev
                change_pct = (change / close_prev * 100) if close_prev else 0
                direction  = "up" if change >= 0 else "down"
                sign       = "+" if change >= 0 else ""
                st.markdown(f"""
                <div class="metric-row">
                    {_metric_html("Giá đóng cửa", f"{close_now:,.0f}", direction)}
                    {_metric_html("Thay đổi", f"{sign}{change:,.0f} ({sign}{change_pct:.2f}%)", direction)}
                    {_metric_html("RSI(14)", f"{latest.get('RSI', 0):.1f}")}
                    {_metric_html("Khối lượng", f"{latest.get('volume', 0)/1e6:.1f}M")}
                </div>
                """, unsafe_allow_html=True)

            # Biểu đồ intraday khi trong phiên
            if in_session and isinstance(intraday_bars, pd.DataFrame) and not intraday_bars.empty and "close" in intraday_bars.columns:
                st.markdown("**📈 Diễn biến giá trong phiên hôm nay (nến 1 phút)**")
                intra_chart = intraday_bars[["close"]].copy().reset_index(drop=True)
                st.line_chart(intra_chart, use_container_width=True, height=200)

                # Volume intraday
                if "volume" in intraday_bars.columns:
                    st.markdown("**📊 Khối lượng trong phiên**")
                    vol_chart = intraday_bars[["volume"]].copy().reset_index(drop=True)
                    st.bar_chart(vol_chart, use_container_width=True, height=120)

                refresh_col, _ = st.columns([1, 3])
                with refresh_col:
                    if st.button("🔄 Cập nhật dữ liệu phiên", use_container_width=True):
                        fetcher = get_agent().fetcher
                        new_bars, new_price = build_intraday_data(symbol)
                        st.session_state.single_intraday      = new_bars
                        st.session_state.single_current_price = new_price
                        st.rerun()

            # Biểu đồ giá lịch sử (luôn hiển thị)
            if isinstance(price_df, pd.DataFrame) and not price_df.empty:
                st.markdown("**📈 Lịch sử giá 20 phiên gần nhất**")
                chart_df = price_df[["close"]].copy().reset_index(drop=True)
                st.line_chart(chart_df, use_container_width=True, height=180)

                st.markdown("**📋 Dữ liệu kỹ thuật 20 phiên**")
                show_cols = [c for c in ["time", "open", "high", "low", "close", "volume", "MA5", "MA20", "RSI", "MACD"] if c in price_df.columns]
                display_df = price_df[show_cols].copy()
                if "time" in display_df.columns:
                    display_df["time"] = display_df["time"].astype(str).str[:10]
                st.dataframe(display_df, use_container_width=True, hide_index=True, height=260)

        # ── Follow-up Q&A ──
        st.markdown("---")
        st.markdown('<div class="followup-box">', unsafe_allow_html=True)
        st.markdown("#### 💬 Hỏi thêm về mã vừa phân tích")
        fu_col1, fu_col2 = st.columns([4, 1])
        with fu_col1:
            follow_up = st.text_input(
                "Câu hỏi bổ sung",
                placeholder="Ví dụ: Cổ phiếu này có nên mua ở giá hiện tại không?",
                key="single_follow_up",
                label_visibility="collapsed",
            )
        with fu_col2:
            ask_btn = st.button("Gửi", use_container_width=True)

        if ask_btn:
            if not follow_up.strip():
                st.warning("Vui lòng nhập câu hỏi.")
            else:
                with st.spinner("🤖 AI đang trả lời..."):
                    try:
                        answer = get_agent().follow_up(follow_up.strip())
                        st.session_state.follow_up_answer = answer
                    except Exception as exc:
                        st.error(f"Lỗi: {exc}")

        if st.session_state.get("follow_up_answer"):
            st.markdown("**Trả lời từ AI:**")
            st.markdown(st.session_state.follow_up_answer)
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB: PHÂN TÍCH NHIỀU MÃ
# ─────────────────────────────────────────────
def render_batch_tab() -> None:
    st.markdown("#### 📊 Phân tích nhiều mã chứng khoán cùng lúc")
    st.caption("Hệ thống sẽ phân tích các mã song song bằng AI và tổng hợp kết quả vào 1 báo cáo PDF duy nhất.")

    c1, c2 = st.columns([3, 1])
    with c1:
        batch_input = st.text_area(
            "Danh sách mã cổ phiếu",
            placeholder="Nhập các mã cách nhau bởi dấu phẩy hoặc khoảng trắng\nVí dụ: STB, VNM, HPG, MWG, FPT",
            key="batch_input",
            height=110,
        )
    with c2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        run_batch = st.button("🚀 Chạy phân tích", type="primary", use_container_width=True)

    if run_batch:
        symbols = parse_tickers(batch_input)
        agent = get_agent()
        if not symbols:
            st.warning("⚠️ Vui lòng nhập ít nhất 1 mã hợp lệ.")
            return
        if len(symbols) > agent.max_batch_symbols:
            st.info(f"ℹ️ Chỉ phân tích tối đa **{agent.max_batch_symbols} mã** mỗi lần. Hệ thống sẽ bỏ qua các mã dư.")
            symbols = symbols[:agent.max_batch_symbols]

        progress_bar = st.progress(0, text=f"⏳ Đang phân tích {len(symbols)} mã: {', '.join(symbols)}...")
        try:
            results = asyncio.run(analyze_batch(symbols))
            progress_bar.progress(100, text="✅ Hoàn thành!")
        except Exception as exc:
            st.error(f"❌ Lỗi khi chạy phân tích batch: {exc}")
            return

        st.session_state.batch_results = results
        st.session_state.batch_symbols = symbols
        st.session_state.pop("batch_pdf_bytes", None)
        st.session_state.pop("batch_pdf_name", None)

    results = st.session_state.get("batch_results")
    symbols = st.session_state.get("batch_symbols", [])

    if results and symbols:
        st.markdown("---")
        col_pdf, _ = st.columns([1, 3])
        with col_pdf:
            _pdf_download_batch(results)

        st.markdown("<br>", unsafe_allow_html=True)
        for i, symbol in enumerate(symbols):
            text = results.get(symbol, "Không có dữ liệu.")
            with st.expander(f"📌 {symbol} — Kết quả phân tích", expanded=(i == 0)):
                st.markdown(text)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
render_sidebar()
render_header()
render_disclaimer()

single_tab, batch_tab = st.tabs(["🔍 Phân tích 1 mã", "📊 Phân tích nhiều mã"])
with single_tab:
    render_single_tab()
with batch_tab:
    render_batch_tab()

st.caption(
    "Lưu ý pháp lý: Hiệu suất quá khứ không đảm bảo kết quả tương lai. "
    "Sản phẩm chỉ cung cấp thông tin và công cụ hỗ trợ phân tích."
)
