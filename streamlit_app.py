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


st.set_page_config(
    page_title="Agent phan tich chung khoan",
    page_icon="📈",
    layout="wide",
)

if not os.getenv("ANTHROPIC_API_KEY") and "ANTHROPIC_API_KEY" in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = str(st.secrets["ANTHROPIC_API_KEY"])


def get_agent() -> StockAnalysisAgent:
    if "agent" not in st.session_state:
        st.session_state.agent = StockAnalysisAgent()
    return st.session_state.agent


def render_header() -> None:
    st.title("Agent phan tich chung khoan")
    st.caption("Web app MVP de public som: phan tich 1 hoac nhieu ma, xem du lieu va tai PDF ngay tren web.")


def build_price_table(symbol: str) -> pd.DataFrame:
    agent = get_agent()
    df = agent.fetcher.get_ohlcv(symbol)
    df = agent.fetcher.calculate_indicators(df)
    return df.tail(20).copy()


def analyze_single(symbol: str) -> str:
    agent = get_agent()
    agent.reset()
    return agent.analyze(symbol)


async def analyze_batch(symbols: List[str]) -> Dict[str, str]:
    agent = get_agent()
    return await agent.analyze_many(symbols)


def download_single_pdf(symbol: str, result_text: str) -> None:
    pdf_bytes = st.session_state.get("single_pdf_bytes")
    pdf_name = st.session_state.get("single_pdf_name")

    if pdf_bytes is None or pdf_name is None:
        agent = get_agent()
        pdf_path = Path(agent.export_single_report(symbol, result_text))
        pdf_bytes = pdf_path.read_bytes()
        pdf_name = pdf_path.name
        st.session_state.single_pdf_bytes = pdf_bytes
        st.session_state.single_pdf_name = pdf_name

    st.download_button(
        label=f"Tai PDF {symbol}",
        data=pdf_bytes,
        file_name=pdf_name,
        mime="application/pdf",
        use_container_width=True,
    )


def download_batch_pdf(results: Dict[str, str]) -> None:
    pdf_bytes = st.session_state.get("batch_pdf_bytes")
    pdf_name = st.session_state.get("batch_pdf_name")

    if pdf_bytes is None or pdf_name is None:
        agent = get_agent()
        pdf_path = Path(agent.export_batch_report(results))
        pdf_bytes = pdf_path.read_bytes()
        pdf_name = pdf_path.name
        st.session_state.batch_pdf_bytes = pdf_bytes
        st.session_state.batch_pdf_name = pdf_name

    st.download_button(
        label="Tai PDF batch",
        data=pdf_bytes,
        file_name=pdf_name,
        mime="application/pdf",
        use_container_width=True,
    )


def render_single_tab() -> None:
    st.subheader("Phan tich 1 ma")
    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        single_symbol = st.text_input("Nhap ma co phieu", placeholder="VD: STB", key="single_symbol")
        run_single = st.button("Phan tich ngay", type="primary", use_container_width=True)

    if run_single:
        symbol = single_symbol.strip().upper()
        if not symbol:
            st.warning("Vui long nhap ma co phieu.")
            return

        with st.spinner(f"Dang phan tich {symbol}..."):
            try:
                result = analyze_single(symbol)
                price_df = build_price_table(symbol)
            except Exception as exc:
                st.error(f"Khong the phan tich {symbol}: {type(exc).__name__}: {exc}")
                return

        st.session_state.single_result = result
        st.session_state.single_symbol_name = symbol
        st.session_state.single_df = price_df
        st.session_state.pop("single_pdf_bytes", None)
        st.session_state.pop("single_pdf_name", None)

    result = st.session_state.get("single_result")
    symbol = st.session_state.get("single_symbol_name")
    price_df = st.session_state.get("single_df")

    if result and symbol:
        with col_left:
            st.markdown("### Ket qua phan tich")
            st.markdown(result)
        with col_right:
            st.markdown("### Du lieu 20 phien gan nhat")
            if isinstance(price_df, pd.DataFrame) and not price_df.empty:
                show_cols = [
                    col for col in ["time", "open", "high", "low", "close", "volume", "MA5", "MA20", "RSI", "MACD", "Signal"]
                    if col in price_df.columns
                ]
                st.dataframe(price_df[show_cols], use_container_width=True, hide_index=True)
                if "close" in price_df.columns:
                    chart_df = price_df.copy()
                    chart_df.index = range(1, len(chart_df) + 1)
                    st.line_chart(chart_df["close"])
            download_single_pdf(symbol, result)

        st.divider()
        st.markdown("### Hoi tiep theo")
        follow_up = st.text_input("Dat cau hoi bo sung cho ma vua phan tich", key="single_follow_up")
        if st.button("Gui cau hoi", use_container_width=True):
            if not follow_up.strip():
                st.warning("Vui long nhap cau hoi tiep theo.")
            else:
                with st.spinner("Dang tra loi..."):
                    try:
                        answer = get_agent().follow_up(follow_up.strip())
                    except Exception as exc:
                        st.error(f"Khong the tra loi tiep: {type(exc).__name__}: {exc}")
                        return
                st.session_state.follow_up_answer = answer

        if st.session_state.get("follow_up_answer"):
            st.markdown(st.session_state.follow_up_answer)


def render_batch_tab() -> None:
    st.subheader("Phan tich nhieu ma")
    batch_input = st.text_area(
        "Nhap danh sach ma, cach nhau boi dau phay hoac khoang trang",
        placeholder="VD: STB, VNM, HPG",
        key="batch_input",
        height=100,
    )
    run_batch = st.button("Chay batch", type="primary", use_container_width=True)

    if run_batch:
        symbols = parse_tickers(batch_input)
        agent = get_agent()
        if not symbols:
            st.warning("Vui long nhap it nhat 1 ma hop le.")
            return
        if len(symbols) > agent.max_batch_symbols:
            st.info(f"Chi phan tich toi da {agent.max_batch_symbols} ma/lần. He thong se cat bot phan du.")
            symbols = symbols[:agent.max_batch_symbols]

        with st.spinner("Dang chay batch..."):
            try:
                results = asyncio.run(analyze_batch(symbols))
            except Exception as exc:
                st.error(f"Khong the chay batch: {type(exc).__name__}: {exc}")
                return
        st.session_state.batch_results = results
        st.session_state.batch_symbols = symbols
        st.session_state.pop("batch_pdf_bytes", None)
        st.session_state.pop("batch_pdf_name", None)

    results = st.session_state.get("batch_results")
    symbols = st.session_state.get("batch_symbols", [])
    if results:
        download_batch_pdf(results)
        for symbol in symbols:
            with st.expander(f"Ket qua {symbol}", expanded=(symbol == symbols[0])):
                st.markdown(results.get(symbol, "Khong co du lieu."))


def render_sidebar() -> None:
    st.sidebar.header("Thong tin")
    st.sidebar.write("- Chay nhanh de demo va public som")
    st.sidebar.write("- Backend dang dung truc tiep logic Agent hien tai")
    st.sidebar.write("- API key van nam o server qua file .env")
    if st.sidebar.button("Reset hoi thoai", use_container_width=True):
        if "agent" in st.session_state:
            st.session_state.agent.reset()
        for key in [
            "single_result",
            "single_symbol_name",
            "single_df",
            "single_follow_up",
            "follow_up_answer",
            "single_pdf_bytes",
            "single_pdf_name",
            "batch_results",
            "batch_symbols",
            "batch_pdf_bytes",
            "batch_pdf_name",
        ]:
            st.session_state.pop(key, None)
        st.sidebar.success("Da reset.")


render_header()
render_sidebar()

single_tab, batch_tab = st.tabs(["1 ma", "Nhieu ma"])
with single_tab:
    render_single_tab()
with batch_tab:
    render_batch_tab()
