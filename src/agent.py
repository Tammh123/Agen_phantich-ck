import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import asyncio
import anthropic
import os
from typing import Dict, List
from dotenv import load_dotenv
from knowledge_loader import KnowledgeLoader
from report_exporter import ReportExporter
from stock_data import StockDataFetcher

load_dotenv()


class StockAnalysisAgent:
    def __init__(self):
        self.client    = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.knowledge = KnowledgeLoader()
        self.fetcher   = StockDataFetcher()
        self.exporter = ReportExporter()
        self.system_prompt = self.knowledge.get_system_prompt()
        self.conversation_history = []
        self.max_batch_symbols = 5

    @staticmethod
    def _extract_response_text(response: object) -> str:
        """Extract full text from all response blocks; fallback if empty."""
        chunks: List[str] = []
        for block in getattr(response, "content", []) or []:
            text_val = getattr(block, "text", None)
            if isinstance(text_val, str) and text_val.strip():
                chunks.append(text_val.strip())

        combined = "\n\n".join(chunks).strip()
        if combined:
            return combined
        return "Khong nhan duoc noi dung phan tich tu model. Vui long thu lai."

    @staticmethod
    def _build_analysis_prompt(data_text: str, intraday_text: str = "") -> str:
        intraday_section = ""
        if intraday_text:
            intraday_section = f"""

{intraday_text}

⚡ LƯU Ý: Dữ liệu trên được lấy TRONG PHIÊN GIAO DỊCH đang diễn ra.
Hãy phân tích cả diễn biến ngày hôm nay và đưa ra nhận xét tức thời.
"""
        return f"""
{data_text}{intraday_section}

Hãy phân tích và đưa ra khuyến nghị theo cấu trúc:

1. **NHẬN DIỆN MÔ HÌNH NẾN** (3-5 phiên gần nhất)
2. **XU HƯỚNG HIỆN TẠI** (ngắn/trung hạn)
3. **CÁC CHỈ BÁO KỸ THUẬT** (RSI, MACD, BB)
4. **VÙNG HỖ TRỢ / KHÁNG CỰ** quan trọng
5. **DIỄN BIẾN PHIÊN HÔM NAY** (nếu đang trong phiên)
   - Xu hướng trong ngày (tăng/giảm/đi ngang)
   - Độ mạnh của volume so với trung bình
   - Áp lực mua/bán tức thời
6. **KHUYẾN NGHỊ** (MUA / BÁN / CHỜ)
   - Điểm vào lệnh
   - Mục tiêu (T1, T2)
   - Stop-loss
   - Tỷ lệ Risk/Reward
7. **MỨC ĐỘ TỰ TIN**: X/10
"""

    def _analyze_single_symbol(self, symbol: str) -> str:
        """Phân tích độc lập một mã, dùng cho xử lý song song."""
        df = self.fetcher.get_ohlcv(symbol)
        df = self.fetcher.calculate_indicators(df)
        data_text = self.fetcher.to_text_summary(df, symbol)

        intraday_text = ""
        if self.fetcher.is_trading_session():
            try:
                intraday_text = self.fetcher.intraday_text_summary(symbol)
            except Exception:
                intraday_text = ""

        user_message = self._build_analysis_prompt(data_text, intraday_text)

        response = self.client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2048,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )
        analysis_result = self._extract_response_text(response)
        return analysis_result

    def analyze(self, symbol: str) -> str:
        """Phân tích một mã chứng khoán"""
        print(f"\n📊 Đang lấy dữ liệu {symbol}...")

        # Lấy và tính toán dữ liệu
        df = self.fetcher.get_ohlcv(symbol)
        df = self.fetcher.calculate_indicators(df)
        data_text = self.fetcher.to_text_summary(df, symbol)

        # Nếu đang trong phiên, bổ sung dữ liệu intraday
        intraday_text = ""
        if self.fetcher.is_trading_session():
            try:
                intraday_text = self.fetcher.intraday_text_summary(symbol)
            except Exception:
                intraday_text = ""

        # Tạo prompt phân tích
        user_message = self._build_analysis_prompt(data_text, intraday_text)

        self.conversation_history.append({"role": "user", "content": user_message})

        print(f"🤖 Agent đang phân tích {symbol}...\n")

        response = self.client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2048,
            system=self.system_prompt,
            messages=self.conversation_history
        )

        analysis_result = self._extract_response_text(response)
        self.conversation_history.append({"role": "assistant", "content": analysis_result})

        return analysis_result

    async def analyze_many(self, symbols: List[str]) -> Dict[str, str]:
        """Phân tích nhiều mã cùng lúc bằng asyncio.gather."""
        normalized = [s.strip().upper() for s in symbols if s.strip()]
        normalized = list(dict.fromkeys(normalized))
        if not normalized:
            return {}

        async def run_symbol(batch_symbol: str):
            print(f"\n📊 Đang xử lý {batch_symbol}...")
            return await asyncio.to_thread(self._analyze_single_symbol, batch_symbol)

        tasks = [run_symbol(item_symbol) for item_symbol in normalized]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        mapped: Dict[str, str] = {}
        for item_symbol, item in zip(normalized, gathered):
            if isinstance(item, Exception):
                mapped[item_symbol] = f"Lỗi khi phân tích {item_symbol}: {type(item).__name__}: {item}"
            else:
                mapped[item_symbol] = item
        return mapped

    def follow_up(self, question: str) -> str:
        """Hỏi thêm về phân tích vừa rồi (multi-turn)"""
        self.conversation_history.append({"role": "user", "content": question})

        response = self.client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            system=self.system_prompt,
            messages=self.conversation_history
        )

        follow_up_result = self._extract_response_text(response)
        self.conversation_history.append({"role": "assistant", "content": follow_up_result})
        return follow_up_result

    def reset(self):
        """Reset conversation cho mã mới"""
        self.conversation_history = []

    def export_single_report(self, symbol: str, analysis_text: str) -> str:
        """Xuất PDF cho phân tích một mã."""
        output_path = self.exporter.export_single(symbol=symbol, analysis_text=analysis_text)
        return str(output_path)

    def export_batch_report(self, result_map: Dict[str, str]) -> str:
        """Xuất PDF cho phân tích nhiều mã."""
        output_path = self.exporter.export_batch(results=result_map)
        return str(output_path)


def parse_tickers(raw_input: str) -> List[str]:
    """Parse comma/space separated tickers and remove duplicates."""
    normalized = raw_input.replace(" ", ",")
    tickers = [s.strip().upper() for s in normalized.split(",") if s.strip()]
    return list(dict.fromkeys(tickers))


def run_batch_mode(active_agent: StockAnalysisAgent, tickers: List[str]) -> None:
    if len(tickers) > active_agent.max_batch_symbols:
        print(
            f"\n⚠️ Bạn nhập {len(tickers)} mã. "
            f"Hệ thống chỉ phân tích tối đa {active_agent.max_batch_symbols} mã/lần."
        )
        tickers = tickers[:active_agent.max_batch_symbols]

    print("\n🤖 Agent đang phân tích nhiều mã đồng thời...\n")
    batch_results = asyncio.run(active_agent.analyze_many(tickers))
    for output_symbol, batch_result in batch_results.items():
        print("=" * 60)
        print(f"KẾT QUẢ {output_symbol}")
        print("=" * 60)
        print(batch_result)

    save_batch = input("\nXuất PDF cho toàn bộ kết quả batch? (y/n): ").strip().lower()
    if save_batch in {"y", "yes"}:
        batch_pdf_path = active_agent.export_batch_report(batch_results)
        print(f"\n✅ Đã xuất báo cáo PDF: {batch_pdf_path}")


# ===== MAIN =====
if __name__ == "__main__":
    agent = StockAnalysisAgent()

    print("=" * 60)
    print("   AI AGENT PHÂN TÍCH CHỨNG KHOÁN - VNSTOCK")
    print("=" * 60)

    # Command-line mode: python agent.py STB,VNM,HPG
    if len(sys.argv) > 1:
        cli_input = " ".join(sys.argv[1:]).strip()
        cli_tickers = parse_tickers(cli_input)
        if not cli_tickers:
            print("Không tìm thấy mã hợp lệ từ tham số dòng lệnh.")
            sys.exit(1)

        if len(cli_tickers) > 1:
            run_batch_mode(agent, cli_tickers)
            sys.exit(0)

        one_ticker = cli_tickers[0]
        one_result = agent.analyze(one_ticker)
        print(one_result)
        save_one = input("\nXuất PDF cho mã này? (y/n): ").strip().lower()
        if save_one in {"y", "yes"}:
            pdf_path = agent.export_single_report(one_ticker, one_result)
            print(f"\n✅ Đã xuất báo cáo PDF: {pdf_path}")
        sys.exit(0)

    while True:
        symbol_input = input("\nNhập mã CK (VD: VNM hoặc VNM,HPG,VIC) hoặc 'quit': ").strip()
        if symbol_input.upper() == "QUIT":
            break

        requested_tickers = parse_tickers(symbol_input)
        if len(requested_tickers) > 1:
            run_batch_mode(agent, requested_tickers)
            continue

        if not requested_tickers:
            continue

        single_ticker = requested_tickers[0]

        # Phân tích chính
        single_result = agent.analyze(single_ticker)
        print(single_result)

        save_single = input("\nXuất PDF cho mã này? (y/n): ").strip().lower()
        if save_single in {"y", "yes"}:
            pdf_path = agent.export_single_report(single_ticker, single_result)
            print(f"\n✅ Đã xuất báo cáo PDF: {pdf_path}")

        # Multi-turn hỏi thêm
        while True:
            follow = input("\nHỏi thêm (hoặc Enter để phân tích mã khác): ").strip()
            if not follow:
                agent.reset()
                break
            print(agent.follow_up(follow))
