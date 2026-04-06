import sys
import io
# Streamlit Cloud can provide wrapped streams without `.buffer`.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import asyncio
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
from knowledge_loader import KnowledgeLoader
from report_exporter import ReportExporter
from stock_data import StockDataFetcher

try:
    import anthropic
except ModuleNotFoundError:
    anthropic = None

try:
    import google.generativeai as genai
except ModuleNotFoundError:
    genai = None

load_dotenv()

_GEMINI_MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]


class StockAnalysisAgent:
    def __init__(self, provider: str = "anthropic"):
        self.provider = provider.lower()
        self.knowledge = KnowledgeLoader()
        self.fetcher   = StockDataFetcher()
        self.exporter  = ReportExporter()
        self.system_prompt = self.knowledge.get_system_prompt()
        self.conversation_history = []
        self.max_batch_symbols = 5

        if self.provider not in {"anthropic", "gemini"}:
            raise ValueError(f"Provider không hợp lệ: '{provider}'. Dùng 'anthropic' hoặc 'gemini'.")

        self.client: Optional[object] = None
        self.gemini_model: Optional[object] = None
        self.gemini_model_name: Optional[str] = None
        self.last_provider_used: str = ""

        self._anthropic_error: Optional[str] = None
        self._gemini_error: Optional[str] = None

        self._init_anthropic()
        self._init_gemini()

        if not self._provider_available("anthropic") and not self._provider_available("gemini"):
            raise RuntimeError(
                "Không thể khởi tạo AI provider nào. "
                f"Anthropic: {self._anthropic_error or 'unknown'} | "
                f"Gemini: {self._gemini_error or 'unknown'}"
            )

    def _init_anthropic(self) -> None:
        if anthropic is None:
            self._anthropic_error = "Thiếu package anthropic"
            return
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            self._anthropic_error = "Thiếu ANTHROPIC_API_KEY"
            return
        try:
            self.client = anthropic.Anthropic(api_key=api_key)
            self._anthropic_error = None
        except Exception as exc:
            self.client = None
            self._anthropic_error = str(exc)

    def _init_gemini(self) -> None:
        if genai is None:
            self._gemini_error = "Thiếu package google-generativeai"
            return
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self._gemini_error = "Thiếu GEMINI_API_KEY"
            return
        try:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel(
                model_name=_GEMINI_MODEL_CANDIDATES[0],
                system_instruction=self.system_prompt,
            )
            self.gemini_model_name = _GEMINI_MODEL_CANDIDATES[0]
            self._gemini_error = None
        except Exception as exc:
            self.gemini_model = None
            self.gemini_model_name = None
            self._gemini_error = str(exc)

    def _call_gemini_with_model_fallback(self, messages: List[Dict]) -> str:
        last_exc: Optional[Exception] = None
        gemini_history = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["content"]]})

        for model_name in _GEMINI_MODEL_CANDIDATES:
            try:
                model = self.gemini_model
                if self.gemini_model_name != model_name or model is None:
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        system_instruction=self.system_prompt,
                    )
                chat = model.start_chat(history=gemini_history)
                response = chat.send_message(messages[-1]["content"])
                self.gemini_model = model
                self.gemini_model_name = model_name
                return response.text
            except Exception as exc:
                last_exc = exc
                continue

        raise RuntimeError(
            f"Gemini không khả dụng với các model {_GEMINI_MODEL_CANDIDATES}. Lỗi cuối: {last_exc}"
        ) from last_exc

    def _provider_available(self, provider_name: str) -> bool:
        if provider_name == "anthropic":
            return self.client is not None
        if provider_name == "gemini":
            return self.gemini_model is not None
        return False

    def _call_provider(self, provider_name: str, messages: List[Dict], max_tokens: int) -> str:
        if provider_name == "anthropic":
            if self.client is None:
                raise RuntimeError(self._anthropic_error or "Anthropic không sẵn sàng")
            response = self.client.messages.create(
                model="claude-opus-4-5",
                max_tokens=max_tokens,
                system=self.system_prompt,
                messages=messages,
            )
            return self._extract_response_text(response)

        if self.gemini_model is None and genai is None:
            raise RuntimeError(self._gemini_error or "Gemini không sẵn sàng")
        return self._call_gemini_with_model_fallback(messages)

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

    def _call_ai(self, messages: List[Dict], max_tokens: int = 2048) -> str:
        """Gọi AI model; tự động fallback sang provider còn lại nếu provider chính lỗi."""
        primary = self.provider
        secondary = "gemini" if primary == "anthropic" else "anthropic"

        primary_error: Optional[Exception] = None
        if self._provider_available(primary):
            try:
                result = self._call_provider(primary, messages, max_tokens)
                self.last_provider_used = primary
                return result
            except Exception as exc:
                primary_error = exc
        else:
            primary_error = RuntimeError(
                self._anthropic_error if primary == "anthropic" else self._gemini_error
            )

        if self._provider_available(secondary):
            try:
                fallback_result = self._call_provider(secondary, messages, max_tokens)
                self.last_provider_used = secondary
                return (
                    f"⚠️ Provider {primary} tạm thời lỗi ({type(primary_error).__name__}: {primary_error}). "
                    f"Đã tự chuyển sang {secondary}.\n\n{fallback_result}"
                )
            except Exception as secondary_exc:
                raise RuntimeError(
                    f"Cả 2 provider đều lỗi. "
                    f"{primary}: {type(primary_error).__name__}: {primary_error} | "
                    f"{secondary}: {type(secondary_exc).__name__}: {secondary_exc}"
                ) from secondary_exc

        raise RuntimeError(
            f"Không có provider dự phòng khả dụng. "
            f"Lỗi {primary}: {type(primary_error).__name__}: {primary_error}."
        ) from primary_error

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
        analysis_result = self._call_ai([{"role": "user", "content": user_message}])
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

        analysis_result = self._call_ai(self.conversation_history)
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
        follow_up_result = self._call_ai(self.conversation_history, max_tokens=1024)
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
