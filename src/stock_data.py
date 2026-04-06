import requests
import pandas as pd
from vnstock import Vnstock  # Cho chứng khoán Việt Nam


_VNSTOCK_SOURCES = ["TCBS", "VCI", "SSI"]

_TCBS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
    "Origin": "https://tcinvest.tcbs.com.vn",
    "Referer": "https://tcinvest.tcbs.com.vn/",
}


class StockDataFetcher:
    def __init__(self):
        self.stock = Vnstock()

    def get_ohlcv(self, symbol: str, period: int = 120) -> pd.DataFrame:
        """Lấy dữ liệu nến ngày, thử nhiều nguồn nếu nguồn chính bị chặn"""
        from datetime import datetime, timedelta
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=period)).strftime("%Y-%m-%d")

        # 1. vnstock sources
        for source in _VNSTOCK_SOURCES:
            try:
                df = self.stock.stock(symbol=symbol, source=source).quote.history(
                    start=start, end=end, interval="1D"
                )
                if df is not None and not df.empty:
                    return df
            except Exception:
                continue

        # 2. TCBS direct REST API (bypass vnstock, gọi thẳng với browser headers)
        try:
            return self._get_ohlcv_tcbs_direct(symbol, start, end)
        except Exception:
            pass

        # 3. yfinance fallback
        try:
            return self._get_ohlcv_yfinance(symbol, start, end)
        except Exception as e:
            raise ConnectionError(
                f"Không thể lấy dữ liệu {symbol} từ tất cả nguồn. Lỗi cuối: {e}"
            )

    def _get_ohlcv_tcbs_direct(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Gọi thẳng TCBS REST API với browser headers"""
        from datetime import datetime
        from_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
        to_ts = int(datetime.strptime(end, "%Y-%m-%d").timestamp())
        url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
        params = {
            "ticker": symbol.upper(),
            "type": "stock",
            "resolution": "D",
            "from": from_ts,
            "to": to_ts,
        }
        resp = requests.get(url, params=params, headers=_TCBS_HEADERS, timeout=15)
        resp.raise_for_status()
        bars = resp.json().get("data", [])
        if not bars:
            raise ValueError(f"TCBS direct API trả về rỗng cho {symbol}")
        df = pd.DataFrame(bars)
        df = df.rename(columns={"tradingDate": "time"})
        df["time"] = pd.to_datetime(df["time"])
        df.columns = [c.lower() for c in df.columns]
        needed = [c for c in ["time", "open", "high", "low", "close", "volume"] if c in df.columns]
        return df[needed].sort_values("time").reset_index(drop=True)

    def _get_ohlcv_yfinance(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fallback yfinance dùng Ticker.history() (sạch hơn download())"""
        import yfinance as yf
        ticker_str = f"{symbol.upper()}.VN"
        ticker_obj = yf.Ticker(ticker_str)
        df = ticker_obj.history(start=start, end=end, auto_adjust=True)
        if df.empty:
            raise ValueError(f"yfinance không tìm thấy dữ liệu cho {ticker_str}")
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"date": "time"})
        needed = [c for c in ["time", "open", "high", "low", "close", "volume"] if c in df.columns]
        return df[needed]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tính các chỉ báo kỹ thuật"""
        close = df['close']

        # MA
        df['MA5']  = close.rolling(5).mean()
        df['MA20'] = close.rolling(20).mean()
        df['MA50'] = close.rolling(50).mean()

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / loss))

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        df['MACD']   = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(span=9).mean()

        # Bollinger Bands
        df['BB_mid']   = close.rolling(20).mean()
        std            = close.rolling(20).std()
        df['BB_upper'] = df['BB_mid'] + 2 * std
        df['BB_lower'] = df['BB_mid'] - 2 * std

        return df

    def to_text_summary(self, df: pd.DataFrame, symbol: str) -> str:
        """Chuyển dữ liệu thành text cho Agent phân tích"""
        last = df.tail(10)
        lines = [f"=== DỮ LIỆU {symbol} (10 phiên gần nhất) ===\n"]

        for _, row in last.iterrows():
            date_val = str(row.get('time', row.name))[:10]
            lines.append(
                f"Ngay: {date_val} | "
                f"O:{row['open']:.0f} H:{row['high']:.0f} "
                f"L:{row['low']:.0f} C:{row['close']:.0f} "
                f"Vol:{row.get('volume', 0) / 1e6:.1f}M"
            )

        latest = df.iloc[-1]
        lines.append(f"\n=== CHỈ BÁO KỸ THUẬT ===")
        lines.append(f"MA5: {latest.get('MA5', 0):.0f} | MA20: {latest.get('MA20', 0):.0f} | MA50: {latest.get('MA50', 0):.0f}")
        lines.append(f"RSI(14): {latest.get('RSI', 0):.1f}")
        lines.append(f"MACD: {latest.get('MACD', 0):.2f} | Signal: {latest.get('Signal', 0):.2f}")
        lines.append(f"BB Upper: {latest.get('BB_upper', 0):.0f} | BB Lower: {latest.get('BB_lower', 0):.0f}")

        return "\n".join(lines)
