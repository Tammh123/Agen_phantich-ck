import pandas as pd
from vnstock import Vnstock  # Cho chứng khoán Việt Nam


_VNSTOCK_SOURCES = ["TCBS", "VCI", "SSI"]


class StockDataFetcher:
    def __init__(self):
        self.stock = Vnstock()

    def get_ohlcv(self, symbol: str, period: int = 120) -> pd.DataFrame:
        """Lấy dữ liệu nến ngày, thử nhiều nguồn nếu nguồn chính bị chặn"""
        from datetime import datetime, timedelta
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=period)).strftime("%Y-%m-%d")

        last_err = None
        for source in _VNSTOCK_SOURCES:
            try:
                df = self.stock.stock(symbol=symbol, source=source).quote.history(
                    start=start, end=end, interval="1D"
                )
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                last_err = e
                continue

        # Fallback: yfinance (hoạt động từ bất kỳ IP nào)
        try:
            return self._get_ohlcv_yfinance(symbol, start, end)
        except Exception as e:
            last_err = e

        raise ConnectionError(
            f"Không thể lấy dữ liệu {symbol} từ tất cả nguồn. Lỗi cuối: {last_err}"
        )

    def _get_ohlcv_yfinance(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fallback dùng yfinance với mã .VN"""
        import yfinance as yf
        ticker = f"{symbol.upper()}.VN"
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"yfinance không tìm thấy dữ liệu cho {ticker}")
        df = df.reset_index()
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        df = df.rename(columns={"date": "time"})
        return df

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
