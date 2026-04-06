from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from fpdf import FPDF
except ModuleNotFoundError:
    FPDF = None


_FPDFBase = FPDF if FPDF is not None else object


class StyledPDF(_FPDFBase):
    """PDF with simple branded header/footer."""

    def __init__(self, report_title: str):
        if FPDF is None:
            raise ModuleNotFoundError("Thiếu package fpdf2. Cài bằng lệnh: pip install fpdf2")
        super().__init__()
        self.report_title = report_title
        self.brand_color = (36, 83, 181)
        self.muted_color = (104, 112, 125)
        self.show_header_footer = True

    def header(self):
        if not self.show_header_footer:
            return
        self.set_font("Helvetica", size=9)
        self.set_text_color(*self.muted_color)
        self.cell(0, 6, self.report_title, new_x="LMARGIN", new_y="NEXT", align="R")
        self.set_draw_color(224, 229, 237)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        if not self.show_header_footer:
            return
        self.set_y(-12)
        self.set_draw_color(224, 229, 237)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(1)
        self.set_font("Helvetica", size=9)
        self.set_text_color(*self.muted_color)
        self.cell(0, 8, "Copyright Tammh AI MASTER", align="L")
        self.set_y(-11)
        self.cell(0, 8, f"Trang {self.page_no()}", align="R")


class ReportExporter:
    """Export stock analysis reports to PDF files."""

    def __init__(self, output_dir: str = "reports"):
        base_dir = Path(__file__).resolve().parent.parent
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = base_dir / output_path
        self.output_dir = output_path
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_single(self, symbol: str, analysis_text: str) -> Path:
        """Export one analysis report and return the PDF path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"report_{symbol.upper()}_{timestamp}.pdf"
        output_path = self.output_dir / file_name
        normalized_text = analysis_text.strip() if analysis_text and analysis_text.strip() else "Khong co noi dung phan tich."
        self._build_pdf(
            output_path=output_path,
            title=f"Bao cao phan tich: {symbol.upper()}",
            sections=[(symbol.upper(), normalized_text)],
            with_cover=False,
            with_toc=False,
        )
        return output_path

    def export_batch(self, results: Dict[str, str]) -> Path:
        """Export many symbol analyses into one PDF and return the path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"report_batch_{timestamp}.pdf"
        sections = []
        for symbol, text in results.items():
            normalized_text = text.strip() if isinstance(text, str) and text.strip() else "Khong co noi dung phan tich."
            sections.append((symbol.upper(), normalized_text))
        self._build_pdf(
            output_path=output_path,
            title="Bao cao phan tich nhieu ma",
            sections=sections,
            with_cover=True,
            with_toc=True,
        )
        return output_path

    def _build_pdf(
        self,
        output_path: Path,
        title: str,
        sections: Iterable[Tuple[str, str]],
        with_cover: bool,
        with_toc: bool,
    ) -> None:
        if FPDF is None:
            raise ModuleNotFoundError(
                "Thiếu package fpdf2. Cài bằng lệnh: pip install fpdf2"
            )

        pdf = StyledPDF(report_title=title)
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_margins(14, 12, 14)
        font_name = self._register_font(pdf)
        section_items = list(sections)

        if with_cover:
            self._add_cover_page(pdf, font_name, title, section_items)

        if with_toc:
            self._add_toc_page(pdf, font_name, section_items)

        if not with_cover and not with_toc:
            pdf.add_page()
            self._draw_intro_block(pdf, font_name, title)

        for symbol, text in section_items:
            if with_cover or with_toc:
                pdf.add_page()
            self._render_symbol_title(pdf, font_name, symbol)
            self._render_section_content(pdf, font_name, text)
            pdf.ln(3)

        self._add_disclaimer_block(pdf, font_name)

        pdf.output(str(output_path))

    def _draw_intro_block(self, pdf: StyledPDF, font_name: str, title: str) -> None:
        pdf.set_fill_color(245, 248, 255)
        pdf.set_draw_color(221, 229, 245)
        pdf.set_text_color(26, 39, 68)
        pdf.set_line_width(0.4)
        pdf.rect(pdf.l_margin, pdf.get_y(), pdf.w - pdf.l_margin - pdf.r_margin, 24, style="DF")
        pdf.ln(4)
        pdf.set_x(pdf.l_margin + 4)
        pdf.set_font(font_name, size=16)
        pdf.cell(0, 7, self._safe_text(title, font_name), new_x="LMARGIN", new_y="NEXT")

        generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.set_x(pdf.l_margin + 4)
        pdf.set_font(font_name, size=10)
        pdf.set_text_color(77, 89, 117)
        pdf.cell(0, 8, self._safe_text(f"Generated at: {generated}", font_name), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(6)

    def _add_cover_page(self, pdf: StyledPDF, font_name: str, title: str, sections: List[Tuple[str, str]]) -> None:
        pdf.show_header_footer = False
        pdf.add_page()
        pdf.set_fill_color(36, 83, 181)
        pdf.rect(0, 0, pdf.w, pdf.h, style="F")

        pdf.set_text_color(255, 255, 255)
        pdf.set_font(font_name, size=11)
        pdf.set_y(40)
        pdf.cell(0, 10, self._safe_text("AI STOCK ANALYSIS REPORT", font_name), new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font(font_name, size=24)
        pdf.cell(0, 14, self._safe_text(title.upper(), font_name), new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(8)

        generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        symbols = ", ".join(symbol for symbol, _ in sections)
        pdf.set_font(font_name, size=12)
        pdf.cell(0, 8, self._safe_text(f"Generated at: {generated}", font_name), new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.cell(0, 8, self._safe_text(f"So luong ma: {len(sections)}", font_name), new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.multi_cell(0, 8, self._safe_text(f"Danh sach ma: {symbols}", font_name), align="C")

        pdf.show_header_footer = True

    def _add_toc_page(self, pdf: StyledPDF, font_name: str, sections: List[Tuple[str, str]]) -> None:
        pdf.add_page()
        pdf.set_text_color(23, 34, 58)
        pdf.set_font(font_name, size=18)
        pdf.cell(0, 10, self._safe_text("Muc luc", font_name), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        pdf.set_font(font_name, size=11)
        for idx, (symbol, _) in enumerate(sections, start=1):
            pdf.set_fill_color(247, 250, 255 if idx % 2 else 242)
            label = f"{idx}. Phan tich ma {symbol}"
            pdf.cell(0, 8, self._safe_text(label, font_name), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    def _add_disclaimer_block(self, pdf: StyledPDF, font_name: str) -> None:
        pdf.ln(4)
        pdf.set_draw_color(241, 196, 15)
        pdf.set_fill_color(255, 248, 230)
        block_h = 24
        pdf.rect(pdf.l_margin, pdf.get_y(), pdf.w - pdf.l_margin - pdf.r_margin, block_h, style="DF")

        pdf.set_xy(pdf.l_margin + 3, pdf.get_y() + 2)
        pdf.set_text_color(102, 60, 0)
        pdf.set_font(font_name, size=10)
        title = "MIEN TRU TRACH NHIEM"
        body = (
            "Bao cao nay chi mang tinh tham khao, khong phai khuyen nghi mua/ban chung khoan. "
            "Nha dau tu tu chiu trach nhiem voi quyet dinh giao dich va can tu danh gia rui ro."
        )
        pdf.multi_cell(0, 6, self._safe_text(title, font_name), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(font_name, size=9)
        pdf.multi_cell(0, 5, self._safe_text(body, font_name), new_x="LMARGIN", new_y="NEXT")

    def _render_symbol_title(self, pdf: StyledPDF, font_name: str, symbol: str) -> None:
        pdf.set_fill_color(36, 83, 181)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font(font_name, size=12)
        pdf.cell(0, 9, self._safe_text(f"MA CHUNG KHOAN: {symbol}", font_name), new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.ln(1)

    def _render_section_content(self, pdf: StyledPDF, font_name: str, text: str) -> None:
        lines = [self._clean_markdown(line) for line in (text.splitlines() or [""])]
        kv_rows: List[Tuple[str, str]] = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                if kv_rows:
                    self._draw_key_value_table(pdf, font_name, kv_rows)
                    kv_rows = []
                pdf.ln(1.5)
                continue

            if self._looks_like_heading(line):
                if kv_rows:
                    self._draw_key_value_table(pdf, font_name, kv_rows)
                    kv_rows = []
                self._draw_heading(pdf, font_name, line)
                continue

            if "|" in line and line.count("|") >= 1:
                if kv_rows:
                    self._draw_key_value_table(pdf, font_name, kv_rows)
                    kv_rows = []
                self._draw_pipe_row(pdf, font_name, line)
                continue

            if self._is_kv_line(line):
                key, value = line.split(":", 1)
                kv_rows.append((key.strip(), value.strip()))
                continue

            if kv_rows:
                self._draw_key_value_table(pdf, font_name, kv_rows)
                kv_rows = []

            self._draw_paragraph(pdf, font_name, line)

        if kv_rows:
            self._draw_key_value_table(pdf, font_name, kv_rows)

    @staticmethod
    def _clean_markdown(line: str) -> str:
        line = line.replace("**", "")
        line = line.replace("###", "")
        line = line.replace("##", "")
        line = line.replace("#", "")
        return line.strip()

    @staticmethod
    def _looks_like_heading(line: str) -> bool:
        return bool(re.match(r"^\d+\.\s+", line)) or line.isupper()

    @staticmethod
    def _is_kv_line(line: str) -> bool:
        if ":" not in line:
            return False
        key, _ = line.split(":", 1)
        return 1 <= len(key.strip()) <= 35

    def _draw_heading(self, pdf: StyledPDF, font_name: str, line: str) -> None:
        pdf.set_font(font_name, size=11)
        pdf.set_text_color(21, 30, 51)
        pdf.set_fill_color(237, 242, 252)
        pdf.cell(0, 8, self._safe_text(line, font_name), new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.ln(1)

    def _draw_paragraph(self, pdf: StyledPDF, font_name: str, line: str) -> None:
        pdf.set_font(font_name, size=11)
        pdf.set_text_color(35, 43, 59)
        text = line
        if text.startswith(("- ", "* ")):
            text = f"• {text[2:].strip()}"
        pdf.multi_cell(0, 6.5, self._safe_text(text, font_name), new_x="LMARGIN", new_y="NEXT")

    def _draw_key_value_table(self, pdf: StyledPDF, font_name: str, rows: List[Tuple[str, str]]) -> None:
        if not rows:
            return

        left_w = (pdf.w - pdf.l_margin - pdf.r_margin) * 0.32
        right_w = (pdf.w - pdf.l_margin - pdf.r_margin) - left_w

        for idx, (key, value) in enumerate(rows):
            pdf.set_font(font_name, size=10)
            if idx % 2 == 0:
                pdf.set_fill_color(250, 252, 255)
            else:
                pdf.set_fill_color(244, 247, 252)
            pdf.set_draw_color(224, 229, 237)
            pdf.set_text_color(26, 34, 50)
            pdf.cell(left_w, 7, self._safe_text(key, font_name), border=1, fill=True)
            pdf.cell(right_w, 7, self._safe_text(value, font_name), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    def _draw_pipe_row(self, pdf: StyledPDF, font_name: str, line: str) -> None:
        values = [part.strip() for part in line.split("|") if part.strip()]
        if not values:
            return

        total_w = pdf.w - pdf.l_margin - pdf.r_margin
        col_w = total_w / len(values)
        pdf.set_font(font_name, size=10)
        pdf.set_fill_color(246, 249, 255)
        pdf.set_draw_color(224, 229, 237)
        pdf.set_text_color(26, 34, 50)

        for idx, value in enumerate(values):
            is_last = idx == len(values) - 1
            if is_last:
                pdf.cell(0, 7, self._safe_text(value, font_name), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
            else:
                pdf.cell(col_w, 7, self._safe_text(value, font_name), border=1, fill=True)
        pdf.ln(1)

    def _register_font(self, pdf: Any) -> str:
        # Thứ tự ưu tiên: Windows → Linux (Streamlit Cloud) → macOS → tải về cache
        candidates = [
            # Windows
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/tahoma.ttf"),
            Path("C:/Windows/Fonts/segoeui.ttf"),
            # Ubuntu / Debian (Streamlit Cloud)
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
            Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
            Path("/usr/share/fonts/noto/NotoSans-Regular.ttf"),
            Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
            # macOS
            Path("/Library/Fonts/Arial.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        ]
        for font_path in candidates:
            if font_path.exists():
                try:
                    pdf.add_font("CustomUnicode", style="", fname=str(font_path))
                    return "CustomUnicode"
                except Exception:
                    continue

        # Tải DejaVu Sans về cache nếu không tìm thấy font nào
        try:
            font_path = self._download_fallback_font()
            if font_path and font_path.exists():
                pdf.add_font("CustomUnicode", style="", fname=str(font_path))
                return "CustomUnicode"
        except Exception:
            pass

        return "Helvetica"

    @staticmethod
    def _download_fallback_font() -> Path:
        """Tải DejaVu Sans (hỗ trợ tiếng Việt) về thư mục cache."""
        import urllib.request
        cache_dir = Path.home() / ".cache" / "vnstock_ai_fonts"
        cache_dir.mkdir(parents=True, exist_ok=True)
        font_path = cache_dir / "DejaVuSans.ttf"
        if not font_path.exists():
            url = (
                "https://github.com/dejavu-fonts/dejavu-fonts/raw/master"
                "/ttf/DejaVuSans.ttf"
            )
            urllib.request.urlretrieve(url, font_path)  # noqa: S310
        return font_path

    @staticmethod
    def _safe_text(text: str, font_name: str) -> str:
        # Chỉ strip về latin-1 khi hoàn toàn không có font Unicode (Helvetica built-in)
        if font_name == "Helvetica":
            return text.encode("latin-1", errors="replace").decode("latin-1")
        return text
