import os
from pathlib import Path


class KnowledgeLoader:
    def __init__(self, knowledge_dir: str = None):
        if knowledge_dir is None:
            # Resolve relative to project root (one level up from src/)
            self.knowledge_dir = Path(__file__).parent.parent / "knowledge"
        else:
            self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_base = {}

    def load_all(self) -> str:
        """Load tất cả file .md từ thư mục knowledge"""
        combined = []
        for file in self.knowledge_dir.glob("*.md"):
            content = file.read_text(encoding="utf-8")
            self.knowledge_base[file.stem] = content
            combined.append(f"## {file.stem.upper()}\n{content}")
        return "\n\n---\n\n".join(combined)

    def get_system_prompt(self) -> str:
        knowledge = self.load_all()
        return f"""Bạn là chuyên gia phân tích kỹ thuật chứng khoán ngắn hạn.
Bạn có kiến thức chuyên sâu từ các tài liệu sau:

{knowledge}

Nhiệm vụ: Phân tích biểu đồ nến Nhật và đưa ra khuyến nghị đầu tư ngắn hạn 
(1-5 phiên) dựa trên dữ liệu thực tế. Luôn đề cập mức rủi ro và stop-loss."""
