from __future__ import annotations

from pathlib import Path


def extract_text_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".csv", ".json", ".yaml", ".yml"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".docx":
        from docx import Document

        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    raise ValueError(f"Unsupported document type: {suffix}")