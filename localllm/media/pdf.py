from __future__ import annotations

from pathlib import Path

import fitz


def extract_text_from_pdf(path: Path, *, max_pages: int | None = None) -> str:
    """Extract embedded text from a PDF using PyMuPDF (no OCR)."""
    doc = fitz.open(path)
    try:
        pages = range(doc.page_count)
        if max_pages is not None:
            pages = range(min(doc.page_count, max_pages))
        chunks: list[str] = []
        for i in pages:
            text = doc.load_page(i).get_text("text").strip()
            if text:
                chunks.append(f"--- Page {i + 1} ---\n{text}")
        return "\n\n".join(chunks)
    finally:
        doc.close()


def pdf_needs_vision_ocr(path: Path, *, max_pages: int = 3) -> bool:
    """Heuristic: True if sampled pages have little or no extractable text."""
    sample = extract_text_from_pdf(path, max_pages=max_pages)
    return len(sample.strip()) < 40


def render_pdf_pages(
    path: Path,
    out_dir: Path,
    *,
    max_pages: int | None = None,
    dpi: int = 150,
) -> list[Path]:
    """Render PDF pages to PNG files for vision OCR (local only)."""
    doc = fitz.open(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    images: list[Path] = []
    try:
        count = doc.page_count
        if max_pages is not None:
            count = min(count, max_pages)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for i in range(count):
            pix = doc.load_page(i).get_pixmap(matrix=matrix, alpha=False)
            out = out_dir / f"page_{i + 1:04d}.png"
            pix.save(out)
            images.append(out)
    finally:
        doc.close()
    return images