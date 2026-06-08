from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

from localllm.config import get_settings
from localllm.media.convert import (
    AUDIO_INPUT_SUFFIXES,
    DOC_EXTRACT_SUFFIXES,
    convert_audio_to_wav,
    ffmpeg_available,
    safe_media_path,
)
from localllm.media.documents import extract_text_file
from localllm.media.pdf import extract_text_from_pdf, pdf_needs_vision_ocr, render_pdf_pages


@tool
def convert_audio_file(path: str, output_dir: str = "") -> str:
    """Convert mp3/m4a/flac/ogg/webm/aac to 16 kHz mono WAV for Gemma multimodal audio."""
    source = safe_media_path(path)
    if source.suffix.lower() not in AUDIO_INPUT_SUFFIXES:
        return json.dumps({
            "error": f"Not an audio file ({source.suffix}). "
            f"Supported: {', '.join(sorted(AUDIO_INPUT_SUFFIXES))}",
        })
    out_dir = safe_media_path(output_dir) if output_dir.strip() else None
    try:
        wav_path = convert_audio_to_wav(source, out_dir=out_dir)
    except Exception as exc:
        hint = ""
        if source.suffix.lower() in {".m4a", ".mp3", ".aac", ".webm"} and not ffmpeg_available():
            hint = " Run `pip install imageio-ffmpeg` for bundled local ffmpeg."
        return json.dumps({"error": str(exc), "hint": hint.strip()})
    return json.dumps({
        "source": str(source),
        "wav_path": str(wav_path),
        "sample_rate": 16000,
        "channels": 1,
    })


@tool
def extract_pdf_text(path: str, max_pages: int = 20) -> str:
    """Extract text from a PDF; reports when vision/OCR page images are needed."""
    pdf_path = safe_media_path(path)
    if pdf_path.suffix.lower() != ".pdf":
        return json.dumps({"error": "Path must be a .pdf file"})
    settings = get_settings()
    pages = max(1, min(int(max_pages), settings.ocr.max_pages))
    text = extract_text_from_pdf(pdf_path, max_pages=pages)
    if text.strip():
        return json.dumps({
            "pdf": str(pdf_path),
            "mode": "text",
            "chars": len(text),
            "text": text[:12000],
        })
    needs_vision = pdf_needs_vision_ocr(pdf_path, max_pages=min(3, pages))
    return json.dumps({
        "pdf": str(pdf_path),
        "mode": "vision" if needs_vision else "empty",
        "message": (
            "Scanned PDF — render page images for vision OCR."
            if needs_vision
            else "No extractable text in PDF."
        ),
    })


@tool
def extract_document_text(path: str) -> str:
    """Extract plain text from txt/md/docx/csv/json/yaml/html and similar files."""
    doc_path = safe_media_path(path)
    suffix = doc_path.suffix.lower()
    if suffix not in DOC_EXTRACT_SUFFIXES:
        return json.dumps({
            "error": f"Unsupported document type ({suffix}). "
            f"Supported: {', '.join(sorted(DOC_EXTRACT_SUFFIXES))}",
        })
    try:
        text = extract_text_file(doc_path)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps({
        "path": str(doc_path),
        "chars": len(text),
        "text": text[:12000],
    })


@tool
def render_pdf_page_images(path: str, max_pages: int = 5, output_dir: str = "") -> str:
    """Render PDF pages to PNG images for vision OCR (scanned documents)."""
    pdf_path = safe_media_path(path)
    if pdf_path.suffix.lower() != ".pdf":
        return json.dumps({"error": "Path must be a .pdf file"})
    pages = max(1, min(int(max_pages), 10))
    if output_dir.strip():
        out = safe_media_path(output_dir)
    else:
        out = pdf_path.parent / f"{pdf_path.stem}_pages"
    out.mkdir(parents=True, exist_ok=True)
    images = render_pdf_pages(pdf_path, out, max_pages=pages)
    return json.dumps({
        "pdf": str(pdf_path),
        "page_images": [str(p) for p in images],
        "count": len(images),
    })