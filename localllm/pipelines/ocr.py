from __future__ import annotations

import json
import re
from pathlib import Path

from localllm.chat.engine import ChatEngine
from localllm.config import get_settings
from localllm.media.images import ensure_local_image, is_image
from localllm.media.pdf import (
    extract_text_from_pdf,
    pdf_needs_vision_ocr,
    render_pdf_pages,
)
from localllm.model.prompts import OCR_SYSTEM, OCR_USER_TEMPLATE


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def ocr_image(engine: ChatEngine, image_path: Path, instructions: str = "") -> dict:
    ensure_local_image(image_path)
    engine.reset(system_prompt=OCR_SYSTEM)
    prompt = OCR_USER_TEMPLATE.format(instructions=instructions or "none")
    raw = engine.send_with_images(prompt, [image_path], max_tokens=2048)
    try:
        return _parse_json_response(raw)
    except json.JSONDecodeError:
        return {"full_text": raw, "blocks": [], "parse_error": True}


def process_path(
    path: Path,
    *,
    instructions: str = "",
    engine: ChatEngine | None = None,
) -> dict:
    settings = get_settings()
    engine = engine or ChatEngine(system_prompt=OCR_SYSTEM)

    if is_image(path):
        return {"source": str(path), "mode": "vision_ocr", "result": ocr_image(engine, path, instructions)}

    if path.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(path, max_pages=settings.ocr.max_pages)
        if text.strip() and not pdf_needs_vision_ocr(path):
            return {
                "source": str(path),
                "mode": "pymupdf_text",
                "result": {"full_text": text, "blocks": []},
            }
        import tempfile

        page_results: list[dict] = []
        with tempfile.TemporaryDirectory(prefix="localllm_pdf_") as tmp:
            pages = render_pdf_pages(
                path,
                Path(tmp),
                max_pages=settings.ocr.max_pages,
            )
            for page_path in pages:
                page_results.append(ocr_image(engine, page_path, instructions))
        full = "\n\n".join(
            r.get("full_text") or json.dumps(r, ensure_ascii=False) for r in page_results
        )
        return {
            "source": str(path),
            "mode": "vision_ocr_pages",
            "result": {"pages": page_results, "full_text": full},
        }

    raise ValueError(f"Unsupported OCR input: {path}")