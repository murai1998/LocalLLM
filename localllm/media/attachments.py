from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from localllm.config import AppSettings, get_settings
from localllm.media.audio import to_wav_16k
from localllm.media.documents import extract_text_file
from localllm.media.images import is_image
from localllm.media.pdf import (
    extract_text_from_pdf,
    pdf_needs_vision_ocr,
    render_pdf_pages,
)

AppMode = Literal["chat", "agent"]
MAX_PDF_TEXT_CHARS = 12_000
MAX_PDF_VISION_PAGES = 5

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
CHAT_AUDIO_SUFFIXES = {".wav"}
AGENT_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac"}
DOC_SUFFIXES = {".txt", ".md", ".docx", ".csv", ".json", ".yaml", ".yml"}
PDF_SUFFIX = ".pdf"

CHAT_ALLOWED = IMAGE_SUFFIXES | CHAT_AUDIO_SUFFIXES | DOC_SUFFIXES | {PDF_SUFFIX}
AGENT_ALLOWED = (
    IMAGE_SUFFIXES
    | AGENT_AUDIO_SUFFIXES
    | DOC_SUFFIXES
    | {PDF_SUFFIX, ".html", ".htm", ".log", ".xml"}
)


class AttachmentError(Exception):
    """User-facing attachment validation or preparation failure."""


@dataclass
class PreparedAttachment:
    name: str
    kind: str
    text_blocks: list[str] = field(default_factory=list)
    image_paths: list[Path] = field(default_factory=list)
    audio_path: Path | None = None
    agent_note: str = ""


def attachment_kind(suffix: str) -> str:
    lower = suffix.lower()
    if lower in IMAGE_SUFFIXES:
        return "image"
    if lower == PDF_SUFFIX:
        return "pdf"
    if lower in CHAT_AUDIO_SUFFIXES | AGENT_AUDIO_SUFFIXES:
        return "audio"
    if lower in DOC_SUFFIXES:
        return "document"
    return "file"


def validate_extension(filename: str, mode: AppMode) -> str | None:
    suffix = Path(filename).suffix.lower()
    allowed = CHAT_ALLOWED if mode == "chat" else AGENT_ALLOWED
    if suffix not in allowed:
        if mode == "chat" and suffix in AGENT_AUDIO_SUFFIXES - CHAT_AUDIO_SUFFIXES:
            return (
                f"**{filename}** — Chat accepts **WAV only** for audio. "
                "Switch to **Agent** mode for mp3/m4a (auto-converted), or convert to WAV first."
            )
        return (
            f"**{filename}** — unsupported in **{mode.title()}** mode "
            f"(`.{suffix.lstrip('.')}`). Remove it or switch to Agent mode."
        )
    return None


def _process_pdf(path: Path, *, max_pages: int) -> tuple[list[str], list[Path]]:
    text_blocks: list[str] = []
    image_paths: list[Path] = []
    text = extract_text_from_pdf(path, max_pages=max_pages)

    if text.strip():
        if len(text) > MAX_PDF_TEXT_CHARS:
            text = text[:MAX_PDF_TEXT_CHARS] + "\n\n[... truncated ...]"
        text_blocks.append(f"### PDF: {path.name}\n{text}")
        return text_blocks, image_paths

    if pdf_needs_vision_ocr(path, max_pages=min(3, max_pages)):
        with tempfile.TemporaryDirectory(prefix="localllm_pdf_") as tmp:
            pages = render_pdf_pages(
                path, Path(tmp), max_pages=min(MAX_PDF_VISION_PAGES, max_pages)
            )
            image_paths.extend(pages)
        text_blocks.append(
            f"### PDF: {path.name}\n"
            f"(scanned document — {len(image_paths)} page image(s) for vision)"
        )
    else:
        text_blocks.append(f"### PDF: {path.name}\n(no extractable text)")

    return text_blocks, image_paths


def prepare_attachment(
    path: Path,
    *,
    mode: AppMode,
    settings: AppSettings | None = None,
) -> PreparedAttachment:
    settings = settings or get_settings()
    suffix = path.suffix.lower()
    name = path.name
    kind = attachment_kind(suffix)

    err = validate_extension(name, mode)
    if err:
        raise AttachmentError(err)

    prepared = PreparedAttachment(name=name, kind=kind)

    if is_image(path):
        prepared.image_paths.append(path)
        prepared.agent_note = f"[Image attachment: {name}]"
        return prepared

    if suffix == PDF_SUFFIX:
        blocks, images = _process_pdf(path, max_pages=settings.ocr.max_pages)
        prepared.text_blocks.extend(blocks)
        prepared.image_paths.extend(images)
        prepared.agent_note = f"[PDF attachment: {name}]"
        return prepared

    if suffix in CHAT_AUDIO_SUFFIXES | AGENT_AUDIO_SUFFIXES:
        try:
            wav_path = to_wav_16k(path)
        except Exception as exc:
            from localllm.media.convert import ffmpeg_available

            hint = ""
            if suffix in {".m4a", ".mp3", ".aac", ".webm"} and not ffmpeg_available():
                hint = " Run `pip install imageio-ffmpeg` or install system ffmpeg for m4a/mp3."
            raise AttachmentError(
                f"**{name}** — could not read or convert audio ({exc}).{hint} "
                "Or re-export as WAV."
            ) from exc
        prepared.audio_path = wav_path
        if suffix != ".wav":
            prepared.agent_note = (
                f"[Audio attachment: {name} → converted to 16 kHz mono WAV for analysis]"
            )
        else:
            prepared.agent_note = f"[Audio attachment: {name}]"
        return prepared

    if suffix in DOC_SUFFIXES:
        try:
            text = extract_text_file(path)
        except Exception as exc:
            raise AttachmentError(f"**{name}** — could not read document: {exc}") from exc
        prepared.text_blocks.append(f"### {name}\n{text}")
        prepared.agent_note = f"[Document attachment: {name}]"
        return prepared

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        raise AttachmentError(f"**{name}** — could not read file: {exc}") from exc
    prepared.text_blocks.append(f"### {name}\n{text[:MAX_PDF_TEXT_CHARS]}")
    prepared.agent_note = f"[File attachment: {name}]"
    return prepared


def prepare_chat_turn(
    prompt: str,
    attachment_paths: list[Path],
    *,
    settings: AppSettings | None = None,
) -> tuple[str, list[Path], Path | None]:
    user_text = prompt.strip()
    image_paths: list[Path] = []
    audio_path: Path | None = None
    extra_text: list[str] = []

    for path in attachment_paths:
        prepared = prepare_attachment(path, mode="chat", settings=settings)
        image_paths.extend(prepared.image_paths)
        extra_text.extend(prepared.text_blocks)
        if prepared.audio_path:
            if audio_path is not None:
                raise AttachmentError(
                    "Chat supports **one audio file** per message. Remove extra audio attachments."
                )
            audio_path = prepared.audio_path

    if extra_text:
        user_text = (user_text + "\n\n" + "\n\n".join(extra_text)).strip()

    return user_text, image_paths, audio_path


def prepare_agent_context(
    prompt: str,
    attachment_paths: list[Path],
    *,
    settings: AppSettings | None = None,
) -> tuple[str, list[Path], Path | None]:
    """Build agent user text plus optional multimodal paths."""
    user_text = prompt.strip()
    image_paths: list[Path] = []
    audio_path: Path | None = None
    notes: list[str] = []

    for path in attachment_paths:
        prepared = prepare_attachment(path, mode="agent", settings=settings)
        image_paths.extend(prepared.image_paths)
        if prepared.text_blocks:
            user_text = (user_text + "\n\n" + "\n\n".join(prepared.text_blocks)).strip()
        if prepared.audio_path:
            audio_path = prepared.audio_path
        if prepared.agent_note:
            notes.append(prepared.agent_note)

    if notes:
        header = "Attached files:\n" + "\n".join(f"- {n}" for n in notes)
        user_text = f"{header}\n\n{user_text}".strip() if user_text else header

    return user_text, image_paths, audio_path


def build_multimodal_content(
    text: str,
    *,
    image_paths: list[Path],
    audio_path: Path | None,
    client,
) -> str | list:
    """OpenAI-compatible user content for Gemma multimodal."""
    if not image_paths and not audio_path:
        return text

    parts: list = []
    for image in image_paths:
        parts.append(client.image_part(image))
    if text:
        parts.append(client.text_part(text))
    if audio_path:
        parts.append(client.audio_part(audio_path))
    return parts


def format_user_error(exc: Exception) -> str:
    if isinstance(exc, AttachmentError):
        return str(exc)
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    if "audio" in lowered or "wav" in lowered or "m4a" in lowered:
        return (
            "Could not process the audio attachment. "
            "In **Chat** mode use **WAV** only; in **Agent** mode mp3/m4a are converted automatically."
        )
    if "400" in message or "bad request" in lowered:
        return (
            "The model rejected this message (often unsupported attachment format). "
            "Check attachments — Chat audio must be **WAV**."
        )
    return f"Something went wrong: {message}"