"""Echo-stub models for UI rehearsal without torch, a GPU, or model downloads.

Activated with SHOWCASE_FAKE=1 — `app.py` imports this instead of `models.py`.
Every function mimics the real signatures; replies clearly say they are fake.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

GEMMA_ID = "fake/gemma (SHOWCASE_FAKE=1)"
WHISPER_ID = "fake/whisper"

_NOTE = "[fake mode — set SHOWCASE_FAKE=0 and install torch to run real models]"


def transcribe_and_translate(audio, sample_rate, source_lang, target_lang, tone, context):
    seconds = len(audio) / max(sample_rate, 1)
    transcript = f"(fake transcript of {seconds:.1f}s of speech)"
    translation = f"(fake {target_lang} translation, tone={tone}) {_NOTE}"
    time.sleep(0.3)
    return transcript, translation


def transcribe_file(audio, sample_rate, source_lang):
    seconds = len(audio) / max(sample_rate, 1)
    return f"(fake transcript of {seconds:.1f}s audio) {_NOTE}"


def translate_text(text, source_lang, target_lang, tone):
    return f"(fake {target_lang} translation of {len(text)} chars, tone={tone}) {_NOTE}"


def chat_stream(history, image=None) -> Iterator[str]:
    last = history[-1]["content"] if history else ""
    reply = f"Fake echo: {last!r}"
    if image is not None:
        reply += " (and I pretend to see your image)"
    reply += f" {_NOTE}"
    text = ""
    for word in reply.split(" "):
        text = f"{text} {word}".strip()
        time.sleep(0.03)
        yield text


def ocr_images(images, instructions=""):
    return f"(fake OCR of {len(images)} page(s)) {_NOTE}"
