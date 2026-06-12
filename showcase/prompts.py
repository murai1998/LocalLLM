"""Vendored prompt/preset constants — kept in sync with the full local app.

Source of truth: `localllm/pipelines/translate.py` and `localllm/model/prompts.py`
in https://github.com/murai1998/LocalLLM. `scripts/build_showcase.py` (and the
test suite) assert these stay identical.
"""

from __future__ import annotations

GITHUB_URL = "https://github.com/murai1998/LocalLLM"

LANGUAGE_LABELS: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "pt": "Portuguese",
    "it": "Italian",
    "ko": "Korean",
    "ar": "Arabic",
}

TONE_PRESETS: dict[str, dict[str, str]] = {
    "exact": {
        "label": "Exact",
        "hint": "Literal and neutral — no stylistic flourish.",
        "instruction": (
            "Use a dry, precise tone. Stay literal and neutral. "
            "Do not add warmth, filler, or embellishment."
        ),
    },
    "professional": {
        "label": "Professional",
        "hint": "Clear, polished language for business settings.",
        "instruction": (
            "Use a professional tone. Be clear, polished, and natural for work contexts."
        ),
    },
    "friendly": {
        "label": "Friendly",
        "hint": "Warm and conversational, still accurate.",
        "instruction": (
            "Use a friendly tone. Sound warm and conversational while staying accurate."
        ),
    },
    "cordial": {
        "label": "Cordial",
        "hint": "Polite, gracious, and personable.",
        "instruction": (
            "Use a cordial tone. Be polite, gracious, and personable without being casual."
        ),
    },
}

DEFAULT_TONE = "professional"


def language_label(code: str | None) -> str:
    if not code:
        return "auto-detected"
    return LANGUAGE_LABELS.get(code, code)


def tone_instruction(tone: str) -> str:
    return TONE_PRESETS.get(tone, TONE_PRESETS[DEFAULT_TONE])["instruction"]


def build_translate_messages(
    transcript: str,
    *,
    source_lang: str | None,
    target_lang: str,
    tone: str = DEFAULT_TONE,
    context: list[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Same prompt as the local app, plus the live pipeline's rolling context."""
    source_name = language_label(source_lang)
    target_name = language_label(target_lang)
    system = (
        "You are a simultaneous interpreter. "
        f"Translate from {source_name} to {target_name}. "
        f"{tone_instruction(tone)} "
        "Output ONLY the translation with no commentary, labels, or quotes."
    )
    user = f"Source text:\n{transcript.strip()}"
    if context:
        ctx_lines = "\n".join(f"- {src} → {tgt}" for src, tgt in context)
        user = (
            "Recent segments already translated (context only — do not repeat):\n"
            f"{ctx_lines}\n\n{user}"
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


CHAT_SYSTEM = (
    "You are LocalLLM, a helpful assistant. This is a reduced-capability online demo of "
    f"a fully offline local AI suite ({GITHUB_URL}). Answer concisely."
)

OCR_SYSTEM = (
    "You are a precise OCR engine. Extract ALL text from the image faithfully, "
    "preserving reading order and structure. Use Markdown for layout (headings, "
    "lists, tables). Do not add commentary."
)
