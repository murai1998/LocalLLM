"""Vendored Piper TTS (CPU-only) — adapted from `localllm/tts/piper.py`.

TTS runs on the Space's CPU so it costs zero GPU quota. Voices download once
into ./voices (ephemeral Space disk) and are cached for the process lifetime.
VOICE_OPTIONS must stay in sync with the full local app (asserted by tests).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    from piper import PiperVoice
    from piper.download_voices import download_voice

    PIPER_AVAILABLE = True
except ImportError:  # keeps the module importable for tests without piper
    PiperVoice = None
    download_voice = None
    PIPER_AVAILABLE = False

VOICES_DIR = Path(os.environ.get("SHOWCASE_VOICES_DIR", "voices"))

# Identical to localllm/tts/piper.py — no cross-language fallbacks (ja/ko have
# no Piper voice and report tts_supported() == False instead of gibberish).
VOICE_OPTIONS: dict[str, list[dict[str, str]]] = {
    "en": [
        {"id": "en_lessac", "label": "Lessac (US, neutral)", "voice": "en_US-lessac-medium"},
        {"id": "en_amy", "label": "Amy (US, female)", "voice": "en_US-amy-medium"},
        {"id": "en_ryan", "label": "Ryan (US, male)", "voice": "en_US-ryan-medium"},
    ],
    "es": [
        {"id": "es_sharvard", "label": "Sharvard (Spain)", "voice": "es_ES-sharvard-medium"},
        {"id": "es_ald", "label": "Ald (Mexico)", "voice": "es_MX-ald-medium"},
        {"id": "es_davefx", "label": "Davefx (Spain)", "voice": "es_ES-davefx-medium"},
    ],
    "fr": [
        {"id": "fr_siwis", "label": "Siwis (female)", "voice": "fr_FR-siwis-medium"},
        {"id": "fr_tom", "label": "Tom (male)", "voice": "fr_FR-tom-medium"},
    ],
    "de": [
        {"id": "de_thorsten", "label": "Thorsten (male)", "voice": "de_DE-thorsten-medium"},
        {"id": "de_ramona", "label": "Ramona (female)", "voice": "de_DE-ramona-low"},
    ],
    "ru": [
        {"id": "ru_irina", "label": "Irina (female)", "voice": "ru_RU-irina-medium"},
        {"id": "ru_denis", "label": "Denis (male)", "voice": "ru_RU-denis-medium"},
    ],
    "pt": [
        {"id": "pt_faber", "label": "Faber (Brazil)", "voice": "pt_BR-faber-medium"},
        {"id": "pt_edresson", "label": "Edresson (Brazil)", "voice": "pt_BR-edresson-low"},
    ],
    "it": [
        {"id": "it_paola", "label": "Paola (female)", "voice": "it_IT-paola-medium"},
        {"id": "it_riccardo", "label": "Riccardo (male)", "voice": "it_IT-riccardo-x_low"},
    ],
    "zh": [
        {"id": "zh_huayan", "label": "Huayan (female)", "voice": "zh_CN-huayan-medium"},
        {"id": "zh_chaowen", "label": "Chaowen (male)", "voice": "zh_CN-chaowen-medium"},
    ],
    "ar": [
        {"id": "ar_kareem", "label": "Kareem", "voice": "ar_JO-kareem-medium"},
    ],
}

_voice_name_by_id = {
    option["id"]: option["voice"]
    for options in VOICE_OPTIONS.values()
    for option in options
}


def voice_options_for_language(language: str) -> list[dict[str, str]]:
    return VOICE_OPTIONS.get(language.split("-")[0].lower(), [])


def tts_supported(language: str) -> bool:
    return bool(voice_options_for_language(language))


def resolve_voice_name(*, language: str, voice_id: str | None = None) -> str:
    options = voice_options_for_language(language)
    # Only honour a voice_id that belongs to the requested language — the live
    # tab's voice dropdown can briefly hold the previous language's voice while
    # the target language changes (e.g. es_sharvard with language="en").
    if voice_id and any(option["id"] == voice_id for option in options):
        return _voice_name_by_id[voice_id]
    if not options:
        raise ValueError(f"No Piper voice available for language '{language}'")
    return options[0]["voice"]


@lru_cache(maxsize=8)
def _load_voice(voice_name: str):
    if not PIPER_AVAILABLE:
        raise RuntimeError("piper-tts is not installed")
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = VOICES_DIR / f"{voice_name}.onnx"
    if not onnx_path.is_file():
        download_voice(voice_name, VOICES_DIR)
    return PiperVoice.load(onnx_path, use_cuda=False)


def synthesize(
    text: str,
    *,
    language: str,
    voice_id: str | None = None,
) -> tuple[int, np.ndarray]:
    """Synthesize speech on CPU → (sample_rate, mono int16 array)."""
    if not text.strip():
        raise ValueError("Cannot synthesize empty text")
    voice = _load_voice(resolve_voice_name(language=language, voice_id=voice_id))
    pcm = bytearray()
    for chunk in voice.synthesize(text.strip()):
        pcm.extend(chunk.audio_int16_bytes)
    return voice.config.sample_rate, np.frombuffer(bytes(pcm), dtype=np.int16)


def warmup(language: str = "en") -> bool:
    """Pre-download/load the default voice for a language (CPU, safe at startup)."""
    if not PIPER_AVAILABLE or not tts_supported(language):
        return False
    _load_voice(resolve_voice_name(language=language))
    return True
