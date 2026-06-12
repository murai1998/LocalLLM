from __future__ import annotations

import io
import wave
from functools import lru_cache
from pathlib import Path
from typing import Final

from localllm.config import ROOT, get_settings

try:
    from piper import PiperVoice
    from piper.download_voices import download_voice

    PIPER_AVAILABLE = True
except ImportError:
    PiperVoice = None
    download_voice = None
    PIPER_AVAILABLE = False

# Local Piper voices — 2–3 per language (offline after first download).
VOICE_OPTIONS: Final[dict[str, list[dict[str, str]]]] = {
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
    "nl": [
        {"id": "nl_pim", "label": "Pim (Netherlands, male)", "voice": "nl_NL-pim-medium"},
        {"id": "nl_ronnie", "label": "Ronnie (Netherlands)", "voice": "nl_NL-ronnie-medium"},
        {"id": "nl_nathalie", "label": "Nathalie (Belgium, female)", "voice": "nl_BE-nathalie-medium"},
    ],
    "uk": [
        {"id": "uk_ukrainian", "label": "Ukrainian TTS (neutral)", "voice": "uk_UA-ukrainian_tts-medium"},
        {"id": "uk_tetiana", "label": "Tetiana (female, high quality)", "voice": "uk_UA-tetiana-high"},
        {"id": "uk_mykyta", "label": "Mykyta (male, high quality)", "voice": "uk_UA-mykyta-high"},
    ],
}
# No Piper voices exist for ja/ko; cross-language fallbacks produce gibberish,
# so those languages report tts_supported() == False instead.

_voice_name_by_id: dict[str, str] = {
    option["id"]: option["voice"]
    for options in VOICE_OPTIONS.values()
    for option in options
}


def voice_options_for_language(language: str) -> list[dict[str, str]]:
    code = language.split("-")[0].lower()
    return VOICE_OPTIONS.get(code, [])


def tts_supported(language: str) -> bool:
    return bool(voice_options_for_language(language))


def resolve_piper_voice_name(*, language: str, voice_id: str | None = None) -> str:
    if voice_id and voice_id in _voice_name_by_id:
        return _voice_name_by_id[voice_id]
    options = voice_options_for_language(language)
    if not options:
        raise ValueError(f"No local Piper voice available for language '{language}'")
    return options[0]["voice"]


def _model_dir() -> Path:
    settings = get_settings()
    p = Path(settings.tts.model_dir)
    return p if p.is_absolute() else ROOT / p


def _ensure_voice_files(voice_name: str) -> Path:
    if not PIPER_AVAILABLE or download_voice is None:
        raise RuntimeError("piper-tts is not installed. Run: pip install piper-tts")

    model_dir = _model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = model_dir / f"{voice_name}.onnx"
    if not onnx_path.is_file():
        try:
            download_voice(voice_name, model_dir)
        except Exception as exc:
            raise RuntimeError(
                f"Piper voice '{voice_name}' is not cached in {model_dir} and could not "
                f"be downloaded ({exc.__class__.__name__}: {exc}). "
                "Pre-fetch voices while online with: localllm-download --voices"
            ) from exc
    if not onnx_path.is_file():
        raise FileNotFoundError(
            f"Piper voice '{voice_name}' not found in {model_dir}. "
            "Pre-fetch voices while online with: localllm-download --voices"
        )
    return onnx_path


def download_voices(languages: list[str] | None = None) -> list[str]:
    """Pre-download Piper voices so runtime stays fully offline."""
    codes = languages or sorted(VOICE_OPTIONS)
    downloaded: list[str] = []
    for code in codes:
        for option in voice_options_for_language(code):
            _ensure_voice_files(option["voice"])
            downloaded.append(option["voice"])
    return downloaded


@lru_cache(maxsize=8)
def _load_voice(voice_name: str, use_cuda: bool) -> PiperVoice:
    if PiperVoice is None:
        raise RuntimeError("piper-tts is not installed")
    model_path = _ensure_voice_files(voice_name)
    return PiperVoice.load(model_path, use_cuda=use_cuda)


def _chunks_to_wav(voice: PiperVoice, text: str) -> bytes:
    sample_rate = voice.config.sample_rate
    pcm = bytearray()
    for chunk in voice.synthesize(text):
        pcm.extend(chunk.audio_int16_bytes)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(pcm))
    return buf.getvalue()


def warmup_tts(*, language: str = "en", voice_id: str | None = None) -> bool:
    """Load a Piper voice into memory (offline after model is cached)."""
    if not PIPER_AVAILABLE or not tts_supported(language):
        return False
    voice_name = resolve_piper_voice_name(language=language, voice_id=voice_id)
    _load_voice(voice_name, get_settings().tts.use_cuda)
    return True


def synthesize_speech(
    text: str,
    *,
    language: str,
    voice_id: str | None = None,
) -> bytes:
    if not text.strip():
        raise ValueError("Cannot synthesize empty text")
    voice_name = resolve_piper_voice_name(language=language, voice_id=voice_id)
    voice = _load_voice(voice_name, get_settings().tts.use_cuda)
    return _chunks_to_wav(voice, text.strip())
