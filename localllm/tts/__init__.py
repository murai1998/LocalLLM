from localllm.tts.piper import (
    PIPER_AVAILABLE,
    VOICE_OPTIONS,
    download_voices,
    synthesize_speech,
    tts_supported,
    voice_options_for_language,
    warmup_tts,
)

# Offline-only stack; edge-tts intentionally not exported.
TTS_ENGINE = "piper"
TTS_REQUIRES_INTERNET = False

__all__ = [
    "PIPER_AVAILABLE",
    "TTS_ENGINE",
    "TTS_REQUIRES_INTERNET",
    "VOICE_OPTIONS",
    "download_voices",
    "synthesize_speech",
    "tts_supported",
    "voice_options_for_language",
    "warmup_tts",
]
