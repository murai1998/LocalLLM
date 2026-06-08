from localllm.tts.piper import (
    PIPER_AVAILABLE,
    VOICE_OPTIONS,
    resolve_piper_voice_name,
    voice_options_for_language,
)


def test_voice_options_for_spanish():
    options = voice_options_for_language("es")
    assert len(options) >= 2
    assert options[0]["voice"].startswith("es_")


def test_resolve_piper_voice_by_id():
    voice = resolve_piper_voice_name(language="es", voice_id="es_sharvard")
    assert voice == "es_ES-sharvard-medium"


def test_resolve_piper_voice_fallback():
    voice = resolve_piper_voice_name(language="es", voice_id=None)
    assert voice == voice_options_for_language("es")[0]["voice"]


def test_voice_options_cover_common_languages():
    for code in ("en", "es", "fr", "de", "ru"):
        assert code in VOICE_OPTIONS


def test_piper_available_after_install():
    assert PIPER_AVAILABLE is True