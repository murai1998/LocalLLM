from pathlib import Path
from unittest.mock import MagicMock

from localllm.config import AppSettings, TranslateConfig
from localllm.pipelines.translate import (
    build_translate_messages,
    parse_unified_response,
    retranslate_transcript,
    translate_audio,
    translate_text,
)


def test_build_translate_messages_uses_language_labels():
    messages = build_translate_messages("Hola", source_lang="es", target_lang="en")
    assert messages[0]["role"] == "system"
    assert "Spanish" in messages[0]["content"]
    assert "English" in messages[0]["content"]
    assert messages[1]["content"] == "Source text:\nHola"


def test_build_translate_messages_includes_tone():
    messages = build_translate_messages(
        "Hello",
        source_lang="en",
        target_lang="es",
        tone="cordial",
    )
    assert "cordial" in messages[0]["content"].lower()


def test_parse_unified_response():
    transcript, translation = parse_unified_response(
        "TRANSCRIPT:\nHello there\nTRANSLATION:\nHola"
    )
    assert transcript == "Hello there"
    assert translation == "Hola"


def test_translate_audio_split(monkeypatch, tmp_path: Path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"wav")

    llm = MagicMock()
    llm.chat.return_value = "Hola"
    llm.text_part.return_value = {"type": "text", "text": "prompt"}
    llm.audio_part.return_value = {"type": "input_audio", "input_audio": {"data": "x", "format": "wav"}}

    monkeypatch.setattr(
        "localllm.pipelines.translate.to_wav_16k",
        lambda path: path,
    )
    monkeypatch.setattr(
        "localllm.pipelines.stt_batch.transcribe_file",
        lambda *args, **kwargs: "Hello",
    )

    settings = AppSettings()
    settings.translate.pipeline = "split"

    result = translate_audio(
        audio,
        target_lang="es",
        llm_client=llm,
        settings=settings,
    )
    assert result.transcript == "Hello"
    assert result.translation == "Hola"


def test_translate_audio_unified(monkeypatch, tmp_path: Path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"wav")

    llm = MagicMock()
    llm.chat.return_value = "TRANSCRIPT:\nHello\nTRANSLATION:\nHola"
    llm.text_part.return_value = {"type": "text", "text": "prompt"}
    llm.audio_part.return_value = {"type": "input_audio", "input_audio": {"data": "x", "format": "wav"}}

    monkeypatch.setattr(
        "localllm.pipelines.translate.to_wav_16k",
        lambda path: path,
    )

    settings = AppSettings()
    settings.translate.pipeline = "unified"

    result = translate_audio(
        audio,
        target_lang="es",
        llm_client=llm,
        settings=settings,
    )
    assert result.transcript == "Hello"
    assert result.translation == "Hola"
    assert result.llm_elapsed_sec >= 0.0


def test_retranslate_transcript_skips_audio():
    llm = MagicMock()
    llm.chat.return_value = "Bonjour"

    result = retranslate_transcript(
        "Hello there",
        source_lang="en",
        target_lang="fr",
        tone="friendly",
        llm_client=llm,
    )
    assert result.transcript == "Hello there"
    assert result.translation == "Bonjour"
    assert result.tone == "friendly"
    assert result.target_language == "fr"
    llm.chat.assert_called_once()


def test_translate_text_returns_elapsed():
    llm = MagicMock()
    llm.chat.return_value = "Hola"

    text, elapsed = translate_text("Hello", target_lang="es", llm_client=llm)
    assert text == "Hola"
    assert elapsed >= 0.0