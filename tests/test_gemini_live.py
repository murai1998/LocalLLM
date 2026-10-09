import asyncio
import base64
import contextlib
import io
import wave
from types import SimpleNamespace

import pytest

pytest.importorskip("google.genai", reason="gemini extra not installed")

from localllm.config import AppSettings  # noqa: E402
from localllm.live import gemini  # noqa: E402
from localllm.live.gemini import GeminiLiveTranslateSession, GeminiUnavailable  # noqa: E402


def _msg(*, text_in=None, text_out=None, audio=None, turn_complete=False):
    parts = [SimpleNamespace(inline_data=SimpleNamespace(data=audio))] if audio else []
    return SimpleNamespace(
        go_away=None,
        server_content=SimpleNamespace(
            input_transcription=SimpleNamespace(text=text_in) if text_in else None,
            output_transcription=SimpleNamespace(text=text_out) if text_out else None,
            model_turn=SimpleNamespace(parts=parts) if parts else None,
            turn_complete=turn_complete,
            generation_complete=False,
        ),
    )


class FakeLiveSession:
    """Mimics google-genai's AsyncSession: receive() ends after each turn."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.sent = []
        self.stream_ended = asyncio.Event()

    async def send_realtime_input(self, *, audio=None, audio_stream_end=None):
        if audio_stream_end:
            self.stream_ended.set()
        else:
            self.sent.append(audio)

    async def receive(self):
        if not self.turns:
            return
        await self.stream_ended.wait()
        for message in self.turns.pop(0):
            yield message


def _connect(fake):
    @contextlib.asynccontextmanager
    async def connect(model, config):
        yield fake

    return connect


def _run(fake, pcm: bytes):
    events = []

    async def on_event(event):
        events.append(event)

    async def main():
        session = GeminiLiveTranslateSession(
            target_lang="es",
            on_event=on_event,
            settings=AppSettings(),
            connect=_connect(fake),
            config=object(),
        )
        await session.start()
        await session.submit_pcm(pcm)
        await session.finish()

    asyncio.run(main())
    return events


def test_events_map_to_local_pipeline_schema(monkeypatch):
    monkeypatch.setattr(gemini, "DRAIN_IDLE_SECONDS", 0.2)
    pcm_out = b"\x01\x00" * 24000  # 1 s of 24 kHz audio
    fake = FakeLiveSession(
        [
            [
                _msg(text_in="Hello "),
                _msg(text_in="world."),
                _msg(text_out="Hola "),
                _msg(audio=pcm_out),
                _msg(text_out="mundo.", turn_complete=True),
            ],
            [_msg(text_in="Bye."), _msg(text_out="Adiós.", turn_complete=True)],
        ]
    )
    events = _run(fake, b"\x00\x00" * 8000)  # 0.5 s of input

    # 0.5 s input → five 100 ms upstream chunks.
    assert len(fake.sent) == 5
    assert all(blob.mime_type == "audio/pcm;rate=16000" for blob in fake.sent)

    transcripts = [e for e in events if e["type"] == "transcript"]
    assert transcripts[-2]["text"] == "Hello world."  # cumulative per row
    translations = {e["index"]: e["text"] for e in events if e["type"] == "translation"}
    assert translations == {0: "Hola mundo.", 1: "Adiós."}

    audio = [e for e in events if e["type"] == "audio"]
    assert len(audio) == 1 and audio[0]["index"] == 0 and audio[0]["duration_sec"] == 1.0
    with wave.open(io.BytesIO(base64.b64decode(audio[0]["wav_base64"]))) as wav:
        assert wav.getframerate() == 24000

    closed = [e for e in events if e["type"] == "segment" and "end_sec" in e]
    assert [e["index"] for e in closed] == [0, 1]

    done = events[-1]
    assert done["type"] == "done" and done["engine"] == "gemini"
    assert done["segments"] == 2
    assert done["output_audio_sec"] == pytest.approx(1.0)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.setattr(gemini, "read_gemini_api_key", lambda: None)

    async def on_event(event):
        pass

    with pytest.raises(GeminiUnavailable, match="GEMINI_API_KEY"):
        GeminiLiveTranslateSession(target_lang="es", on_event=on_event)


def test_language_codes_follow_gemini_bcp47():
    assert gemini.gemini_language_code("zh") == "zh-Hans"
    assert gemini.gemini_language_code("de") == "de"


def test_build_live_config_sets_translation_target():
    config = gemini.build_live_config("zh", echo_target_language=True)
    assert config.translation_config.target_language_code == "zh-Hans"
    assert config.translation_config.echo_target_language is True
    assert config.input_audio_transcription is not None


def test_normal_close_is_not_reported_as_error(monkeypatch):
    monkeypatch.setattr(gemini, "DRAIN_IDLE_SECONDS", 0.2)

    class ClosingSession(FakeLiveSession):
        async def receive(self):
            await self.stream_ended.wait()
            exc = RuntimeError("1000 None.")
            exc.code = 1000
            raise exc
            yield  # pragma: no cover

    events = _run(ClosingSession([]), b"\x00\x00" * 1600)
    assert not [e for e in events if e["type"] == "error"]
    assert events[-1]["type"] == "done"
