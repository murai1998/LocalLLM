import io
import wave
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import localllm.webui.server as server

SR = 16000


@pytest.fixture
def client():
    server._llm_client = MagicMock()
    server._llm_client.is_ready.return_value = True
    yield TestClient(server.create_app())
    server._llm_client = None


def _tiny_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(b"\x00\x00" * 800)
    return buf.getvalue()


def _pcm_frames(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


def _tone(seconds: float) -> np.ndarray:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_ws_translate_full_session(client, monkeypatch):
    # Replace the real model stages with fakes via the session class.
    original = server.LiveTranslateSession

    def patched_session(**kwargs):
        kwargs["stt_fn"] = lambda job: "hola mundo"
        kwargs["mt_fn"] = lambda text, ctx: "hello world"
        kwargs["tts_fn"] = lambda text: _tiny_wav()
        return original(**kwargs)

    monkeypatch.setattr(server, "LiveTranslateSession", patched_session)

    audio = np.concatenate(
        [_tone(2.0), np.zeros(int(SR * 1.2), dtype=np.float32)]
    )

    with client.websocket_connect("/ws/translate") as ws:
        ws.send_json({"type": "start", "target_lang": "en", "sample_rate": SR})
        pcm = _pcm_frames(audio)
        step = SR // 2 * 2  # 0.5 s of int16 bytes
        for off in range(0, len(pcm), step):
            ws.send_bytes(pcm[off : off + step])
        ws.send_json({"type": "stop"})

        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] == "done":
                break

    types = [e["type"] for e in events]
    assert "segment" in types
    assert "transcript" in types
    assert "translation" in types
    assert "audio" in types
    transcript = next(e for e in events if e["type"] == "transcript")
    assert transcript["text"] == "hola mundo"
    translation = next(e for e in events if e["type"] == "translation")
    assert translation["text"] == "hello world"
    audio_event = next(e for e in events if e["type"] == "audio")
    assert audio_event["lag_sec"] >= 0


def test_ws_translate_rejects_bad_first_message(client):
    with client.websocket_connect("/ws/translate") as ws:
        ws.send_json({"type": "frames"})
        event = ws.receive_json()
        assert event["type"] == "error"


def test_ws_translate_silence_only_yields_done_with_zero_segments(client, monkeypatch):
    original = server.LiveTranslateSession

    def patched_session(**kwargs):
        kwargs["stt_fn"] = lambda job: "should never run"
        return original(**kwargs)

    monkeypatch.setattr(server, "LiveTranslateSession", patched_session)

    with client.websocket_connect("/ws/translate") as ws:
        ws.send_json({"type": "start", "target_lang": "en", "sample_rate": SR})
        ws.send_bytes(_pcm_frames(np.zeros(SR * 3, dtype=np.float32)))
        ws.send_json({"type": "stop"})
        event = ws.receive_json()
        assert event == {"type": "done", "segments": 0}
