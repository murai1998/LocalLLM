import asyncio
import base64
import io
import time
import wave

import numpy as np
import pytest

from localllm.live import LiveTranslateSession

SR = 16000


def _tiny_wav(seconds: float = 0.1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(b"\x00\x00" * int(SR * seconds))
    return buf.getvalue()


def _session(events, **overrides):
    async def on_event(event):
        events.append(event)

    defaults = dict(
        target_lang="en",
        on_event=on_event,
        stt_fn=lambda job: f"hola {job.index}",
        mt_fn=lambda text, ctx: text.replace("hola", "hello"),
        tts_fn=lambda text: _tiny_wav(),
    )
    defaults.update(overrides)
    return LiveTranslateSession(**defaults)


async def _run(session, n_segments=2, seg_seconds=2.0):
    await session.start()
    audio = np.zeros(int(SR * seg_seconds), dtype=np.float32)
    for i in range(n_segments):
        await session.submit(audio, sample_rate=SR, start_sample=i * len(audio))
    await session.finish()


def test_pipeline_emits_full_event_sequence_per_segment():
    events: list[dict] = []
    asyncio.run(_run(_session(events), n_segments=2))

    by_type = {}
    for e in events:
        by_type.setdefault(e["type"], []).append(e)
    assert len(by_type["segment"]) == 2
    assert len(by_type["transcript"]) == 2
    assert len(by_type["translation"]) == 2
    assert len(by_type["audio"]) == 2
    assert by_type["transcript"][0]["text"] == "hola 0"
    assert by_type["translation"][0]["text"] == "hello 0"
    assert events[-1] == {"type": "done", "segments": 2}

    audio_event = by_type["audio"][0]
    assert audio_event["lag_sec"] >= 0
    assert audio_event["duration_sec"] > 0
    base64.b64decode(audio_event["wav_base64"])  # valid base64 payload


def test_rolling_context_passed_to_mt():
    events: list[dict] = []
    seen_context: list[list] = []

    def mt(text, ctx):
        seen_context.append(list(ctx))
        return text.upper()

    asyncio.run(_run(_session(events, mt_fn=mt), n_segments=3))
    assert seen_context[0] == []
    assert seen_context[1] == [("hola 0", "HOLA 0")]
    assert len(seen_context[2]) == 2  # capped deque


def test_empty_transcript_skips_mt_and_tts():
    events: list[dict] = []
    asyncio.run(_run(_session(events, stt_fn=lambda job: ""), n_segments=1))
    types = [e["type"] for e in events]
    assert "transcript" in types
    assert "translation" not in types
    assert "audio" not in types
    assert types[-1] == "done"


def test_stage_error_is_reported_and_pipeline_continues():
    events: list[dict] = []

    def flaky_stt(job):
        if job.index == 0:
            raise RuntimeError("gateway hiccup")
        return "ok"

    asyncio.run(_run(_session(events, stt_fn=flaky_stt), n_segments=2))
    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["stage"] == "stt"
    translations = [e for e in events if e["type"] == "translation"]
    assert len(translations) == 1  # segment 1 still made it through


def test_tts_none_means_no_audio_event():
    events: list[dict] = []
    asyncio.run(_run(_session(events, tts_fn=lambda text: None), n_segments=1))
    assert not [e for e in events if e["type"] == "audio"]
    assert [e for e in events if e["type"] == "translation"]


def test_stages_overlap_across_segments():
    """STT of segment N+1 must run while MT/TTS of segment N is in flight."""
    events: list[dict] = []
    windows: dict[str, list[tuple[float, float]]] = {"stt": [], "mt": []}

    def stt(job):
        start = time.perf_counter()
        time.sleep(0.15)
        windows["stt"].append((start, time.perf_counter()))
        return f"t{job.index}"

    def mt(text, ctx):
        start = time.perf_counter()
        time.sleep(0.15)
        windows["mt"].append((start, time.perf_counter()))
        return text

    asyncio.run(_run(_session(events, stt_fn=stt, mt_fn=mt), n_segments=2))
    # MT of segment 0 should overlap STT of segment 1.
    mt0_start, mt0_end = windows["mt"][0]
    stt1_start, stt1_end = windows["stt"][1]
    assert stt1_start < mt0_end, "pipeline stages did not overlap"


@pytest.mark.parametrize("n", [1, 4])
def test_done_event_reports_segment_count(n):
    events: list[dict] = []
    asyncio.run(_run(_session(events), n_segments=n))
    assert events[-1] == {"type": "done", "segments": n}
