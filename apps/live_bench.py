#!/usr/bin/env python3
"""localllm-live-bench - replay a WAV through the live translate pipeline.

Feeds audio in (simulated) real time through the streaming endpointer and the
pipelined STT -> MT -> TTS session, then reports per-stage latency and the
steady-state lag behind the speaker. This is the regression harness for the
plan.md Workstream B target: lag <= 8 s.

  localllm-live-bench speech.wav --target en           # real models (gateway must run)
  localllm-live-bench speech.wav --fake --speed 8      # fake stages, 8x pace
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import Any

from localllm.config import get_settings
from localllm.live import LiveTranslateSession, StreamingEndpointer
from localllm.media.audio import load_mono_16k

CHUNK_MS = 250


def _fake_stages(speed: float):
    """Deterministic stand-ins scaled from the measured budget (plan.md)."""

    def stt(job) -> str:
        time.sleep((len(job.audio) / job.sample_rate) * (5 / 30) / speed)
        return f"segment {job.index} transcript"

    def mt(transcript: str, context) -> str:
        time.sleep(0.4 / speed)
        return transcript.replace("transcript", "translation")

    def tts(translation: str) -> bytes:
        import io
        import wave

        time.sleep(0.6 / speed)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 1600)
        return buf.getvalue()

    return stt, mt, tts


async def run_bench(args: argparse.Namespace) -> dict[int, dict[str, Any]]:
    settings = get_settings()
    sr = 16000
    audio = load_mono_16k(Path(args.audio), sample_rate=sr)
    total_sec = len(audio) / sr
    print(f"Audio: {args.audio} | {total_sec:.1f}s | feeding at {args.speed:g}x real time")

    rows: dict[int, dict[str, Any]] = {}

    async def on_event(event: dict[str, Any]) -> None:
        kind = event.get("type")
        idx = event.get("index")
        if idx is not None:
            rows.setdefault(idx, {})[kind] = event
        if kind == "segment":
            print(f"  seg #{idx}: {event['start_sec']}-{event['end_sec']}s")
        elif kind == "transcript":
            print(f"  stt #{idx} ({event['elapsed_sec']}s): {event['text'][:70]}")
        elif kind == "translation":
            print(f"  mt  #{idx} ({event['elapsed_sec']}s): {event['text'][:70]}")
        elif kind == "audio":
            print(
                f"  tts #{idx} ({event['elapsed_sec']}s) | "
                f"lag behind speaker: {event['lag_sec']}s"
            )
        elif kind == "error":
            print(f"  ERROR #{idx} [{event.get('stage')}]: {event['message']}")

    stage_overrides = {}
    if args.fake:
        stt, mt, tts = _fake_stages(args.speed)
        stage_overrides = {"stt_fn": stt, "mt_fn": mt, "tts_fn": tts}

    session = LiveTranslateSession(
        source_lang=args.source or None,
        target_lang=args.target,
        on_event=on_event,
        settings=settings,
        **stage_overrides,
    )
    endpointer = StreamingEndpointer(sr, settings.translate.stream)
    await session.start()

    chunk_len = int(sr * CHUNK_MS / 1000)
    started = time.monotonic()
    for offset in range(0, len(audio), chunk_len):
        chunk = audio[offset : offset + chunk_len]
        # Pace the feed like a live microphone.
        target_time = started + (offset / sr) / args.speed
        delay = target_time - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        for segment in endpointer.feed(chunk):
            await session.submit(segment.audio, sample_rate=sr, start_sample=segment.start_sample)

    tail = endpointer.flush()
    if tail is not None:
        await session.submit(tail.audio, sample_rate=sr, start_sample=tail.start_sample)
    await session.finish()
    return rows


def _report(rows: dict[int, dict[str, Any]]) -> None:
    if not rows:
        print("\nNo speech segments detected.")
        return
    print(f"\n{'seg':>4} {'dur':>6} {'stt':>6} {'mt':>6} {'tts':>6} {'lag':>6}")
    lags: list[float] = []
    for idx in sorted(rows):
        row = rows[idx]
        dur = row.get("segment", {}).get("duration_sec", 0.0)
        stt = row.get("transcript", {}).get("elapsed_sec", float("nan"))
        mt = row.get("translation", {}).get("elapsed_sec", float("nan"))
        tts = row.get("audio", {}).get("elapsed_sec", float("nan"))
        lag = row.get("audio", {}).get("lag_sec")
        if lag is not None:
            lags.append(lag)
        lag_s = f"{lag:.1f}" if lag is not None else "-"
        print(f"{idx:>4} {dur:>6.1f} {stt:>6} {mt:>6} {tts:>6} {lag_s:>6}")
    if lags:
        steady = lags[len(lags) // 2 :] or lags
        print(
            f"\nLag behind speaker - first: {lags[0]:.1f}s | "
            f"max: {max(lags):.1f}s | steady-state avg: {sum(steady) / len(steady):.1f}s "
            f"(target <= 8s)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Live translate pipeline benchmark")
    parser.add_argument("audio", help="WAV/MP3/M4A file to replay")
    parser.add_argument("--source", default="", help="Source language code (default: auto)")
    parser.add_argument("--target", default="en", help="Target language code")
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Feed pace multiplier (1 = real time)"
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Use deterministic fake STT/MT/TTS stages (no gateway/GPU needed)",
    )
    args = parser.parse_args()

    rows = asyncio.run(run_bench(args))
    _report(rows)


if __name__ == "__main__":
    main()
