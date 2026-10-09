#!/usr/bin/env python3
"""localllm-live-bench - replay a WAV through the live translate pipeline.

Feeds audio in (simulated) real time through the streaming endpointer and the
pipelined STT -> MT -> TTS session, then reports per-stage latency and the
steady-state lag behind the speaker. This is the regression harness for the
plan.md Workstream B target: lag <= 8 s.

  localllm-live-bench speech.wav --target en           # real models (gateway must run)
  localllm-live-bench speech.wav --fake --speed 8      # fake stages, 8x pace
  localllm-live-bench speech.wav --target en --engine gemini --out gemini.json

Engine-neutral metrics (time to first translated audio, tail after the speaker
stops, total translated audio) make the local pipeline and Gemini Live
Translate directly comparable on the same recording; ``--out`` saves the
per-row transcripts/translations for a side-by-side quality read.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from localllm.config import get_settings
from localllm.live import LiveTranslateSession, StreamingEndpointer
from localllm.live.gemini import GeminiLiveTranslateSession
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


async def run_bench(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    sr = 16000
    audio = load_mono_16k(Path(args.audio), sample_rate=sr)
    total_sec = len(audio) / sr
    engine = args.engine
    print(
        f"Audio: {args.audio} | {total_sec:.1f}s | engine: {engine} | "
        f"feeding at {args.speed:g}x real time"
    )

    rows: dict[int, dict[str, Any]] = {}
    timeline: list[tuple[float, str]] = []
    summary: dict[str, Any] = {}
    started = time.monotonic()

    async def on_event(event: dict[str, Any]) -> None:
        kind = event.get("type")
        idx = event.get("index")
        timeline.append((time.monotonic() - started, str(kind)))
        if idx is not None:
            row = rows.setdefault(idx, {})
            if kind == "audio":
                row.setdefault("audio_events", []).append(event)
                row["audio_sec"] = row.get("audio_sec", 0.0) + event.get("duration_sec", 0.0)
            row[kind] = {k: v for k, v in event.items() if k != "wav_base64"}
        # Gemini streams cumulative text fragments; only print stage completions.
        if kind == "segment" and "end_sec" in event:
            print(f"  seg #{idx}: {event['start_sec']}-{event['end_sec']}s")
        elif kind == "transcript" and engine == "local":
            print(f"  stt #{idx} ({event.get('elapsed_sec')}s): {event['text'][:70]}")
        elif kind == "translation" and engine == "local":
            print(f"  mt  #{idx} ({event.get('elapsed_sec')}s): {event['text'][:70]}")
        elif kind == "audio" and engine == "local":
            print(
                f"  tts #{idx} ({event.get('elapsed_sec')}s) | "
                f"lag behind speaker: {event['lag_sec']}s"
            )
        elif kind == "error":
            print(f"  ERROR #{idx} [{event.get('stage', engine)}]: {event['message']}")
        elif kind == "done":
            summary.update(event)

    if engine == "gemini":
        session: Any = GeminiLiveTranslateSession(
            target_lang=args.target, on_event=on_event, settings=settings
        )
        endpointer = None
    else:
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
    timeline.clear()
    for offset in range(0, len(audio), chunk_len):
        chunk = audio[offset : offset + chunk_len]
        # Pace the feed like a live microphone.
        target_time = started + (offset / sr) / args.speed
        delay = target_time - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        if endpointer is None:
            pcm = (chunk.clip(-1.0, 1.0) * 32767).astype("<i2").tobytes()
            await session.submit_pcm(pcm)
            continue
        for segment in endpointer.feed(chunk):
            await session.submit(segment.audio, sample_rate=sr, start_sample=segment.start_sample)

    input_end = time.monotonic() - started
    if endpointer is not None:
        tail = endpointer.flush()
        if tail is not None:
            await session.submit(tail.audio, sample_rate=sr, start_sample=tail.start_sample)
    await session.finish()

    audio_times = [t for t, kind in timeline if kind == "audio"]
    return {
        "engine": engine,
        "audio_file": str(args.audio),
        "input_sec": round(total_sec, 2),
        "speed": args.speed,
        "time_to_first_audio_sec": round(audio_times[0], 2) if audio_times else None,
        "tail_after_input_end_sec": (
            round(audio_times[-1] - input_end, 2) if audio_times else None
        ),
        "translated_audio_sec": round(sum(r.get("audio_sec", 0.0) for r in rows.values()), 2),
        "summary": summary,
        "rows": {
            idx: {
                "transcript": row.get("transcript", {}).get("text", ""),
                "translation": row.get("translation", {}).get("text", ""),
                "lag_sec": row.get("audio", {}).get("lag_sec"),
                "segment": row.get("segment", {}),
            }
            for idx, row in sorted(rows.items())
        },
        "_raw_rows": rows,
    }


def _report(result: dict[str, Any]) -> None:
    rows: dict[int, dict[str, Any]] = result["_raw_rows"]
    _report_common(result)
    if result["engine"] == "gemini":
        for idx, row in result["rows"].items():
            print(f"\n#{idx} {row['transcript'][:200]}\n  → {row['translation'][:200]}")
        return
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


def _report_common(result: dict[str, Any]) -> None:
    print(f"\n== {result['engine']} ==")
    print(f"Time to first translated audio: {result['time_to_first_audio_sec']}s")
    print(f"Tail after speaker stopped:     {result['tail_after_input_end_sec']}s")
    print(
        f"Translated audio produced:      {result['translated_audio_sec']}s "
        f"(input {result['input_sec']}s)"
    )
    cost = result["summary"].get("est_cost_usd")
    if cost is not None:
        print(f"Estimated cost (paid tier):     ${cost}")


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
    parser.add_argument(
        "--engine",
        choices=["local", "gemini"],
        default="local",
        help="local = STT->MT->TTS via the gateway; gemini = Gemini 3.5 Live Translate",
    )
    parser.add_argument("--out", help="Write metrics + per-row texts to this JSON file")
    args = parser.parse_args()
    if args.engine == "gemini" and args.fake:
        parser.error("--fake only applies to the local engine")

    result = asyncio.run(run_bench(args))
    _report(result)
    if args.out:
        payload = {k: v for k, v in result.items() if k != "_raw_rows"}
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
