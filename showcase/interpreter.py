"""Live interpreter tab logic — FastRTC ReplyOnPause handler.

The handler is built by `build_reply_handler(translate_fn, tts_fn)` so tests and
local rehearsal can inject fakes; `app.py` wires in the real GPU functions.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator

import numpy as np
from piper_voices import tts_supported

MIN_UTTERANCE_SECONDS = 0.4
MAX_CONTEXT_TURNS = 2

TranslateFn = Callable[..., tuple[str, str]]
TtsFn = Callable[..., tuple[int, np.ndarray]]

# Split on sentence-final punctuation (Latin + CJK) so we can speak each
# sentence the instant it is ready instead of waiting for the whole utterance.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…。！？])\s+")


def split_sentences(text: str) -> list[str]:
    """Break a translation into speakable sentence chunks (order preserved)."""
    return [chunk for chunk in (s.strip() for s in _SENTENCE_SPLIT.split(text)) if chunk]


def is_degenerate(text: str, *, max_run: int = 6) -> bool:
    """Detect Whisper/LLM repetition loops — the mangled live "last chunk".

    Near-silent trailing audio makes Whisper emit one word dozens of times; that
    then drives the translator into the same loop. Flag a long run of the same
    word, or a very low unique-word ratio over a long output, so the caller can
    drop the utterance instead of showing (and speaking) the garbage.
    """
    words = text.split()
    if len(words) < max_run:
        return False
    run = best = 1
    for prev, cur in zip(words, words[1:]):
        run = run + 1 if cur == prev else 1
        best = max(best, run)
    if best >= max_run:
        return True
    return len(words) >= 20 and len(set(words)) / len(words) < 0.2


def to_mono_float32(pcm: np.ndarray) -> np.ndarray:
    """FastRTC delivers int16, possibly (channels, n) — normalize to mono float32.

    Scale by dtype BEFORE averaging channels: `.mean()` silently promotes int16
    to float64, which used to skip the /32768 scaling and ship ±32767 amplitudes
    downstream (Whisper's log-mel shrugged that off; a WAV encoder clipping to
    [-1, 1] turns it into a square-wave buzz — "the audio is silent").
    """
    audio = np.asarray(pcm)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)
    if audio.ndim == 2:
        axis = 0 if audio.shape[0] < audio.shape[1] else 1
        audio = audio.mean(axis=axis)
    return audio


def format_panes(rows: list[tuple[str, str]]) -> tuple[str, str]:
    """Two side-by-side streams: all transcripts, all translations."""
    sources = "\n\n".join(src for src, _tgt in rows)
    translations = "\n\n".join(tgt for _src, tgt in rows)
    return sources, translations


def build_reply_handler(translate_fn: TranslateFn, tts_fn: TtsFn):
    """Returns a ReplyOnPause generator handler with PRIVATE rolling history.

    History lives in this closure (one handler instance per WebRTC connection —
    see StatefulReplyOnPause in app.py) instead of round-tripping through a
    gr.State: fastrtc's additional-outputs event is an endless generator, and
    gr.State outputs from a still-running generator never commit reliably, so a
    State-based history reads as [] on every utterance and the transcript
    "loses" all previous rows.

    Signature matches the WebRTC stream inputs wired in app.py:
    (audio, source_lang, target_lang, tone, voice_id).
    """
    from fastrtc import AdditionalOutputs

    memory: list[tuple[str, str]] = []

    def reply(audio: tuple[int, np.ndarray], *rest) -> Iterator:
        # fastrtc passes (audio, *stream_inputs); depending on version the
        # WebRTC component's own value is or isn't included in stream_inputs,
        # so take our four known inputs from the tail.
        if len(rest) < 4:
            raise ValueError(f"expected at least 4 stream inputs, got {len(rest)}")
        source_lang, target_lang, tone, voice_id = rest[-4:]
        sample_rate, pcm = audio
        samples = to_mono_float32(pcm)
        if len(samples) < sample_rate * MIN_UTTERANCE_SECONDS:
            return

        context = memory[-MAX_CONTEXT_TURNS:]
        transcript, translation = translate_fn(
            samples,
            sample_rate,
            source_lang or None,
            target_lang,
            tone,
            context,
        )
        # Drop hallucinated repetition loops (the mangled trailing chunk) so they
        # never reach the transcript or the speakers.
        if not transcript or is_degenerate(transcript) or is_degenerate(translation):
            return

        memory.append((transcript, translation))
        yield AdditionalOutputs(*format_panes(memory))

        # Speak sentence-by-sentence so the headphone starts playing the moment
        # the first sentence is synthesized, rather than after the whole reply —
        # a simultaneous-interpreter feel instead of a one-shot dump.
        if translation and tts_supported(target_lang):
            for sentence in split_sentences(translation):
                tts_rate, voice = tts_fn(
                    sentence, language=target_lang, voice_id=voice_id or None
                )
                if voice.size:
                    yield (tts_rate, voice.reshape(1, -1))

    return reply
