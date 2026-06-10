import numpy as np
import pytest

from localllm.config import GenerationConfig, ServiceConfig, SttConfig
from localllm.media.audio import chunk_audio, merge_transcripts
from localllm.tts.piper import (
    VOICE_OPTIONS,
    resolve_piper_voice_name,
    tts_supported,
    voice_options_for_language,
)


def test_enable_thinking_defaults_off():
    # Project decision: thinking disabled by default — enabling it without a
    # large token budget yields empty replies (AGENT_HANDOFF §2).
    assert GenerationConfig().enable_thinking is False


def test_service_guardrail_defaults():
    cfg = ServiceConfig()
    assert cfg.api_key is None  # auth opt-in, localhost-only by default
    assert cfg.queue_timeout_sec > 0
    assert cfg.max_request_bytes > 1024 * 1024


def test_no_cross_language_voice_fallbacks():
    # ja/ko previously fell back to zh/en voices and produced gibberish.
    assert "ja" not in VOICE_OPTIONS
    assert "ko" not in VOICE_OPTIONS
    assert voice_options_for_language("ja") == []
    assert not tts_supported("ko")
    with pytest.raises(ValueError):
        resolve_piper_voice_name(language="ja")


def test_unknown_language_has_no_english_fallback():
    assert voice_options_for_language("xx") == []


def test_chunk_audio_keeps_short_tail():
    sr = 16000
    # 52.3 s: windows start at 0 / 26 / 52 s, leaving a 0.3 s tail that the old
    # code silently dropped.
    audio = np.arange(int(sr * 52.3), dtype=np.float32)
    cfg = SttConfig(chunk_seconds=28, overlap_seconds=2, max_chunk_seconds=30)
    chunks = chunk_audio(audio, sr, cfg)
    assert chunks[-1][-1] == audio[-1]
    assert all(len(c) <= sr * 30 for c in chunks)


def test_merge_transcripts_dedupes_repeated_boundary_text():
    # Documented behavior: a >8-char overlap between chunk transcripts is
    # treated as the STT overlap window and deduplicated — even when the
    # speaker genuinely repeated the phrase.
    merged = merge_transcripts(["we must go we must go", "we must go now"])
    assert merged == "we must go we must go now"
