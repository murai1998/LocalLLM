#!/usr/bin/env python3
"""Translate tab UI — Gemma audio + Phase 2 VAD chunking + local Piper TTS."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import streamlit as st

from localllm.client.factory import create_llm_client
from localllm.config import get_settings
from localllm.pipelines.translate import (
    LANGUAGE_LABELS,
    TONE_PRESETS,
    ToneId,
    TranslationResult,
    translate_audio,
)
from localllm.pipelines.translate_chunked import (
    ChunkedTranslationResult,
    ChunkTranslation,
    display_translation_text,
    synthesize_new_sentences,
    translate_audio_chunked,
)
from localllm.tts import PIPER_AVAILABLE, synthesize_speech, voice_options_for_language, warmup_tts
from localllm.tts.sentence_queue import SentenceQueue

SUPPORTED_EXTENSIONS = {"wav", "mp3", "m4a", "ogg", "flac", "webm"}
LANGUAGE_OPTIONS = [("auto", "Auto-detect")] + [
    (code, label) for code, label in sorted(LANGUAGE_LABELS.items(), key=lambda item: item[1])
]


def init_translate_state() -> None:
    defaults = {
        "transcript": "",
        "translation": "",
        "tts_audio": None,
        "translate_result": None,
        "translate_live_result": None,
        "recorded_audio": None,
        "recorded_audio_name": "recording.wav",
        "tts_voice_id": None,
        "tts_spoken_sentence_count": 0,
        "live_auto_tts": True,
        "live_last_sentences": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def get_translate_llm_client():
    return create_llm_client()


@st.cache_resource
def get_tts_warmup():
    return warmup_tts()


def render_translate_sidebar() -> tuple[str | None, str, ToneId, str | None]:
    settings = get_settings()
    default_target = settings.translate.target_language

    st.header("Languages")
    source_lang = _lang_selectbox("Source", "translate_source_lang", include_auto=True)
    if "translate_target_lang" not in st.session_state:
        st.session_state.translate_target_lang = default_target
    target_lang = _lang_selectbox("Target", "translate_target_lang") or default_target

    st.header("Translation style")
    tone_ids = list(TONE_PRESETS.keys())
    tone: ToneId = st.radio(
        "Tone",
        tone_ids,
        format_func=lambda key: TONE_PRESETS[key]["label"],
        captions=[TONE_PRESETS[key]["hint"] for key in tone_ids],
        key="translate_tone",
    )

    voice_options = voice_options_for_language(target_lang)
    voice_ids = [option["id"] for option in voice_options]
    if st.session_state.get("tts_voice_id") not in voice_ids:
        st.session_state.tts_voice_id = voice_ids[0]
    voice_id = st.radio(
        "Voice",
        voice_ids,
        format_func=lambda vid: next(o["label"] for o in voice_options if o["id"] == vid),
        key="translate_voice_radio",
    )
    st.session_state.tts_voice_id = voice_id

    llm_ok = get_translate_llm_client().is_ready()
    tts_ready = get_tts_warmup()
    st.markdown("**Service**")
    st.write(f"LocalLLM (:8090): {'✅ ready' if llm_ok else '❌ offline'}")
    st.write(f"TTS (Piper, offline): {'✅ warmed' if tts_ready else '❌ unavailable'}")
    st.caption("No internet at runtime; voice models download once to `models/piper/`.")
    live = settings.translate.live
    st.caption(
        f"Phase 2 live: **{live.min_chunk_seconds:.0f}–{live.max_chunk_seconds:.0f}s** chunks, "
        f"**{live.overlap_seconds:.1f}s** overlap"
    )
    if not PIPER_AVAILABLE:
        st.caption("Install: `pip install piper-tts`")
    if not llm_ok:
        st.info("Start: `localllm-serve`")

    return source_lang, target_lang, tone, voice_id


def _lang_selectbox(label: str, key: str, *, include_auto: bool = False) -> str | None:
    options = LANGUAGE_OPTIONS if include_auto else LANGUAGE_OPTIONS[1:]
    labels = [name for _, name in options]
    codes = [code for code, _ in options]
    index = 0
    if key in st.session_state and st.session_state[key] in codes:
        index = codes.index(st.session_state[key])
    choice = st.selectbox(label, labels, index=index, key=f"{key}_label")
    code = codes[labels.index(choice)]
    st.session_state[key] = code
    return None if code == "auto" else code


def _save_audio_temp(data: bytes, filename: str) -> Path:
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return Path(tmp.name)


def _run_translation(
    data: bytes,
    filename: str,
    *,
    source_lang: str | None,
    target_lang: str,
    tone: ToneId,
) -> TranslationResult:
    tmp_path = _save_audio_temp(data, filename)
    try:
        return translate_audio(
            tmp_path,
            source_lang=source_lang,
            target_lang=target_lang,
            tone=tone,
            llm_client=get_translate_llm_client(),
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def _sentence_queue() -> SentenceQueue:
    queue = SentenceQueue()
    queue.spoken_sentence_count = int(st.session_state.get("tts_spoken_sentence_count", 0))
    queue.pending_audio = st.session_state.get("tts_audio")
    return queue


def _persist_sentence_queue(queue: SentenceQueue) -> None:
    st.session_state.tts_spoken_sentence_count = queue.spoken_sentence_count
    st.session_state.tts_audio = queue.pending_audio


def _reset_tts_queue() -> None:
    st.session_state.tts_spoken_sentence_count = 0
    st.session_state.tts_audio = None
    st.session_state.live_last_sentences = []


def _store_batch_result(result: TranslationResult) -> None:
    st.session_state.transcript = result.transcript
    st.session_state.translation = result.translation
    st.session_state.translate_target_lang = result.target_language
    st.session_state.translate_result = result
    st.session_state.translate_live_result = None
    _reset_tts_queue()


def _store_live_result(result: ChunkedTranslationResult) -> None:
    st.session_state.transcript = result.transcript
    st.session_state.translation = display_translation_text(result.translation)
    st.session_state.translate_target_lang = result.target_language
    st.session_state.translate_live_result = result
    st.session_state.translate_result = None


def _render_timing_batch(result: TranslationResult) -> None:
    st.markdown("### Timing")
    cols = st.columns(3)
    cols[0].metric("Gemma (audio)", f"{result.llm_elapsed_sec:.2f}s", help="Transcribe + translate")
    cols[1].metric("Total", f"{result.total_elapsed_sec:.2f}s")
    if result.tts_elapsed_sec > 0:
        cols[2].metric("TTS", f"{result.tts_elapsed_sec:.2f}s")
    tone_label = TONE_PRESETS[result.tone]["label"]
    st.caption(
        f"Source: **{result.detected_language}** · "
        f"Target: **{result.target_language}** · "
        f"Tone: **{tone_label}**"
    )


def _render_timing_live(result: ChunkedTranslationResult) -> None:
    st.markdown("### Timing (live chunks)")
    cols = st.columns(4)
    cols[0].metric("Chunks", str(result.chunk_count))
    cols[1].metric("Gemma total", f"{result.llm_elapsed_sec:.2f}s")
    cols[2].metric("Total", f"{result.total_elapsed_sec:.2f}s")
    if result.tts_elapsed_sec > 0:
        cols[3].metric("TTS queue", f"{result.tts_elapsed_sec:.2f}s")
    if result.chunks:
        avg = sum(c.elapsed_sec for c in result.chunks) / len(result.chunks)
        st.caption(f"Avg per chunk: **{avg:.2f}s** · Tone: **{TONE_PRESETS[result.tone]['label']}**")


def _play_translation(voice_id: str | None) -> None:
    translation = st.session_state.get("translation", "")
    target_lang = st.session_state.get("translate_target_lang", "es")
    if not translation:
        st.warning("Translate audio first.")
        return
    if not PIPER_AVAILABLE:
        st.error("Install Piper TTS: `pip install piper-tts`")
        return

    started = time.perf_counter()
    with st.spinner("Generating speech…"):
        audio_bytes = synthesize_speech(
            translation,
            language=target_lang,
            voice_id=voice_id,
        )
    elapsed = time.perf_counter() - started

    st.session_state.tts_audio = audio_bytes
    result: TranslationResult | None = st.session_state.get("translate_result")
    if result is not None:
        result.tts_elapsed_sec = elapsed
        st.session_state.translate_result = result
    live: ChunkedTranslationResult | None = st.session_state.get("translate_live_result")
    if live is not None:
        live.tts_elapsed_sec = elapsed
        st.session_state.translate_live_result = live
    st.audio(audio_bytes, format="audio/wav")


def _run_live_chunked(
    data: bytes,
    filename: str,
    *,
    source_lang: str | None,
    target_lang: str,
    tone: ToneId,
    voice_id: str | None,
    auto_tts: bool,
) -> None:
    tmp_path = _save_audio_temp(data, filename)
    progress = st.progress(0.0, text="Preparing VAD chunks…")
    status = st.empty()
    transcript_live = st.empty()
    translation_live = st.empty()

    partial_transcript: list[str] = []
    partial_translation: list[str] = []
    chunk_rows: list[ChunkTranslation] = []

    def on_progress(done: int, total: int, item: ChunkTranslation) -> None:
        chunk_rows.append(item)
        partial_transcript.append(item.transcript)
        partial_translation.append(item.translation)
        from localllm.media.audio import merge_transcripts

        transcript_live.markdown(
            "**Transcript (partial)**\n\n"
            + (merge_transcripts(partial_transcript) or "…")
        )
        translation_live.markdown(
            "**Translation (partial)**\n\n"
            + (display_translation_text(merge_transcripts(partial_translation)) or "…")
        )
        progress.progress(done / total, text=f"Chunk {done}/{total} · {item.elapsed_sec:.1f}s")

    try:
        result = translate_audio_chunked(
            tmp_path,
            source_lang=source_lang,
            target_lang=target_lang,
            tone=tone,
            llm_client=get_translate_llm_client(),
            on_progress=on_progress,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    _store_live_result(result)
    _reset_tts_queue()

    if auto_tts and result.translation.strip() and PIPER_AVAILABLE:
        queue = _sentence_queue()
        audio, spoken, tts_elapsed = synthesize_new_sentences(
            result.translation,
            target_lang=target_lang,
            voice_id=voice_id,
            queue=queue,
        )
        _persist_sentence_queue(queue)
        st.session_state.live_last_sentences = spoken
        result.tts_elapsed_sec = tts_elapsed
        st.session_state.translate_live_result = result

    progress.progress(1.0, text=f"Done — {result.chunk_count} chunk(s)")
    status.success(
        f"Live chunked translate complete: **{result.chunk_count}** chunks in "
        f"**{result.llm_elapsed_sec:.1f}s**"
    )


def run_translate_ui(
    *,
    source_lang: str | None,
    target_lang: str,
    tone: ToneId,
    voice_id: str | None,
) -> None:
    init_translate_state()
    st.subheader("Voice Translator")
    st.caption("Batch translate · Phase 2 live VAD chunks · Piper TTS")

    tab_upload, tab_record, tab_live = st.tabs([
        "Upload audio",
        "Record voice",
        "Live (chunked)",
    ])

    with tab_upload:
        uploaded = st.file_uploader(
            "Audio file",
            type=sorted(SUPPORTED_EXTENSIONS),
            key="translate_upload_file",
        )
        if st.button("Translate upload", type="primary", key="translate_upload_btn"):
            if uploaded is None:
                st.warning("Choose an audio file first.")
            else:
                with st.spinner("Translating with Gemma…"):
                    result = _run_translation(
                        uploaded.getvalue(),
                        uploaded.name,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        tone=tone,
                    )
                _store_batch_result(result)
                st.rerun()

    with tab_record:
        st.markdown("Record, then click translate.")
        audio_input = st.audio_input("Microphone", key="translate_mic")
        if audio_input is not None:
            st.session_state.recorded_audio = audio_input.getvalue()
            st.session_state.recorded_audio_name = "recording.wav"
        if st.session_state.recorded_audio:
            st.audio(st.session_state.recorded_audio, format="audio/wav")
        if st.button("Translate recording", type="primary", key="translate_record_btn"):
            recorded = st.session_state.get("recorded_audio")
            if not recorded:
                st.warning("Record audio first.")
            else:
                with st.spinner("Translating with Gemma…"):
                    result = _run_translation(
                        recorded,
                        st.session_state.recorded_audio_name,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        tone=tone,
                    )
                _store_batch_result(result)
                st.rerun()

    with tab_live:
        st.markdown(
            "Phase 2: VAD splits audio into **2–4 s** chunks with overlap. "
            "Transcript and translation update per chunk; TTS speaks **completed sentences** only."
        )
        st.session_state.live_auto_tts = st.checkbox(
            "Auto-speak completed sentences (TTS queue)",
            value=st.session_state.live_auto_tts,
            key="live_auto_tts_cb",
        )
        live_upload = st.file_uploader(
            "Audio for live chunking",
            type=sorted(SUPPORTED_EXTENSIONS),
            key="translate_live_upload",
        )
        live_mic = st.audio_input("Or record", key="translate_live_mic")
        if live_mic is not None:
            st.session_state.recorded_audio = live_mic.getvalue()
            st.session_state.recorded_audio_name = "recording.wav"

        use_recorded = st.checkbox(
            "Use microphone recording above",
            value=bool(st.session_state.recorded_audio),
            key="live_use_recording",
        )

        if st.button("Run live chunked translate", type="primary", key="translate_live_btn"):
            data: bytes | None = None
            name = "audio.wav"
            if use_recorded and st.session_state.recorded_audio:
                data = st.session_state.recorded_audio
                name = st.session_state.recorded_audio_name
            elif live_upload is not None:
                data = live_upload.getvalue()
                name = live_upload.name
            if not data:
                st.warning("Provide audio via upload or microphone.")
            else:
                _run_live_chunked(
                    data,
                    name,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    tone=tone,
                    voice_id=voice_id,
                    auto_tts=st.session_state.live_auto_tts,
                )
                st.rerun()

    if st.button("Play full translation", type="secondary", key="translate_play_btn"):
        _play_translation(voice_id)

    batch_result: TranslationResult | None = st.session_state.get("translate_result")
    live_result: ChunkedTranslationResult | None = st.session_state.get("translate_live_result")
    if batch_result is not None:
        _render_timing_batch(batch_result)
    elif live_result is not None:
        _render_timing_live(live_result)
        if live_result.chunks:
            with st.expander("Chunk details", expanded=False):
                for chunk in live_result.chunks:
                    st.markdown(
                        f"**#{chunk.index + 1}** "
                        f"({chunk.start_sec:.1f}s–{chunk.end_sec:.1f}s, {chunk.elapsed_sec:.2f}s)"
                    )
                    st.caption(f"Transcript: {chunk.transcript}")
                    st.caption(f"Translation: {chunk.translation}")

    spoken = st.session_state.get("live_last_sentences") or []
    if spoken:
        st.caption("TTS queue spoke: " + " | ".join(spoken))

    left, right = st.columns(2)
    with left:
        st.markdown("**Transcript**")
        st.text_area("Transcript", height=260, label_visibility="collapsed", key="transcript")
    with right:
        st.markdown("**Translation**")
        st.text_area("Translation", height=260, label_visibility="collapsed", key="translation")

    if st.session_state.tts_audio:
        st.audio(st.session_state.tts_audio, format="audio/wav")