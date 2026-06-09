#!/usr/bin/env python3
"""Translate tab UI — Gemma audio + Phase 2 VAD chunking + local Piper TTS."""

from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path

import streamlit as st

from localllm.client.factory import create_llm_client
from localllm.config import get_settings
from localllm.pipelines.translate import (
    LANGUAGE_LABELS,
    TONE_PRESETS,
    ToneId,
    TranslationResult,
    retranslate_transcript,
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
LIVE_UPLOAD_KEY = "translate_live_upload"
LIVE_MIC_KEY = "translate_live_mic"
LIVE_STAGING_ROOT = Path(tempfile.gettempdir()) / "localllm" / "translate_live"
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
        "live_chunk_audio": None,
        "live_chunk_audio_name": None,
        "live_chunk_path": None,
        "live_recorded_audio": None,
        "live_recorded_audio_name": "recording.wav",
        "live_recorded_path": None,
        "live_chunk_source": None,
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


def _live_chunk_staging_dir() -> Path:
    if "translate_live_session_id" not in st.session_state:
        st.session_state.translate_live_session_id = uuid.uuid4().hex
    path = LIVE_STAGING_ROOT / st.session_state.translate_live_session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _persist_live_bytes(data: bytes, filename: str, *, source: str) -> Path:
    safe_name = Path(filename).name or "audio.wav"
    dest = _live_chunk_staging_dir() / safe_name
    dest.write_bytes(data)
    st.session_state.live_chunk_audio = data
    st.session_state.live_chunk_audio_name = safe_name
    st.session_state.live_chunk_path = str(dest)
    st.session_state.live_chunk_source = source
    return dest


def _valid_audio_path(path: Path | None) -> Path | None:
    if path is None or not path.is_file():
        return None
    if path.stat().st_size < 128:
        return None
    return path


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


def _source_lang_for_retranslate(source_lang: str | None) -> str | None:
    if source_lang is not None:
        return source_lang
    batch: TranslationResult | None = st.session_state.get("translate_result")
    if batch is not None and batch.source_language and batch.source_language != "auto":
        return batch.source_language
    live: ChunkedTranslationResult | None = st.session_state.get("translate_live_result")
    if live is not None and live.source_language and live.source_language != "auto":
        return live.source_language
    return None


def _retranslate_existing(
    *,
    source_lang: str | None,
    target_lang: str,
    tone: ToneId,
) -> bool:
    transcript = st.session_state.get("transcript", "").strip()
    if not transcript:
        st.warning("Translate audio first to get a transcript.")
        return False

    result = retranslate_transcript(
        transcript,
        source_lang=_source_lang_for_retranslate(source_lang),
        target_lang=target_lang,
        tone=tone,
        llm_client=get_translate_llm_client(),
    )
    _store_batch_result(result)
    return True


def _play_translation(voice_id: str | None) -> bool:
    translation = st.session_state.get("translation", "")
    target_lang = st.session_state.get("translate_target_lang", "es")
    if not translation:
        st.warning("Translate audio first.")
        return False
    if not PIPER_AVAILABLE:
        st.error("Install Piper TTS: `pip install piper-tts`")
        return False

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
    return True


def _uploaded_file_bytes(uploaded) -> bytes | None:
    if uploaded is None:
        return None
    try:
        data = uploaded.getvalue()
        if data:
            return data
    except Exception:
        pass
    try:
        if hasattr(uploaded, "seek"):
            uploaded.seek(0)
        data = uploaded.read()
        if data:
            return data
    except Exception:
        pass
    return None


def _stage_live_upload(uploaded) -> Path | None:
    if uploaded is None:
        return None
    data = _uploaded_file_bytes(uploaded)
    if not data:
        return None
    name = getattr(uploaded, "name", None) or "audio.wav"
    return _persist_live_bytes(data, name, source="upload")


def _on_live_upload_changed() -> None:
    _stage_live_upload(st.session_state.get(LIVE_UPLOAD_KEY))


def _stage_live_recording(audio_input) -> Path | None:
    if audio_input is None:
        return None
    data = audio_input.getvalue()
    if not data:
        return None
    recorded_path = _live_chunk_staging_dir() / "recording.wav"
    recorded_path.write_bytes(data)
    st.session_state.live_recorded_audio = data
    st.session_state.live_recorded_audio_name = "recording.wav"
    st.session_state.live_recorded_path = str(recorded_path)
    st.session_state.live_chunk_source = "recording"
    return recorded_path


def _resolve_live_audio_path(*, at_run: bool = False) -> tuple[Path | None, str | None]:
    """Resolve live audio from widget (at click) or staged disk path."""
    uploaded = st.session_state.get(LIVE_UPLOAD_KEY)
    if uploaded is not None:
        staged = _stage_live_upload(uploaded)
        if staged is not None:
            return staged, "upload"

    mic = st.session_state.get(LIVE_MIC_KEY)
    if mic is not None:
        staged = _stage_live_recording(mic)
        if staged is not None:
            return staged, "recording"

    if at_run:
        for path_key, source in (
            ("live_chunk_path", "upload"),
            ("live_recorded_path", "recording"),
        ):
            path = _valid_audio_path(
                Path(st.session_state[path_key])
                if st.session_state.get(path_key)
                else None
            )
            if path is not None:
                return path, source

    return None, None


def _run_live_chunked(
    audio_path: Path,
    *,
    source_lang: str | None,
    target_lang: str,
    tone: ToneId,
    voice_id: str | None,
    auto_tts: bool,
) -> None:
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
            audio_path,
            source_lang=source_lang,
            target_lang=target_lang,
            tone=tone,
            llm_client=get_translate_llm_client(),
            on_progress=on_progress,
        )
    except Exception as exc:
        progress.empty()
        status.error(f"Live chunked translate failed: {exc}")
        return

    _store_live_result(result)
    _reset_tts_queue()

    if result.chunk_count == 0:
        progress.empty()
        status.error(
            "No speech chunks were transcribed. "
            "Check the file has audible audio and is a supported format (WAV, MP3, M4A, …)."
        )
        return

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

    input_methods = ["Upload audio", "Record voice", "Live (chunked)"]
    if "translate_input_method" not in st.session_state:
        st.session_state.translate_input_method = input_methods[0]
    input_method = st.radio(
        "Input method",
        input_methods,
        horizontal=True,
        label_visibility="collapsed",
        key="translate_input_method",
    )

    if input_method == "Upload audio":
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

    elif input_method == "Record voice":
        st.markdown("Record, then click translate.")
        audio_input = st.audio_input("Microphone", key="translate_mic")
        if audio_input is not None:
            st.session_state.recorded_audio = audio_input.getvalue()
            st.session_state.recorded_audio_name = "recording.wav"
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

    else:
        settings = get_settings()
        live_cfg = settings.translate.live
        st.markdown(
            f"Phase 2: VAD splits audio into **{live_cfg.min_chunk_seconds:.0f}–"
            f"{live_cfg.max_chunk_seconds:.0f} s** chunks with overlap. "
            "Transcript and translation update per chunk; TTS speaks **completed sentences** only."
        )
        st.session_state.live_auto_tts = st.checkbox(
            "Auto-speak completed sentences (TTS queue)",
            value=st.session_state.live_auto_tts,
            key="live_auto_tts_cb",
        )
        st.file_uploader(
            "Audio for live chunking",
            type=sorted(SUPPORTED_EXTENSIONS),
            key=LIVE_UPLOAD_KEY,
            on_change=_on_live_upload_changed,
        )
        st.audio_input("Or record", key=LIVE_MIC_KEY)

        live_path, live_source = _resolve_live_audio_path()
        if live_path is not None and live_source == "upload":
            size_kb = live_path.stat().st_size // 1024
            st.caption(
                f"Ready: uploaded file **{st.session_state.get('live_chunk_audio_name', live_path.name)}** "
                f"({size_kb} KB)"
            )
        elif live_path is not None and live_source == "recording":
            st.caption("Ready: microphone recording")

        if st.button("Run live chunked translate", type="primary", key="translate_live_btn"):
            audio_path, source = _resolve_live_audio_path(at_run=True)
            if audio_path is None:
                st.warning("Upload an audio file or record from the microphone first.")
            else:
                _run_live_chunked(
                    audio_path,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    tone=tone,
                    voice_id=voice_id,
                    auto_tts=st.session_state.live_auto_tts,
                )
                st.rerun()

    has_transcript = bool(st.session_state.get("transcript", "").strip())
    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("Play full translation", type="secondary", key="translate_play_btn"):
            if _play_translation(voice_id):
                st.rerun()
    with action_cols[1]:
        if has_transcript and st.button(
            "Re-translate with current settings",
            type="secondary",
            key="translate_retranslate_btn",
        ):
            if _retranslate_existing(
                source_lang=source_lang,
                target_lang=target_lang,
                tone=tone,
            ):
                st.rerun()

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
        st.markdown("**Translation audio**")
        st.audio(st.session_state.tts_audio, format="audio/wav")