"""LocalLLM showcase — Gradio app for Hugging Face Spaces (ZeroGPU).

⚠️ This is a REDUCED-CAPABILITY online demo. The full project — offline
llama.cpp inference, OpenAI-compatible gateway, agent with skills, React web
UI, and the streaming voice-to-voice pipeline — lives at:
https://github.com/murai1998/LocalLLM
"""

from __future__ import annotations

import asyncio
import os

import gradio as gr
import numpy as np
import piper_voices
from fastrtc import (
    AlgoOptions,
    ReplyOnPause,
    WebRTC,
    get_cloudflare_turn_credentials_async,
)
from interpreter import build_reply_handler, to_mono_float32
from prompts import DEFAULT_TONE, GITHUB_URL, LANGUAGE_LABELS, TONE_PRESETS
from zerogpu import ON_SPACES, friendly_errors

# Backend selection (env SHOWCASE_BACKEND, or sensible defaults):
#   fake          echo stubs — UI rehearsal, no models at all (SHOWCASE_FAKE=1 alias)
#   llama         llama.cpp: gemma-4-12b GGUF Q6_K via llama-server — the proven
#                 local stack (vision + audio via mmproj, no torch needed).
#                 Default everywhere except HF Spaces.
#   transformers  torch/transformers models.py — required on ZeroGPU Spaces
#                 (ZeroGPU timeslices CUDA for PyTorch only; llama.cpp can't use it).
BACKEND = os.environ.get("SHOWCASE_BACKEND", "").strip().lower()
if not BACKEND:
    if os.environ.get("SHOWCASE_FAKE", "").lower() in ("1", "true", "yes"):
        BACKEND = "fake"
    elif os.environ.get("SPACE_ID"):
        BACKEND = "transformers"
    else:
        BACKEND = "llama"

if BACKEND == "fake":
    import fake_models as models
elif BACKEND == "llama":
    import llama_models as models
else:
    import models

# Mermaid diagrams in the comparison page render on GitHub (not in HF's blob
# viewer), so the banner links to the GitHub copy of the file.
COMPARISON_URL = f"{GITHUB_URL}/blob/main/showcase/COMPARISON.md"

OUTPUT_RATE = 22050
MAX_OCR_PAGES = 3

LANG_CHOICES = sorted(((label, code) for code, label in LANGUAGE_LABELS.items()))
TONE_CHOICES = [(preset["label"], tone_id) for tone_id, preset in TONE_PRESETS.items()]

DEMO_BANNER = f"""
<div style="border:2px solid #f59e0b; border-radius:12px; padding:14px 18px;
            background:rgba(245,158,11,.08); margin-bottom:8px; font-size:15px;">
  <strong>⚠️ Reduced-capability demo.</strong> This Space runs trimmed-down models under
  free ZeroGPU limits (daily GPU quota per visitor, per-utterance GPU calls, no
  continuous pipeline). The <strong>full LocalLLM</strong> — a private, fully offline
  suite with llama.cpp Q6_K inference, an OpenAI-compatible gateway, a tool-using agent,
  a React web UI, and true streaming voice-to-voice translation (≤ 8 s lag) — runs on
  your own GPU with <em>no quotas and no data leaving your machine</em>:<br>
  👉 <a href="{GITHUB_URL}" target="_blank"><strong>{GITHUB_URL}</strong></a> ·
  <a href="{COMPARISON_URL}" target="_blank">demo vs. full version — 1-page comparison with
  architecture diagrams</a>
</div>
"""

FOOTER = f"""
<div style="text-align:center; opacity:.7; font-size:13px; margin-top:12px;">
  Demo version · full local deployment instructions in the
  <a href="{GITHUB_URL}" target="_blank">GitHub repo</a> ·
  models: {models.GEMMA_ID} + {models.WHISPER_ID} + Piper TTS (CPU)
</div>
"""


def _resample(rate: int, audio: np.ndarray, target_rate: int) -> np.ndarray:
    if rate == target_rate or audio.size == 0:
        return audio
    duration = audio.size / rate
    target_len = max(int(duration * target_rate), 1)
    src_t = np.linspace(0.0, duration, audio.size, endpoint=False)
    dst_t = np.linspace(0.0, duration, target_len, endpoint=False)
    return np.interp(dst_t, src_t, audio.astype(np.float64)).astype(np.int16)


def _tts_at_output_rate(text: str, *, language: str, voice_id: str | None):
    rate, voice = piper_voices.synthesize(text, language=language, voice_id=voice_id)
    return OUTPUT_RATE, _resample(rate, voice, OUTPUT_RATE)


# --- Tab 1: Live Interpreter ---------------------------------------------


def _voice_choices(lang_code: str) -> list[tuple[str, str]]:
    """Voice options for a language; an explicit placeholder when none exist
    (e.g. Japanese/Korean have no Piper voice) so the dropdown is never blank."""
    options = [(o["label"], o["id"]) for o in piper_voices.voice_options_for_language(lang_code)]
    return options or [("No TTS voice — text only", "")]


# Pause detection: a window of `audio_chunk_duration` seconds with less than
# `speech_threshold` seconds of speech ends the utterance. The fastrtc defaults
# (0.6 s window) fire on any short breath and chop sentences in half — require
# a longer quiet window before translating.
LIVE_VAD = AlgoOptions(
    audio_chunk_duration=1.2,
    started_talking_threshold=0.2,
    speech_threshold=0.15,
)

# A browser can't reach a cloud Space's media ports directly — WebRTC needs a
# TURN relay. fastrtc's default helper, when HF_TOKEN is set, *always* fetches
# from the free HF community relay (turn.fastrtc.org) and ignores any Cloudflare
# keys; that host has intermittent DNS/availability failures (the "Temporary
# failure in name resolution" popup). Prefer the user's own Cloudflare TURN keys
# (direct to the stable rtc.live.cloudflare.com), retry the community relay a few
# times, and surface a clear setup message instead of a raw DNS error.
_CF_KEY_ID = os.environ.get("CLOUDFLARE_TURN_KEY_ID")
_CF_KEY_TOKEN = os.environ.get("CLOUDFLARE_TURN_KEY_API_TOKEN")


async def _turn_credentials():
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            if _CF_KEY_ID and _CF_KEY_TOKEN:
                # hf_token="" forces the direct-Cloudflare path (an HF_TOKEN in
                # the env would otherwise win and route to turn.fastrtc.org).
                return await get_cloudflare_turn_credentials_async(
                    turn_key_id=_CF_KEY_ID,
                    turn_key_api_token=_CF_KEY_TOKEN,
                    hf_token="",
                )
            return await get_cloudflare_turn_credentials_async()
        except Exception as exc:  # noqa: BLE001 — retry any fetch/DNS failure
            last_exc = exc
            await asyncio.sleep(0.5 * (attempt + 1))
    name = last_exc.__class__.__name__ if last_exc else "error"
    raise gr.Error(
        "Couldn't reach a WebRTC TURN relay for the Live interpreter "
        f"({name}). For a reliable relay, add free Cloudflare TURN keys as Space "
        "secrets (CLOUDFLARE_TURN_KEY_ID + CLOUDFLARE_TURN_KEY_API_TOKEN); "
        "otherwise retry — the free HF relay is sometimes briefly unavailable. "
        "The other tabs don't need this."
    )


def _make_live_handler():
    """Fresh reply closure — each one owns a private rolling transcript."""
    return build_reply_handler(
        friendly_errors(models.transcribe_and_translate), _tts_at_output_rate
    )


class StatefulReplyOnPause(ReplyOnPause):
    """fastrtc clones the handler per WebRTC connection via copy(); build a NEW
    reply closure for each clone so every visitor gets their own history
    (a shared closure would interleave conversations across visitors).

    can_interrupt=False: with barge-in enabled (the fastrtc default), every
    detected pause clears the playback queue and kills the running reply — TTS
    echo picked up by the mic registers as speech, the next quiet moment as a
    pause, and the translation audio is cut off or never heard at all. An
    interpreter should finish speaking; the mic is simply ignored meanwhile.
    """

    def __init__(self, fn, **kwargs):
        kwargs.setdefault("can_interrupt", False)
        super().__init__(fn, **kwargs)

    def copy(self):
        return StatefulReplyOnPause(
            _make_live_handler(),
            algo_options=self.algo_options,
            output_sample_rate=self.output_sample_rate,
        )


def build_interpreter_tab() -> None:
    gr.Markdown(
        "Speak naturally — when you pause, your words are transcribed, translated, and "
        "spoken back. Each utterance uses a few seconds of your free daily GPU quota. "
        f"(The [full local version]({GITHUB_URL}) streams continuously with no quota.)"
    )
    with gr.Row():
        source = gr.Dropdown(
            [("Auto-detect", "")] + LANG_CHOICES, value="", label="From"
        )
        target = gr.Dropdown(LANG_CHOICES, value="es", label="To")
        tone = gr.Dropdown(TONE_CHOICES, value=DEFAULT_TONE, label="Tone")
        # allow_custom_value: while the target language changes, fastrtc re-reads
        # all stream inputs; the voice value can briefly belong to the previous
        # language's choices and strict validation throws. piper_voices falls
        # back to the new language's default voice for unknown ids.
        voice = gr.Dropdown(_voice_choices("es"), label="Voice", allow_custom_value=True)

    target.change(
        lambda code: gr.Dropdown(
            choices=_voice_choices(code),
            value=(_voice_choices(code) or [(None, None)])[0][1],
        ),
        inputs=[target],
        outputs=[voice],
    )

    webrtc = WebRTC(
        label="Microphone",
        mode="send-receive",
        modality="audio",
        # fastrtc defaults to full_screen=True, which overlays the whole page
        # with a click-swallowing container and freezes the rest of the UI.
        full_screen=False,
        rtc_configuration=_turn_credentials if ON_SPACES else None,
    )
    with gr.Row():
        transcript_src = gr.Textbox(label="🗣 Transcript", lines=12, interactive=False)
        transcript_tgt = gr.Textbox(label="🌐 Translation", lines=12, interactive=False)

    webrtc.stream(
        StatefulReplyOnPause(
            _make_live_handler(),
            algo_options=LIVE_VAD,
            output_sample_rate=OUTPUT_RATE,
        ),
        inputs=[webrtc, source, target, tone, voice],
        outputs=[webrtc],
        concurrency_limit=5,
        time_limit=300,
    )
    webrtc.on_additional_outputs(
        lambda src, tgt: (src, tgt), outputs=[transcript_src, transcript_tgt]
    )


# --- Tab 2: Chat -----------------------------------------------------------


@friendly_errors
def _chat_respond(message: str, image, chat_history: list[dict]):
    message = (message or "").strip()
    if not message:
        yield chat_history, ""
        return
    chat_history = chat_history + [{"role": "user", "content": message}]
    turns = [m for m in chat_history if m["role"] in ("user", "assistant")]
    yield chat_history + [{"role": "assistant", "content": "…"}], ""
    for partial in models.chat_stream(turns, image=image):
        yield chat_history + [{"role": "assistant", "content": partial}], ""


def build_chat_tab() -> None:
    gr.Markdown(
        "Chat with the demo model, optionally about an image. "
        f"(The [full local version]({GITHUB_URL}) adds a tool-using agent with selectable "
        "skills, document attachments, and unlimited context — entirely offline.)"
    )
    chatbot = gr.Chatbot(type="messages", height=420)
    with gr.Row(equal_height=True):
        image = gr.Image(
            type="pil",
            label="Optional image",
            scale=2,
            height=200,
            elem_id="chat-image",
        )
        message = gr.Textbox(
            label="Message", placeholder="Ask anything…", scale=3, lines=2
        )
    send = gr.Button("Send", variant="primary")
    for trigger in (send.click, message.submit):
        trigger(
            _chat_respond,
            inputs=[message, image, chatbot],
            outputs=[chatbot, message],
        )


# --- Tab 3: Transcribe & translate a file ----------------------------------


@friendly_errors
def _batch_run(audio, source_lang: str, do_translate: bool, target_lang: str, tone: str):
    if audio is None:
        raise gr.Error("Record or upload audio first.")
    sample_rate, pcm = audio
    samples = to_mono_float32(pcm)
    transcript = models.transcribe_file(samples, sample_rate, source_lang or None)
    translation = ""
    speech = None
    if do_translate and transcript:
        translation = models.translate_text(
            transcript, source_lang or None, target_lang, tone
        )
        if translation and piper_voices.tts_supported(target_lang):
            rate, voice = piper_voices.synthesize(translation, language=target_lang)
            speech = (rate, voice)
    return transcript, translation, speech


def build_batch_tab() -> None:
    gr.Markdown(
        "Upload or record audio for one-shot transcription and translation with a "
        "spoken result."
    )
    # elem_id + CSS min-height keep the layout pinned: the record view of
    # gr.Audio is taller than the upload view and otherwise reflows the page
    # the moment the mic icon is clicked.
    audio = gr.Audio(
        sources=["upload", "microphone"], type="numpy", label="Audio",
        elem_id="batch-audio",
    )
    with gr.Row():
        source = gr.Dropdown([("Auto-detect", "")] + LANG_CHOICES, value="", label="Language")
        do_translate = gr.Checkbox(value=True, label="Also translate")
        target = gr.Dropdown(LANG_CHOICES, value="en", label="To")
        tone = gr.Dropdown(TONE_CHOICES, value=DEFAULT_TONE, label="Tone")
    run = gr.Button("Transcribe", variant="primary")
    with gr.Row():
        transcript = gr.Textbox(label="Transcript", lines=8)
        translation = gr.Textbox(label="Translation", lines=8)
    speech = gr.Audio(label="Spoken translation", interactive=False)
    run.click(
        _batch_run,
        inputs=[audio, source, do_translate, target, tone],
        outputs=[transcript, translation, speech],
    )


# --- Tab 4: Document OCR ----------------------------------------------------


@friendly_errors
def _ocr_run(file, instructions: str, do_translate: bool, target_lang: str):
    if file is None:
        raise gr.Error("Upload an image or PDF first.")
    from PIL import Image

    path = file if isinstance(file, str) else file.name
    if path.lower().endswith(".pdf"):
        import fitz  # PyMuPDF

        images = []
        with fitz.open(path) as doc:
            for page in doc[:MAX_OCR_PAGES]:
                pix = page.get_pixmap(dpi=150)
                images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        if len(doc) > MAX_OCR_PAGES:
            gr.Info(f"Demo limit: first {MAX_OCR_PAGES} pages only (full version: unlimited).")
    else:
        images = [Image.open(path).convert("RGB")]
    extracted = models.ocr_images(images, instructions or "")
    translation = ""
    if do_translate and extracted.strip():
        translation = models.translate_text(extracted, None, target_lang, "exact")
    return extracted, translation


def build_ocr_tab() -> None:
    gr.Markdown(
        f"Vision OCR for images and scanned PDFs (demo limit: {MAX_OCR_PAGES} pages — the "
        f"[full local version]({GITHUB_URL}) is about *10x faster* and handles whole documents and DOCX/PDF "
        "translation)."
    )
    file = gr.File(label="Image or PDF", file_types=["image", ".pdf"])
    instructions = gr.Textbox(
        label="Instructions (optional)",
        placeholder="e.g. extract only the table of line items",
    )
    with gr.Row():
        do_translate = gr.Checkbox(value=True, label="Also translate")
        target = gr.Dropdown(LANG_CHOICES, value="en", label="Translate to")
    run = gr.Button("Extract text", variant="primary")
    with gr.Row():
        output = gr.Textbox(label="Extracted text (original)", lines=16, show_copy_button=True)
        translated = gr.Textbox(label="Translation", lines=16, show_copy_button=True)
    run.click(
        _ocr_run,
        inputs=[file, instructions, do_translate, target],
        outputs=[output, translated],
    )


# --- App --------------------------------------------------------------------

# Keep the compact "Optional image" upload widget's prompt text inside its box —
# Gradio's default drop/upload copy overflows a short component otherwise.
CSS = """
#chat-image [data-testid="block-label"] { z-index: 2; }
#chat-image .wrap { font-size: 13px; line-height: 1.3; gap: 4px; }
#chat-image .wrap .or { margin: 0; }
/* gr.Audio's upload view (280px) and record view (215px) differ in height;
   reserve the larger of the two so toggling the mic never reflows the tab. */
#batch-audio { min-height: 280px; }
"""

with gr.Blocks(
    title="LocalLLM — Multitool (demo)", theme=gr.themes.Soft(), css=CSS
) as demo:
    gr.HTML(DEMO_BANNER)
    gr.Markdown("# 🎙️ LocalLLM — with Voice Interpreter, Chat, OCR, and STT transcribe")
    with gr.Tabs():
        with gr.Tab("🎙️ Live Interpreter"):
            build_interpreter_tab()
        with gr.Tab("💬 Chat"):
            build_chat_tab()
        with gr.Tab("🎧 Transcribe a file"):
            build_batch_tab()
        with gr.Tab("📄 Document OCR"):
            build_ocr_tab()
    gr.HTML(FOOTER)

def warm_backend() -> None:
    """Start the llama-server (9 GB GGUF load) in the background at app startup
    so the first click doesn't absorb the whole warmup."""
    if hasattr(models, "ensure_server"):
        import threading

        threading.Thread(target=models.ensure_server, daemon=True).start()


if __name__ == "__main__":
    piper_voices.warmup("es")
    warm_backend()
    demo.launch()
