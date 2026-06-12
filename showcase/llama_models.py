"""llama.cpp backend — the proven "monolith" inference stack, for local runs.

Mirrors the public API of `models.py` (transcribe_and_translate, transcribe_file,
translate_text, chat_stream, ocr_images, GEMMA_ID, WHISPER_ID, MULTIMODAL) but
serves gemma-4-12b as a GGUF through `llama-server` with the mmproj projector —
the same stack as the first working version of the full app (and the full app
today). One model handles chat, translation, vision OCR *and* speech-to-text
(Gemma 4's built-in audio encoder), so there is no torch / transformers /
bitsandbytes version matrix at all.

This backend cannot run on the free ZeroGPU Space tier (ZeroGPU timeslices CUDA
for PyTorch only) — there `models.py` is used instead. It runs identically on a
local GPU or a paid Docker GPU Space.

Environment:
  SHOWCASE_LLAMA_URL    reuse an already-running server (skip spawning)
  SHOWCASE_LLAMA_BIN    path to llama-server (default: found on PATH)
  SHOWCASE_LLAMA_PORT   port for the spawned server (default 8094)
  SHOWCASE_GGUF         model file (default: <repo>/models/gemma-4-12b-it-Q6_K.gguf,
                        downloaded from unsloth/gemma-4-12b-it-GGUF if missing)
  SHOWCASE_MMPROJ       projector file (default: <repo>/models/mmproj-F16.gguf)
"""

from __future__ import annotations

import atexit
import base64
import io
import os
import shutil
import subprocess
import time
import wave
from collections.abc import Iterator
from pathlib import Path

import httpx
import numpy as np
from prompts import CHAT_SYSTEM, OCR_SYSTEM, build_translate_messages, language_label

GGUF_REPO = "unsloth/gemma-4-12b-it-GGUF"
GGUF_FILE = "gemma-4-12b-it-Q6_K.gguf"
MMPROJ_FILE = "mmproj-F16.gguf"

GEMMA_ID = f"{GGUF_FILE} (llama.cpp)"
WHISPER_ID = "gemma-4 built-in audio STT"
MULTIMODAL = True

_PORT = int(os.environ.get("SHOWCASE_LLAMA_PORT", "8094"))
BASE_URL = os.environ.get("SHOWCASE_LLAMA_URL", f"http://127.0.0.1:{_PORT}").rstrip("/")

# Gemma 4 sampling as shipped in the full app's config (and the model card).
GEN_DEFAULTS = {"temperature": 1.0, "top_p": 0.95}
TIMEOUT = httpx.Timeout(600.0, connect=10.0)

NO_SPEECH = "[NO_SPEECH]"
ASR_PROMPT = (
    "Transcribe the following speech segment in its original language.\n\n"
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    f"* If the segment contains no intelligible speech, output exactly {NO_SPEECH} "
    "and nothing else — never apologize or explain.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7 and not "
    "one point seven, and write 3 instead of three."
)

MAX_STT_CHUNK_SECONDS = 28
STT_SAMPLE_RATE = 16000  # Gemma's audio encoder rate; the full app sends 16 kHz too


# --- server management -------------------------------------------------------


def _health_ok(timeout: float = 2.0) -> bool:
    try:
        return httpx.get(f"{BASE_URL}/health", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def _find_binary() -> str:
    override = os.environ.get("SHOWCASE_LLAMA_BIN")
    if override:
        return override
    for name in ("llama-server", "llama-server.exe"):
        path = shutil.which(name)
        if path:
            return path
    raise FileNotFoundError(
        "llama-server not found on PATH. Install llama.cpp from "
        "https://github.com/ggml-org/llama.cpp/releases (or set SHOWCASE_LLAMA_BIN)."
    )


def _resolve_model_files() -> tuple[Path, Path]:
    models_dir = Path(__file__).resolve().parents[1] / "models"
    gguf = Path(os.environ.get("SHOWCASE_GGUF") or models_dir / GGUF_FILE)
    mmproj = Path(os.environ.get("SHOWCASE_MMPROJ") or models_dir / MMPROJ_FILE)
    if not gguf.exists() or not mmproj.exists():
        from huggingface_hub import hf_hub_download

        gguf = Path(hf_hub_download(GGUF_REPO, GGUF_FILE))
        mmproj = Path(hf_hub_download(GGUF_REPO, MMPROJ_FILE))
    return gguf, mmproj


_proc: subprocess.Popen | None = None


def _stop_server() -> None:
    global _proc
    if _proc and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None


def ensure_server(startup_timeout: float = 300.0) -> None:
    """Reuse a healthy server at BASE_URL, else spawn one and wait for /health."""
    global _proc
    if _health_ok():
        return
    if "SHOWCASE_LLAMA_URL" in os.environ:
        raise RuntimeError(
            f"No llama-server responding at SHOWCASE_LLAMA_URL={BASE_URL} "
            "(refusing to spawn one because an explicit URL was given)."
        )
    if _proc and _proc.poll() is None:
        pass  # already spawned, still starting up
    else:
        gguf, mmproj = _resolve_model_files()
        cmd = [
            _find_binary(),
            "--model", str(gguf),
            "--mmproj", str(mmproj),
            "--host", "127.0.0.1",
            "--port", str(_PORT),
            "-c", "16384",
            "-ngl", "999",
            "--jinja",
        ]
        print(f"[showcase] starting llama-server: {gguf.name} + {mmproj.name}", flush=True)
        _proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace",
        )
        atexit.register(_stop_server)

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if _proc and _proc.poll() is not None:
            tail = (_proc.stdout.read() or "")[-2000:] if _proc.stdout else ""
            raise RuntimeError(f"llama-server exited early (code {_proc.returncode}).\n{tail}")
        if _health_ok():
            print("[showcase] llama-server ready.", flush=True)
            return
        time.sleep(1.0)
    raise TimeoutError(f"llama-server not ready within {startup_timeout:.0f}s at {BASE_URL}")


# --- OpenAI-compatible chat calls --------------------------------------------


def _chat(messages: list[dict], *, max_tokens: int = 1024, **overrides) -> str:
    ensure_server()
    body = {
        "model": "gemma-4-12b-it",
        "messages": messages,
        "max_tokens": max_tokens,
        **GEN_DEFAULTS,
        **overrides,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.post(f"{BASE_URL}/v1/chat/completions", json=body)
        response.raise_for_status()
        data = response.json()
    return (data["choices"][0]["message"].get("content") or "").strip()


def _chat_stream_request(messages: list[dict], *, max_tokens: int = 1024) -> Iterator[str]:
    """Yield the accumulated reply as deltas arrive (SSE)."""
    import json

    ensure_server()
    body = {
        "model": "gemma-4-12b-it",
        "messages": messages,
        "max_tokens": max_tokens,
        **GEN_DEFAULTS,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    reply = ""
    with httpx.Client(timeout=TIMEOUT) as client, client.stream(
        "POST", f"{BASE_URL}/v1/chat/completions", json=body
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload.strip() == "[DONE]":
                break
            delta = json.loads(payload)["choices"][0].get("delta", {})
            token = delta.get("content")
            if token:
                reply += token
                yield reply


# --- media helpers -------------------------------------------------------


def resample(audio: np.ndarray, rate: int, target_rate: int) -> np.ndarray:
    """Linear-interpolation resample, mono float32 (no scipy/ffmpeg needed)."""
    if rate == target_rate or audio.size == 0:
        return audio.astype(np.float32)
    duration = audio.size / rate
    target_len = max(int(duration * target_rate), 1)
    src_t = np.linspace(0.0, duration, audio.size, endpoint=False)
    dst_t = np.linspace(0.0, duration, target_len, endpoint=False)
    return np.interp(dst_t, src_t, audio).astype(np.float32)


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Mono float32 [-1, 1] → 16-bit PCM WAV bytes (stdlib only)."""
    pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


def audio_part(audio: np.ndarray, sample_rate: int) -> dict:
    b64 = base64.b64encode(wav_bytes(audio, sample_rate)).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}


def image_part(image) -> dict:
    """PIL image → base64 data-URL content part."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


# --- public API (same signatures as models.py) -------------------------------


def _transcribe(audio: np.ndarray, sample_rate: int, source_lang: str | None) -> str:
    prompt = ASR_PROMPT
    if source_lang:
        prompt = prompt.replace(
            "in its original language", f"in {language_label(source_lang)}"
        )
    audio = resample(np.asarray(audio, dtype=np.float32), sample_rate, STT_SAMPLE_RATE)
    sample_rate = STT_SAMPLE_RATE
    chunk_len = MAX_STT_CHUNK_SECONDS * sample_rate
    pieces: list[str] = []
    for start in range(0, len(audio), chunk_len):
        chunk = audio[start : start + chunk_len]
        if len(chunk) < int(0.2 * sample_rate):
            continue
        text = _chat(
            [
                {"role": "system", "content": "You transcribe speech accurately."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        audio_part(chunk, sample_rate),
                    ],
                },
            ],
            max_tokens=512,
        )
        if text and NO_SPEECH not in text:
            pieces.append(text)
    return " ".join(pieces).strip()


def transcribe_and_translate(
    audio: np.ndarray,
    sample_rate: int,
    source_lang: str | None,
    target_lang: str,
    tone: str,
    context: list[tuple[str, str]],
) -> tuple[str, str]:
    """One live-interpreter utterance: Gemma audio STT + Gemma MT."""
    transcript = _transcribe(audio, sample_rate, source_lang)
    if not transcript:
        return "", ""
    translation = _chat(
        build_translate_messages(
            transcript,
            source_lang=source_lang,
            target_lang=target_lang,
            tone=tone,
            context=context,
        ),
        max_tokens=256,
    )
    return transcript, translation


def transcribe_file(audio: np.ndarray, sample_rate: int, source_lang: str | None) -> str:
    return _transcribe(audio, sample_rate, source_lang)


def translate_text(text: str, source_lang: str | None, target_lang: str, tone: str) -> str:
    # 2048: also serves OCR-page translation, which can exceed 1024 tokens.
    return _chat(
        build_translate_messages(text, source_lang=source_lang, target_lang=target_lang, tone=tone),
        max_tokens=2048,
    )


def chat_stream(history: list[dict], image=None) -> Iterator[str]:
    """Streaming chat; `history` is [{'role','content'}] text turns, optional PIL image."""
    messages: list[dict] = [{"role": "system", "content": CHAT_SYSTEM}]
    for i, turn in enumerate(history):
        if image is not None and i == len(history) - 1 and turn["role"] == "user":
            messages.append(
                {
                    "role": "user",
                    "content": [
                        image_part(image),
                        {"type": "text", "text": turn["content"]},
                    ],
                }
            )
        else:
            messages.append({"role": turn["role"], "content": turn["content"]})
    yield from _chat_stream_request(messages, max_tokens=1024)


def ocr_images(images: list, instructions: str = "") -> str:
    """Gemma vision OCR over one or more page images (markdown output)."""
    pages: list[str] = []
    for image in images:
        user_text = "Extract all text from this image."
        if instructions.strip():
            user_text += f" Additional instructions: {instructions.strip()}"
        pages.append(
            _chat(
                [
                    {"role": "system", "content": OCR_SYSTEM},
                    {
                        "role": "user",
                        "content": [image_part(image), {"type": "text", "text": user_text}],
                    },
                ],
                max_tokens=2048,
            )
        )
    return "\n\n---\n\n".join(pages)
