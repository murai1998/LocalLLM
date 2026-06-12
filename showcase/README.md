---
title: LocalLLM Voice Interpreter (demo)
emoji: 🎙️
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: "5.50.0"
python_version: "3.12"
app_file: app.py
license: apache-2.0
models:
  - google/gemma-4-12b-it
  - openai/whisper-large-v3-turbo
short_description: Voice-to-voice translation demo — full offline app on GitHub
---

# 🎙️ LocalLLM — Voice Interpreter (demo)

> ## ⚠️ This is a **reduced-capability demo**
>
> This Space showcases a trimmed-down version of **LocalLLM** running under free
> ZeroGPU limits: per-utterance GPU calls, daily visitor GPU quotas, transformer
> models instead of the optimized llama.cpp stack, and no continuous streaming
> pipeline.
>
> **The full application runs entirely on your own machine — offline, private,
> unlimited — and can be deployed from the GitHub repository:**
>
> ### 👉 https://github.com/murai1998/LocalLLM
>
> The full local version adds: llama.cpp **Q6_K** inference behind an
> OpenAI-compatible gateway · a **React web UI** · true **streaming voice-to-voice
> translation** (continuous microphone, silence-aware segmentation, pipelined
> STT → MT → TTS, ≤ 8 s lag) · a tool-using **agent with selectable skills** ·
> multi-file chat attachments · whole-document OCR and translation · batch
> transcription — with **no quotas and no data ever leaving your machine**.

## What this demo does

| Tab | What happens | GPU cost per use |
|---|---|---|
| 🎙️ **Live Interpreter** | Speak; on pause your utterance is transcribed (Whisper large-v3-turbo), translated (Gemma) and spoken back (Piper, CPU) | ~3–6 s |
| 💬 **Chat** | Streaming chat with Gemma, optional image understanding | up to 90 s |
| 🎧 **Transcribe a file** | Upload/record → transcript → optional translation + spoken audio | up to 120 s |
| 📄 **Document OCR** | Image or PDF (first 3 pages) → structured text via Gemma vision | up to 120 s |

Free visitors get a daily ZeroGPU quota (sign in to Hugging Face for more). When the
quota runs out, the app tells you — or just run the unrestricted version locally from
the [GitHub repo](https://github.com/murai1998/LocalLLM).

## Architecture (demo vs. full)

| | This Space (ZeroGPU) | Full local app |
|---|---|---|
| LLM | Gemma 12B bf16 via transformers, per-call GPU | Gemma 12B **Q6_K GGUF** via llama.cpp, resident |
| STT | Whisper large-v3-turbo | Gemma native audio (unified) |
| TTS | Piper (CPU) | Piper (CPU/CUDA) |
| Voice mode | Per-utterance (pause-triggered) | **Continuous streaming**, pipelined, ≤ 8 s lag |
| UI | Gradio | React SPA + Streamlit |
| Privacy | Audio processed on HF infra | **Nothing leaves your machine** |

## Hosting / reproducing this Space

1. Hugging Face **PRO** account (ZeroGPU hosting requires PRO).
2. Accept the Gemma license on the model page.
3. Create a Space → SDK **Gradio** → Hardware **ZeroGPU** → add secret `HF_TOKEN`
   (read token, for the gated Gemma download).
4. `hf upload <user>/<space> ./showcase . --repo-type space`

## Running locally (rehearsal)

> Always install with `python -m pip …` so packages land in the same interpreter that
> runs the app (a bare `pip` can resolve to a different Python — notably, venvs created
> by `uv` contain no pip at all; run `python -m ensurepip --upgrade` once if needed).

**UI only — no GPU, no model downloads** (stub models echo back):

```powershell
# Windows PowerShell
python -m pip install "gradio==5.50.0" "fastrtc[vad]==0.0.34" piper-tts pymupdf numpy
$env:SHOWCASE_FAKE = "1"
python app.py            # from the showcase/ directory
```

```bash
# macOS / Linux
python -m pip install "gradio==5.50.0" "fastrtc[vad]==0.0.34" piper-tts pymupdf numpy
SHOWCASE_FAKE=1 python app.py
```

**Real models locally — llama.cpp backend (recommended, the default):**

This is the same proven stack as the full app: `llama-server` serving
`gemma-4-12b-it-Q6_K.gguf` + `mmproj-F16.gguf` (vision *and* audio — Gemma does
STT itself, no Whisper needed). **No torch / transformers / bitsandbytes at all.**

```powershell
# 1. llama.cpp: grab a release from https://github.com/ggml-org/llama.cpp/releases
#    and put llama-server on PATH (or set SHOWCASE_LLAMA_BIN).
# 2. Python deps — UI layer only:
python -m pip install "gradio==5.50.0" "fastrtc[vad]==0.0.34" piper-tts pymupdf pillow numpy
# 3. Model files: <repo>/models/gemma-4-12b-it-Q6_K.gguf + mmproj-F16.gguf are
#    picked up automatically (downloaded from unsloth/gemma-4-12b-it-GGUF if absent).
python app.py            # SHOWCASE_BACKEND defaults to "llama" off-Spaces
```

Useful overrides: `SHOWCASE_LLAMA_URL` (reuse an already-running llama-server, e.g.
the full app's), `SHOWCASE_LLAMA_PORT` (default 8094), `SHOWCASE_GGUF` / `SHOWCASE_MMPROJ`.

**Real models via transformers/bitsandbytes** (`SHOWCASE_BACKEND=transformers` —
mainly for rehearsing exactly what the ZeroGPU Space runs; needs a CUDA torch build
and an `HF_TOKEN` with the Gemma license accepted):

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt "gradio==5.50.0"
$env:HF_TOKEN = "hf_..."          # token of an account that accepted the Gemma license
$env:SHOWCASE_BACKEND = "transformers"
$env:SHOWCASE_PRESET = "small"    # gemma-4-12b in 4-bit (~10 GB VRAM w/ Whisper); "full" is bf16 (~25 GB)
python app.py
```

⚠️ Fair warning: bitsandbytes 4-bit decoding is slow (~3 tok/s on a consumer GPU —
a full OCR page takes minutes). It exists for Space parity, not for daily use;
the llama.cpp backend is ~10× faster at higher quant quality (Q6_K vs NF4).

Gemma 4 on transformers needs **transformers ≥ 5.11** and **torchvision** (the 4.x
line fails with `Unrecognized processing class`); the torch/torchvision pair must
come from the same CUDA index.

On the Space itself none of this applies — ZeroGPU provides CUDA torch automatically,
and the Space always uses the transformers backend (ZeroGPU timeslices CUDA for
PyTorch only, so llama.cpp cannot run there; it *can* run in a paid Docker GPU Space).

**Troubleshooting:**
- *"Transcribe a file" fails with `ffprobe not found`* → Gradio needs **ffmpeg** to decode
  non-WAV uploads (mp3/m4a/ogg). Locally: `winget install Gyan.FFmpeg` (Windows) /
  `brew install ffmpeg` (macOS), then restart the terminal that runs the app. On the
  Space this is handled by `packages.txt`.
- *Page renders but is completely static/unclickable* → the Python server behind it is
  no longer running (closed terminal, Ctrl+C, or crash after the page loaded). Check the
  terminal that ran `python app.py` is still alive, restart it, then hard-refresh the
  browser (Ctrl+F5). First start takes ~20–30 s (FastRTC VAD + Piper voice downloads)
  before the URL responds.
- *`ModuleNotFoundError` although you installed the package* → it went to a different
  interpreter; always use `python -m pip install …` in the same shell that runs the app.
