# LocalLLM — Private, Self-Hosted AI Platform

> A full-featured, **completely free** AI assistant stack that runs a state-of-the-art
> 12B-parameter LLM **entirely on your own hardware** — an NVIDIA GPU workstation or an
> Apple Silicon MacBook Pro. No API keys, no per-token billing, no data ever leaving the
> machine.

## Elevator pitch

LocalLLM packages the kind of capabilities companies normally pay OpenAI or Anthropic for —
multimodal chat, document OCR, speech-to-text, real-time speech translation with spoken
output, and a tool-using autonomous agent — into a single self-hosted platform built around
**Gemma 4 12B** served via **llama.cpp**. Because everything runs locally, it delivers
**zero marginal cost, full data privacy, and offline operation**, while keeping a one-line
config switch to fall back to commercial APIs (OpenAI/Azure/Anthropic) when desired.

For an organization, that combination — **enterprise-grade AI features with no recurring cost
and no data-exfiltration risk** — is increasingly the deciding factor in whether AI can be
adopted at all.

## Capabilities

| Feature | What it does |
|---------|--------------|
| **Multimodal chat** | Text + image + audio understanding through Gemma's vision/audio paths |
| **Tool-using agent** | LangGraph ReAct-style agent with sandboxed file tools, web search & page fetch, and a pluggable skills system |
| **Document OCR** | Native text extraction for digital PDFs (PyMuPDF) + local vision OCR for scans/images |
| **Speech-to-text** | Batch transcription with overlap-aware chunking and local merge |
| **Real-time speech translation** | VAD-segmented audio → ASR → translation → **offline text-to-speech**, streamed incrementally |
| **OpenAI-compatible gateway** | One shared inference endpoint that every app (and external client) can call |
| **Provider abstraction** | Same client SDK targets the local model or any OpenAI-compatible cloud API via config |

## Architecture decisions

- **Single-gateway design.** One FastAPI service (`localllm-serve`) owns the only
  `llama-server` subprocess and exposes an **OpenAI-compatible HTTP API**. All apps are thin
  clients. This prevents multiple processes from each loading a 10 GB model, centralizes
  concurrency control (async semaphore queue), and makes the whole system swappable with a
  commercial API by changing one URL.
- **Provider-agnostic client SDK.** A small protocol-based client layer means business logic
  never hard-codes "local vs. cloud" — the same code runs on Gemma locally or GPT-4o in the
  cloud, selected purely by configuration.
- **Cross-platform GPU acceleration.** Auto-detects **CUDA** (NVIDIA) vs. **Metal** (Apple
  Silicon) and offloads all layers to the GPU by default, with VRAM-tuning profiles
  (e.g. 5-bit `Q5_K` quantization) for constrained cards.
- **Pipeline composition over monoliths.** Speech translation is built from small, testable
  stages (VAD chunking → ASR → translate → sentence-queued TTS) that stream partial results,
  rather than one opaque call.
- **Privacy & safety by default.** Gateway binds to `127.0.0.1`; agent file tools are
  sandboxed to the project directory; the web-fetch tool blocks private/loopback IP ranges
  (SSRF protection); secrets are kept out of version control.
- **Layered, typed configuration.** Pydantic-settings with YAML defaults, named profiles, and
  `LOCALLLM_`-prefixed environment overrides for reproducible, environment-specific deploys.

## Technology stack

**Language & core**
- **Python 3.10+**, fully type-annotated, packaged as an installable distribution with console
  entry points (`pyproject.toml` / setuptools)

**LLM & inference**
- **Gemma 4 12B** (Google) as a **GGUF** model running on **llama.cpp / `llama-server`**
- **Quantization** (Q6_K / Q5_K) for GPU-memory efficiency
- **Hugging Face Hub** for model distribution and gated-model auth

**Backend & APIs**
- **FastAPI** + **Uvicorn** OpenAI-compatible gateway with async concurrency queue
- **httpx** async HTTP client; **Pydantic v2** / **pydantic-settings** for schema & config

**Agents & orchestration**
- **LangGraph** + **LangChain Core** for a stateful, tool-calling agent graph with a custom
  tool-call parser and a discoverable "skills" system

**Multimodal & media**
- **PyMuPDF** (PDF), **python-docx**, **Pillow** (vision), **librosa** / **soundfile** /
  **imageio-ffmpeg** (audio), custom **energy-based VAD** with **NumPy**
- **Piper TTS** for fully offline neural speech synthesis

**Frontend**
- **Streamlit** apps for chat and a split-screen live-translation UI

**Quality & DevEx**
- **pytest** unit suite that runs with **no GPU or model required** (clients, gateway, agent
  parser, media, pipelines all mocked/unit-tested)
- Cross-platform support (Windows / Linux / macOS) with automatic platform detection

## Highlighted high-demand skills demonstrated

`LLM serving & local inference (llama.cpp/GGUF)` · `model quantization` ·
`OpenAI-compatible API design` · `FastAPI microservice / gateway architecture` ·
`LangGraph agentic workflows & tool use` · `Retrieval & web-research tooling` ·
`multimodal AI (vision, audio, OCR)` · `speech-to-text & real-time translation pipelines` ·
`offline TTS` · `async Python & concurrency control` · `Pydantic typed configuration` ·
`GPU acceleration on CUDA & Apple Metal` · `provider-abstraction for cloud/local portability` ·
`security hardening (sandboxing, SSRF defense)` · `Python packaging & testing`

## Why it matters

LocalLLM proves the ability to take a raw open-weights model and turn it into a **production-
shaped, multi-app AI product** — with the service architecture, provider abstraction,
multimodal pipelines, and safety controls of a commercial offering — at **zero ongoing cost
and with complete data sovereignty**. It is equally a demonstration of modern AI engineering
breadth (serving, agents, multimodal, RAG-style tooling) and of pragmatic software
architecture (gateway pattern, typed config, cross-platform GPU support, test discipline).
