# Agent Handoff — LocalLLM / Gemma 4 12B

> **For the AI agent:** Read this file first when resuming work on this project in a new window or on another machine. It captures intent, decisions, architecture, and current status so you can continue without re-discovery.

**Last updated:** 2026-06-05  
**Project path:** `C:\Projects\LocalLLM` (may differ on other machines)  
**Version:** `localllm` 0.2.0 (service-client architecture — Phase 1+2 done)

---

## 1. Original user goal

Build a **skeleton local application suite** around **Gemma 4 12B** (`google/gemma-4-12B-it`) for use on:
- **macOS MacBook Pro** (Apple Silicon / Metal)
- **NVIDIA RTX 5060 16GB** (CUDA)

**Not** tuned for the original dev Windows machine; functional GPU paths for both targets.

### Requested components

| # | Component | v1 status |
|---|-----------|-----------|
| 1 | Terminal chat (load model + REPL) | **Done** — `localllm-chat` |
| 2 | Streamlit multimodal chat (images, docs, audio) | **Done** — `localllm-streamlit` |
| 3 | LangGraph agent wrapping the chatbox | **Done** — `localllm-agent` |
| 4a | OCR bot — structured text from images/PDF | **Done** — `localllm-ocr` |
| 4b | Batch STT — audio files → `.txt` | **Done** — `localllm-stt` |
| 4c | Real-time mic STT (chunked streaming) | **Explicitly skipped** for v1 |

---

## 2. Key decisions (do not undo without asking)

| Topic | Decision |
|-------|----------|
| Inference backend | **llama.cpp** via `llama-server` subprocess, **not** Hugging Face Transformers at runtime |
| Quantization | **Q6_K** GGUF (~6-bit) — `gemma-4-12b-it-Q6_K.gguf` from `unsloth/gemma-4-12b-it-GGUF` |
| Multimodal projector | `mmproj-F16.gguf` (same repo) |
| Python ↔ model | OpenAI-compatible HTTP API at `http://127.0.0.1:8080/v1/chat/completions` |
| MLX | **Skipped** |
| External APIs | **None** for inference/OCR — all local |
| PDF text | **PyMuPDF** (`fitz`) text extraction when text layer exists |
| PDF scanned / images | **Gemma native vision** via llama-server (pages rendered to PNG locally with PyMuPDF) |
| HF token | Read from `hf_token.txt` or `HF_TOKEN` env (`localllm/secrets.py`); file is **gitignored** |
| Thinking mode | **Disabled** by default (`enable_thinking: false`) — empty replies if enabled without enough tokens |
| Audio chunk limit | Gemma max **30s** per call; batch STT uses 28s chunks, 2s overlap |

---

## 3. Architecture (mental model)

```
apps/                    # CLI + Streamlit entrypoints + serve.py
  serve.py               # localllm-serve — FastAPI gateway
  cli_chat.py, streamlit_chat.py, agent_cli.py, ocr_bot.py, stt_batch.py

localllm/
  client/                # LLM client SDK (Phase 1)
    protocol.py          # LLMClient protocol
    openai_compatible.py # HTTP client for any OpenAI-compatible API
    factory.py           # create_llm_client()
    media.py             # image/audio/text content parts
  service/               # Gateway (Phase 2)
    app.py               # FastAPI: /v1/chat/completions, /health, /v1/models
    manager.py           # ServiceManager — subprocess gateway autostart
  config.py              # AppSettings + LLMConfig + ServiceConfig
  backends/
    llama_server.py      # llama-server subprocess (singleton, used by gateway)
  chat/engine.py         # ChatEngine → client SDK → gateway :8090
  agents/, pipelines/, media/, model/ …

config/default.yaml      # service :8090, llm client URL, llama_server :8080
config/profiles/         # openai.yaml example for commercial switch
```

### Inference flow (v0.2)

1. Client app → `ChatEngine` → `create_llm_client()` → `http://127.0.0.1:8090/v1/chat/completions`
2. If gateway not running: `ServiceManager.ensure_running()` spawns `uvicorn localllm.service.app:app`
3. Gateway lifespan → `LlamaServerManager.start()` → `llama-server` on `:8080`
4. Gateway proxies chat request to llama-server with concurrency semaphore
5. Modality order in client: **images before text**, **audio after text**

---

## 4. Commands cheat sheet

```bash
# Setup (once per machine)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
localllm-download                  # ~10 GB; needs HF token + Gemma license accepted

# Requires llama-server on PATH (CUDA or Metal build from ggml-org/llama.cpp releases)
localllm-serve                     # recommended: start gateway + inference once
localllm-chat
localllm-chat --no-server          # if gateway already running
localllm-streamlit
localllm-agent "your task"
localllm-ocr image.png -o out.json
localllm-stt recording.wav -o transcripts/

pytest -q                          # unit tests; no GPU/model required
```

---

## 5. Current status on last dev session

| Item | Status |
|------|--------|
| v1 apps | Complete |
| Phase 1 client SDK | **Done** — `localllm/client/` |
| Phase 2 gateway service | **Done** — `localllm-serve`, `localllm/service/` |
| Unit tests | **16 passed** (client, gateway, agent parser, media, secrets) |
| GGUF downloaded | **Yes** on dev machine — `models/gemma-4-12b-it-Q6_K.gguf` + `models/mmproj-F16.gguf` |
| llama-server integration tested end-to-end | **Confirmed** on RTX 5060 (chat works); gateway path needs smoke test |
| Streamlit | Fixed (launcher subprocess + no nested `launch()`) |
| Git repo | Initialized (has remote `origin/main`) |
| Real-time STT (4c) | Not implemented |

---

## 6. Known limitations & skeleton quality notes

- **llama-server binary** is an external dependency — Python only manages subprocess + HTTP.
- **Agent tool calling** uses heuristic JSON parsing from model text (`agents/graph.py`), not native Gemma function-calling wire format. Works as skeleton; may need hardening.
- **Streamlit** keeps upload temp files only for the request lifecycle; long sessions may need persistence tweaks.
- **OCR JSON** parsing has fence-stripping fallback; malformed JSON returns `parse_error` wrapper.
- **16GB VRAM**: Q6_K + mmproj should fit; if OOM, lower `n_gpu_layers` in `config/default.yaml`.
- **RTX 50-series**: needs recent CUDA drivers + CUDA-enabled llama.cpp build.
- **Scanned PDFs**: `render_pdf_pages()` at 150 DPI — quality/speed tradeoff not tuned.

---

## 7. Next steps (roadmap)

| Phase | Status | Task |
|-------|--------|------|
| 1 — Client SDK | **Done** | `LLMClient`, `create_llm_client()`, all apps refactored |
| 2 — Gateway | **Done** | `localllm-serve`, FastAPI proxy, `ServiceManager` autostart |
| 3 — Commercial profiles | **Next** | `LOCALLLM_PROFILE=openai`, profile YAML loader, LiteLLM optional |
| 4 — Streaming | Planned | SSE token streaming in CLI + Streamlit |
| 5 — Agent hardening | Planned | Native Gemma tool-call template |
| 6 — Real-time STT (4c) | Planned | Mic capture, chunked streaming |
| 7 — Integration tests | Planned | Gated on gateway + llama-server |
| 8 — Production deploy | Planned | systemd/launchd, Docker, LAN auth |

**Immediate:** smoke-test `localllm-serve` + `localllm-chat` on Mac Metal; then Phase 3 profiles.

---

## 8. Config reference (`config/default.yaml`)

```yaml
service.port: 8090                               # gateway (clients)
service.max_concurrent_requests: 2
llm.provider: local
llm.base_url: http://127.0.0.1:8090/v1
llm.model: gemma-4-12b-it
llama_server.port: 8080                          # inference (internal)
llama_server.n_gpu_layers: 999                   # full GPU offload
generation.enable_thinking: false
ocr.max_pages: 20
stt.chunk_seconds: 28
stt.max_chunk_seconds: 30                        # hard Gemma limit
```

Env overrides: prefix `LOCALLLM_`, nested `__` (see `.env.example`).

---

## 9. Files to read first when debugging

| Problem | Start here |
|---------|------------|
| Server won't start | `localllm/backends/llama_server.py`, `localllm/devices/resolver.py` |
| Chat/multimodal | `localllm/backends/openai_client.py`, `localllm/chat/engine.py` |
| OCR pipeline | `localllm/pipelines/ocr.py`, `localllm/media/pdf.py` |
| STT pipeline | `localllm/pipelines/stt_batch.py`, `localllm/media/audio.py` |
| Agent | `localllm/agents/graph.py`, `localllm/agents/tools.py` |
| Model download | `localllm/model/download.py`, `scripts/download_model.py` |
| User-facing docs | `README.md` |

---

## 10. How to resume a session (user instruction)

When starting a new chat, user should say something like:

> Read `AGENT_HANDOFF.md` and continue the LocalLLM project.

Agent should read this file + `config/default.yaml` + relevant module before making changes.

---

## 11. Conversation arc (brief)

1. User asked for a **plan** for Gemma 4 12B skeleton (terminal, Streamlit, LangGraph, OCR, STT, real-time STT).
2. Plan presented; user approved with modifications: **llama.cpp Q6_K**, **pymupdf for PDF text**, **native vision OCR**, **skip real-time STT**, **skip MLX**, use existing **hf_token.txt**.
3. Full v1 implemented in one session; tests green; README written.
4. User asked for this handoff file for cross-machine continuity.

**Do not re-plan from scratch unless user asks for architectural changes.**