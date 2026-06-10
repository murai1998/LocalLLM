# LocalLLM — Gemma 4 12B (llama.cpp Q6_K)

Local apps around **Gemma 4 12B** using **llama.cpp** with a **Q6_K** GGUF (~6-bit) from [unsloth/gemma-4-12b-it-GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF).

**Architecture (v0.2):** a single **LLM gateway service** (`localllm-serve`) owns one `llama-server` instance. All apps (chat, Streamlit, agent, OCR, STT) are **clients** that talk to the gateway over an OpenAI-compatible HTTP API. The same client SDK can be pointed at commercial APIs (OpenAI, etc.) via config.

## What's included

| Component | Command | Description |
|-----------|---------|-------------|
| **LLM gateway** | `localllm-serve` | FastAPI service — single shared inference entry point |
| Terminal chat | `localllm-chat` | REPL with `--image` / `--audio` |
| Streamlit UI | `localllm-streamlit` | Chat + file uploads |
| LangGraph agent | `localllm-agent` | Tool-using agent (read/list/write note) |
| OCR bot | `localllm-ocr` | PyMuPDF text for PDFs; Gemma vision for images / scanned pages |
| Batch STT | `localllm-stt` | Transcribe audio files to `.txt` |
| Model download | `localllm-download` | Fetch GGUF + mmproj from Hugging Face |

Real-time microphone STT is **not** implemented (deferred).

## Architecture

```
┌─────────────┐  ┌──────────────┐  ┌────────────┐  ┌─────────┐  ┌─────────┐
│ localllm-   │  │ localllm-    │  │ localllm-  │  │ OCR /   │  │ future  │
│ chat        │  │ streamlit    │  │ agent      │  │ STT     │  │ clients │
└──────┬──────┘  └──────┬───────┘  └─────┬──────┘  └────┬────┘  └────┬────┘
       │                │                │              │            │
       └────────────────┴────────────────┴──────────────┴────────────┘
                                        │
                          OpenAI-compatible HTTP (localllm/client)
                                        │
                          ┌─────────────▼─────────────┐
                          │  LLM Gateway :8090        │
                          │  localllm-serve (FastAPI) │
                          │  · /v1/chat/completions   │
                          │  · /health, /v1/models    │
                          │  · concurrency queue      │
                          └─────────────┬─────────────┘
                                        │
                          ┌─────────────▼─────────────┐
                          │  llama-server :8080       │
                          │  (single subprocess)      │
                          │  Gemma 4 Q6_K + mmproj    │
                          └───────────────────────────┘
```

**Commercial fallback:** point `llm.base_url` + `llm.provider` at OpenAI (or any OpenAI-compatible API). Clients unchanged.

## Requirements

- Python 3.10+
- [llama.cpp](https://github.com/ggml-org/llama.cpp/releases) with **`llama-server`** on your `PATH`
  - **CUDA build** for Windows / Linux with NVIDIA GPU (e.g. RTX 5060 16GB)
  - **Metal build** for macOS Apple Silicon (MacBook Pro)
- Hugging Face access to Gemma 4 (token in `hf_token.txt` or `HF_TOKEN` env)
- ~10 GB disk for `gemma-4-12b-it-Q6_K.gguf` + mmproj

## Setup

```bash
cd LocalLLM
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS
source .venv/bin/activate

pip install -e ".[dev]"
localllm-download
```

Accept the Gemma license on Hugging Face for `google/gemma-4-12B-it` if prompted.

### macOS (MacBook Pro / Apple Silicon)

Install a **Metal** llama.cpp build:

```bash
export PATH="/path/to/llama.cpp/build/bin:$PATH"
localllm-serve    # terminal 1 — starts gateway + llama-server
localllm-chat     # terminal 2
```

Platform is auto-detected (`metal`). Default `-ngl 999` offloads all layers to GPU.

### Windows / Linux (NVIDIA CUDA)

Install a **CUDA** llama.cpp build. Default config uses `-ngl 999` (full GPU offload).

```powershell
localllm-serve    # terminal 1
localllm-chat     # terminal 2
```

If you hit CUDA OOM on 16 GB VRAM, lower `llama_server.n_gpu_layers` in `config/default.yaml`.

## Usage

### Recommended: run the gateway once

```bash
localllm-serve
```

This starts:
1. `llama-server` on `http://127.0.0.1:8080` (inference, internal)
2. FastAPI gateway on `http://127.0.0.1:8090` (clients connect here)

Check status:

```bash
curl http://127.0.0.1:8090/health
```

**Auto-start:** if you skip `localllm-serve`, the first client (`localllm-chat`, Streamlit, etc.) will spawn the gateway subprocess automatically. For daily use, keeping `localllm-serve` running in one terminal is cleaner.

Gateway only (llama-server already running separately):

```bash
localllm-serve --no-inference
```

### Chat (terminal)

```bash
localllm-chat
You> hello
You> --image photo.png what is in this image?
You> --audio clip.wav transcribe this
```

If the gateway is already running:

```bash
localllm-chat --no-server
```

### Streamlit

```bash
localllm-streamlit
```

Opens at `http://localhost:8501`. Attach files in the sidebar; images/audio use Gemma multimodal paths, PDFs through PyMuPDF text extraction.

### Agent

```bash
localllm-agent --verbose "List files in the project root"
localllm-agent   # interactive
```

### OCR

```bash
localllm-ocr scan.png -o out.json
localllm-ocr document.pdf -o out.json
```

- **PDF with text layer:** PyMuPDF extraction only (fast).
- **Scanned PDF / images:** local Gemma vision OCR.

### Batch STT

```bash
localllm-stt recording.wav
localllm-stt *.mp3 -o transcripts/
```

Audio is chunked (≤30s per Gemma limit) with overlap, then merged locally.

## Configuration

Edit `config/default.yaml`:

```yaml
model:
  quantization: q6_k            # q6_k (default) | q5_k (5-bit, ~lower VRAM)

service:
  port: 8090                    # gateway (clients connect here)
  max_concurrent_requests: 2    # queue for 12B model

llm:
  provider: local               # local | openai | anthropic | azure
  base_url: http://127.0.0.1:8090/v1
  model: gemma-4-12b-it

llama_server:
  port: 8080                    # internal inference (gateway proxies)
  n_gpu_layers: 999             # 0 = CPU only; 999 = full GPU (CUDA/Metal)
  context_size: 8192

generation:
  enable_thinking: false        # keep false for Gemma 4
```

### Quantization (5-bit option)

Default is **6-bit Q6_K**. For tighter VRAM (e.g. running Whisper on GPU alongside Gemma):

```bash
localllm-download --quant q5_k
```

Or in config / env:

```yaml
model:
  quantization: q5_k
```

```bash
export LOCALLLM_MODEL__QUANTIZATION=q5_k
```

Profiles: `config/profiles/q5_k.yaml`, `vram_dual_gpu.yaml`, `vram_llm_gpu.yaml`.

See **[docs/INTEGRATION_PLAN.md](docs/INTEGRATION_PLAN.md)** for combining with the Whisper STT service.

Environment overrides use prefix `LOCALLLM_` with nested `__` (see `.env.example`).

### Switch to a commercial API

Set env vars or edit `config/default.yaml`:

```yaml
llm:
  provider: openai
  base_url: https://api.openai.com/v1
  model: gpt-4o
```

```bash
export OPENAI_API_KEY=sk-...
localllm-chat   # uses OpenAI; no local gateway started
```

See `config/profiles/openai.yaml` for a template. Multimodal (image/audio) features require a model that supports them on the chosen provider.

## Project layout

```
localllm/
  client/          # LLM client SDK (OpenAI-compatible)
  service/         # FastAPI gateway + ServiceManager
  backends/        # llama-server subprocess manager
  chat/            # ChatEngine
  agents/          # LangGraph agent
  pipelines/       # OCR, STT
apps/              # CLI / Streamlit entrypoints
config/            # default.yaml, profiles/
models/            # Downloaded GGUF (gitignored)
```

## Security

- `hf_token.txt` is gitignored — do not commit tokens.
- Agent tools are sandboxed to the project directory.
- Gateway binds to `127.0.0.1` by default (local only).

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `llama-server not found` | Install llama.cpp; add `llama-server` to PATH (CUDA or Metal build) |
| Gateway timeout | First model load can take minutes; increase `service.startup_timeout_sec` |
| Empty assistant reply | Set `enable_thinking: false` (already default) |
| CUDA OOM on 16 GB | Lower `llama_server.n_gpu_layers` |
| Streamlit restart loop | Use `localllm-streamlit` (not `streamlit run` directly on the app file) |
| Port in use | Change `service.port` / `llama_server.port` in config |

## Tests

```bash
pytest -q
```

16 unit tests (client, gateway, agent parser, media). No GPU or model required. Integration tests against a live gateway are planned (see roadmap below).

## Roadmap — next steps

| Phase | Status | What |
|-------|--------|------|
| **1 — Client SDK** | Done | `localllm/client`, provider config, all apps refactored |
| **2 — Gateway service** | Done | `localllm-serve`, FastAPI proxy, concurrency queue |
| **3 — Whisper translator** | Next | STT + LLM cascade, split-screen Streamlit, VRAM profiles — see `docs/INTEGRATION_PLAN.md` |
| **4 — Commercial profiles** | Planned | Profile files (`local.yaml`, `openai.yaml`), `LOCALLLM_PROFILE` env, LiteLLM optional proxy |
| **5 — Streaming** | Planned | Token streaming in CLI + Streamlit via SSE |
| **6 — Agent hardening** | Planned | Native Gemma tool-call template, structured output |
| **7 — Integration tests** | Planned | Gated on `llama-server` + gateway availability |
| **8 — Production deploy** | Planned | systemd / launchd service files, Docker Compose, API key auth for LAN |

**Suggested immediate next PR:** Phase 3 — config profiles so switching `local` ↔ `openai` is one env var, no file edits.