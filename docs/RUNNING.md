# LocalLLM — Gemma 4 12B (llama.cpp Q6_K)

Local apps around **Gemma 4 12B** using **llama.cpp** with a **Q6_K** GGUF (~6-bit) from [unsloth/gemma-4-12b-it-GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF).

**Architecture (v0.2):** a single **LLM gateway service** (`localllm-serve`) owns one `llama-server` instance. All apps (chat, Streamlit, agent, OCR, STT) are **clients** that talk to the gateway over an OpenAI-compatible HTTP API. The same client SDK can be pointed at commercial APIs (OpenAI, etc.) via config.

## What's included

| Component | Command | Description |
|-----------|---------|-------------|
| **LLM gateway** | `localllm-serve` | FastAPI service — single shared inference entry point |
| Terminal chat | `localllm-chat` | REPL with `--image` / `--audio` |
| Streamlit UI | `localllm-streamlit` | Chat + file uploads |
| LangGraph agent | `localllm-agent` | Tool-using agent; `--multi` for a parallel researcher → analyst → critic team |
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
  - **Metal build** for macOS Apple Silicon (MacBook Pro) — `brew install llama.cpp`
  - Must be **build ≥ 9590** (older builds reject the Gemma 4 multimodal projector
    with `unknown projector type: gemma4uv`); on macOS run `brew upgrade llama.cpp`
- Hugging Face access to Gemma 4 (token in `hf_token.txt` or `HF_TOKEN` env)
- ~10 GB disk for `gemma-4-12b-it-Q6_K.gguf` + mmproj (bigger presets: see [Bigger local models](#bigger-local-models-model-presets))

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
localllm-agent --multi "Compare two approaches to X and recommend one"
```

Add `--multi` for **multi-agent mode**: a team of role specialists works the
task as a hybrid graph — two researchers gather facts **in parallel** (each a
full tool-using agent with a different lens), an analyst synthesises their
findings, and a critic red-teams that draft and produces the final answer. In
the Web UI, the same mode is the "Multi-agent team" toggle in the agent panel.
Concurrency is bounded by `service.max_concurrent_requests` (default 2), so with
two researchers raise it if you want them fully overlapped.

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

### Bigger local models (model presets)

The default is Gemma 4 12B, but the stack can serve larger models that fit a 48 GB Apple Silicon Mac. Presets live in `localllm/model/presets.py`:

| Preset id | Repo | Default quant (size) | Other quants | Audio input |
|-----------|------|----------------------|--------------|-------------|
| `gemma-4-12b` (default) | `unsloth/gemma-4-12b-it-GGUF` | `q6_k` (9.8 GB) | `q5_k` (8.4 GB) | yes |
| `gemma-4-31b` | `google/gemma-4-31B-it-qat-q4_0-gguf` | `qat_q4_0` (17.7 GB + 1.2 GB mmproj) — Google's quantization-aware-trained 4-bit, near-BF16 quality | `q5_k` 21.7 GB, `q6_k` 25.2 GB (both from `unsloth/gemma-4-31B-it-GGUF`; `q6_k` needs 64 GB or no audio sidecar) | no |
| `qwen3.8-27b` | `unsloth/Qwen3.8-27B-GGUF` | `ud_q5_k_xl` (20.9 GB + 0.9 GB mmproj) | `ud_q4_k_xl` 17.6 GB, `ud_q6_k` 22.0 GB | no |

Sizes are the GGUF weights; budget a few GB more for the KV cache and compute buffers.

**Download** (files land in `models/<preset>/`; the 12B stays flat in `models/` for backward compatibility):

```bash
localllm-download --model gemma-4-31b --with-audio-sidecar
```

Add `--quant q5_k` (or any quant from the table) to pick a non-default file.

**Select a model** — any one of these:

```bash
localllm-serve --model gemma-4-31b
```

```bash
localllm-serve --model gemma-4-31b --quant q5_k
```

```bash
LOCALLLM_PROFILE=qwen3.8-27b localllm-serve
```

`LOCALLLM_PROFILE` loads any `config/profiles/<name>.yaml` overlay; the new ones are `gemma-4-31b.yaml` and `qwen3.8-27b.yaml`. You can also set `model.name` in `config/default.yaml`, or use the env var `LOCALLLM_MODEL__NAME`. The web UI reads the active model from the gateway, so it needs no flag.

**macOS:** install llama.cpp with Homebrew (verified with 0.6.0; older builds may not know the Qwen3.8 architecture).

```bash
brew install llama.cpp
```

#### Audio sidecar

Gemma 4 31B and Qwen3.8-27B accept text and images only. When the main model lacks audio, an **audio sidecar** starts a second `llama-server` with Gemma 4 12B `q5_k` on port 8081. The gateway routes any `/v1/chat/completions` request containing `input_audio` parts to the sidecar, on its own slot pool, so STT, batch translate and live voice translation work unchanged with the big models.

```yaml
audio_sidecar:
  mode: auto          # auto (only when the main model has no audio) | on | off
  name: gemma-4-12b
  quantization: q5_k
  port: 8081
```

Routing is per request: a chat or agent turn that carries an audio attachment is answered entirely by the 12B sidecar, and so is `translate.pipeline: unified`. Keep the default `split` pipeline when comparing models, so the big model does the translating and the 12B only transcribes.

`/health` on the gateway reports `model`, `model_label`, `quantization` and `audio` (`{model, ready}`). `llama-server` output goes to `logs/llama-server-primary.log` and `logs/llama-server-audio.log`.

#### Measured on a 48 GB M5 Pro

| Model | Generation speed |
|-------|------------------|
| Gemma 4 31B QAT (`qat_q4_0`) | ≈ 14.6 tok/s |
| Qwen3.8-27B `ud_q5_k_xl` | ≈ 13 tok/s |

With the audio sidecar loaded, total wired memory is ≈ 35–36 GB. Live voice-to-voice on the 31B ran 4.8 s behind the speaker on an 8 s test clip (target ≤ 8 s).

#### Memory guard

`localllm-download` and `localllm-serve` estimate what the chosen model needs (weights + vision projector + KV cache, plus the audio sidecar when it runs) and refuse when that exceeds what this machine can give llama-server:

| Platform | Budget |
|----------|--------|
| Apple Silicon | `iogpu.wired_limit_mb` if you raised it, else ~75% of unified memory |
| NVIDIA | total VRAM (or ~80% of RAM with `n_gpu_layers: 0`) |
| CPU only | ~80% of RAM |

The error names the best quantization that would fit. On a 48 GB Mac (~38.7 GB budget): Gemma 4 31B `qat_q4_0` needs ~33 GB and `q5_k` ~37 GB, so both pass; `q6_k` needs ~40.5 GB and is refused. Qwen `ud_q5_k_xl` needs ~34 GB. On a 16 GB Mac both big models are refused. To override, pass `--force` to either command, or set `LOCALLLM_MODEL__SKIP_MEMORY_CHECK=true`.

#### Memory tips

- Close heavy apps before loading a 20 GB+ model.
- Pick a smaller quant (e.g. Qwen `ud_q4_k_xl`).
- Advanced, optional: macOS caps GPU-wired memory at about 75% of RAM by default. You can raise the cap until the next reboot — your call, and leave ~8 GB for macOS:

```bash
sudo sysctl iogpu.wired_limit_mb=40960
```

#### Thinking

Qwen3.8 thinks by default. The project's `generation.enable_thinking` setting is passed through as `chat_template_kwargs`, the same as for Gemma, so the config value controls both.

### Gemini 3.5 Live Translate (optional, remote)

For live translation you can compare the local pipeline against Google's `gemini-3.5-live-translate-preview` (Gemini Live API, public preview). It is streaming speech-to-speech, 70+ languages, auto-detects the source language and keeps the speaker's voice characteristics. It has no source-language, tone or voice options.

> **Privacy:** this engine is remote. Audio leaves the machine and is sent to Google. It is off unless you configure a key and pick it explicitly; the local engine stays the default.

Install the optional dependency:

```bash
pip install -e ".[gemini]"
```

#### Getting an API key

1. Open <https://aistudio.google.com/apikey> and sign in with your Google account.
2. Click **Create API key**. Accept the Terms the first time; a default Google Cloud project is created. Existing Cloud users can import a project via Dashboard → Projects → Import projects.
3. Recommended: restrict the key ("Restrict to Gemini API only").
4. The free tier works for this model (rate-limited, $0). For the paid tier, open <https://aistudio.google.com/projects> (or the API keys page), find your project's **Billing Tier** column, click **Set up billing**, and choose or create a billing account (may require a $5 prepay). Paid-tier prompts are not used to improve Google products. Set a budget / billing alert in the Google Cloud console.
5. Store the key, either in your shell profile (then open a new terminal):

   ```bash
   echo 'export GEMINI_API_KEY="your-key-here"' >> ~/.zshrc
   ```

   or as a line `GEMINI_API_KEY=your-key-here` in `.env.local` at the repo root (gitignored). Never commit the key and never paste it into chats or issues. If it leaks, create a new key, switch to it, then disable the old one.
6. Restart `localllm-webui`.

#### Pricing (paid tier)

From [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing): input audio $0.0053/min, output audio $0.0315/min (25 tokens per second of audio). The web UI and the bench print an estimated paid-tier cost per session.

#### Using it

Web UI → **Translate → Live** → **Engine** toggle: "Local · `<model>`" or "Gemini 3.5 Live Translate" (greyed out with a hint until the key is set). Transcript/translation rows and audio playback are the same for both engines.

Under the hood, `localllm/live/gemini.py` (`GeminiLiveTranslateSession`) is selected by `engine: "gemini"` in the `/ws/translate` start message. Config section:

```yaml
gemini:
  live_translate_model: gemini-3.5-live-translate-preview
  echo_target_language: false
```

#### Side-by-side benchmark

Run both engines on the same recording:

```bash
localllm-live-bench talk.wav --target es --out local.json
```

```bash
localllm-live-bench talk.wav --target es --engine gemini --out gemini.json
```

Both report engine-neutral metrics: time to first translated audio, tail after the speaker stopped, seconds of translated audio produced and, for Gemini, estimated cost. `--out` saves per-row transcripts and translations for a quality read. Gemini streams continuously, so its per-row "lag" is time since that row opened — compare the engine-neutral metrics instead.

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
| `unknown projector type: gemma4uv` | llama-server too old for the Gemma 4 mmproj — update to build ≥ 9590 (`brew upgrade llama.cpp` on macOS) |
| Web UI returns 503 / "bundle not found" | `webui/dist` missing — `cd webui && npm install && npm run build` |
| Stale process holding a stack port | `./scripts/stop_stale_ports.sh` (macOS/Linux) or `.\scripts\stop_stale_ports.ps1` (Windows) |
| Startup stops with an error about a port held by another model | A `llama-server` from a previous run (different model) still holds the port — run `./scripts/stop_stale_ports.sh`, then start again |
| Qwen3.8 won't load | Older llama.cpp builds may not know the Qwen3.8 architecture — `brew upgrade llama.cpp` (verified with 0.6.0) |
| Need to see why a model or the audio sidecar failed | Check `logs/llama-server-primary.log` and `logs/llama-server-audio.log` |
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