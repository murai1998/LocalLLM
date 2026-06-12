# Showcase Plan — LocalLLM on Hugging Face Spaces (ZeroGPU)

> **Status (2026-06-11): Plan A implemented (milestone S0 complete).** The
> self-contained distribution lives in [`showcase/`](showcase/) — Gradio app with all
> four tabs, FastRTC live interpreter, vendored prompt/voice modules verified in sync by
> `scripts/build_showcase.py` and 11 tests, quota-friendly error UX, and a prominent
> "reduced-capability demo → full app on GitHub" notice in both the Space README and the
> app banner. The full Gradio UI was smoke-built locally (models stubbed; FastRTC VAD
> warmed up). **Remaining (S1/S2, needs your HF account):** subscribe to PRO, accept the
> Gemma license, create the ZeroGPU Space, add the `HF_TOKEN` secret, and run
> `python scripts/build_showcase.py && hf upload <user>/<space> ./showcase . --repo-type space`
> — then test the interpreter tab on-Space and flip Public (§5 below).

**Date:** 2026-06-11 · **Goal:** a public, self-contained showcase of this project — the
**voice-to-voice interpreter** front and center, plus the other components where feasible —
hosted on Hugging Face Spaces under the tightest budget that works.

**Verdict up front: feasible on ZeroGPU**, with one architecture change (no resident
llama-server — per-call PyTorch inference instead) and one billing caveat: **a fully free
account can *use* ZeroGPU Spaces but cannot *host* one — hosting requires PRO at $9/month.**
A truly $0 hosting path exists (CPU-only Space, degraded models) and is included as Plan A0.

---

## 1. The rules of the game (verified 2026-06-11)

What ZeroGPU actually permits today — each of these shapes the design:

| Constraint | Value | Consequence for us |
|---|---|---|
| Who can host | **PRO ($9/mo)** personal accounts (max 10 ZeroGPU Spaces); Team/Enterprise orgs | The "free account" path is Plan A0 (CPU); ZeroGPU is Plan A (recommended, $9/mo) |
| Who can use | Everyone — anonymous 2 min GPU/day, free accounts 5 min/day, PRO 40 min/day | Budget ~5 s GPU per utterance → a free visitor gets ~60 spoken utterances/day. Fine for a showcase |
| SDK | **Gradio only** (4+) | The React web UI cannot ship; we build a Gradio Blocks app (as you anticipated) |
| GPU framework | **PyTorch only** (2.8 – latest); no `torch.compile` (AoT compile OK) | **llama.cpp / GGUF is out.** Inference moves to `transformers` |
| GPU residency | GPU exists **only inside `@spaces.GPU` functions** (default 60 s/call, extensible via `duration=`); models loaded to `cuda` at module level via CUDA emulation | **The gateway + llama-server client-server-server setup is not portable** — confirmed infeasible. Each pipeline stage becomes a decorated function call |
| Hardware | NVIDIA RTX Pro 6000 Blackwell — `large` = 48 GB (default), `xlarge` = 96 GB (2× quota) | Gemma 12B bf16 (~24 GB) fits `large` comfortably |
| Python | 3.10.13 or 3.12.12 | Our code is 3.10-clean (CI-tested) ✓ |
| Real-time audio | **FastRTC** (HF's WebRTC lib) works on ZeroGPU + Gradio; Cloudflare TURN relay is free for 10 GB/mo with any HF token; `ReplyOnPause` does VAD/turn-taking | The live interpreter is genuinely portable — see §3 |
| Free CPU tier | CPU Basic: 2 vCPU / 16 GB RAM, $0, any account | Plan A0 baseline |

---

## 2. Architecture mapping — local stack → Space stack

| Local (this repo) | Space (showcase) | Why |
|---|---|---|
| `llama-server` + FastAPI gateway (:8080/:8090) | **In-process `transformers`** pipelines, `@spaces.GPU` per call | No persistent GPU process on ZeroGPU |
| Gemma 12B **Q6_K GGUF** | **`google/gemma-4-12b-it` bf16** via transformers (~24 GB on the 48 GB slice) | GGUF/llama.cpp unsupported; bf16 fits easily |
| STT: Gemma unified audio | **`openai/whisper-large-v3-turbo`** (~1.6 GB) for STT; Gemma for MT only | Whisper turbo transcribes an 8 s segment in well under 1 s on this GPU — burns far less visitor quota than Gemma-audio; Gemma-audio kept as a config toggle for parity |
| MT: `translate_text` + tone presets + rolling context | **Reused as-is** (prompts are model-agnostic strings) | Direct port |
| TTS: Piper (ONNX) | **Piper, unchanged, on CPU** — no GPU call at all | onnxruntime CPU is fast enough (~0.5–2 s/sentence on 8 vCPU); spends zero visitor quota |
| Mic → WebSocket → endpointer → pipeline | **FastRTC `ReplyOnPause`** (built-in silero-style pause detection replaces our endpointer) → per-utterance GPU call → streamed audio reply | This is exactly the B-2/B-3 design, with HF's own infra doing capture + VAD |
| React SPA | **Gradio Blocks**, dark theme, tabs | SDK requirement |
| `localllm-live-bench` | not shipped (dev tool) | — |

Fallback if FastRTC misbehaves on ZeroGPU: `gr.Audio(sources=["microphone"], streaming=True)`
input + our `localllm/live/endpointer.py` (it's dependency-light NumPy — ports verbatim) +
streamed audio output. Slightly clunkier UX, zero WebRTC moving parts.

---

## 3. What the Space contains (Plan A — ZeroGPU, recommended)

One Gradio Blocks app, four tabs, ordered by wow-factor:

### Tab 1 — 🎙️ Live Interpreter (the headline)
- FastRTC `Stream` + `ReplyOnPause`: visitor speaks, pause detection fires, the utterance
  goes through one `@spaces.GPU(duration=dynamic)` call doing Whisper STT + Gemma MT
  (rolling 2-segment context, tone selector reused from `pipelines/translate.py`), then
  Piper TTS on CPU streams the translated voice back over the same WebRTC connection.
- Per-utterance GPU cost ≈ 3–6 s → the visitor lag story from plan.md holds (≤ 8 s).
- Controls: source (auto) / target language, tone, voice — same vocabulary as the local app.
- Bilingual rolling transcript below the audio widget (mirrors the web UI's Live view).

### Tab 2 — 💬 Chat
- Gemma 12B chat with **token streaming** (`TextIteratorStreamer`), `@spaces.GPU(duration=60)`.
- Optional image attachment (Gemma vision) — gives the multimodal story in one tab.

### Tab 3 — 🎧 Transcribe & Translate a file
- Upload/record → Whisper STT → optional Gemma translation + Piper voice-over.
- This is the batch path of the Translate page, nearly free to port.

### Tab 4 — 📄 Document OCR
- Image or PDF page → Gemma vision OCR (reuses `OCR_SYSTEM` prompt), structured text out.
- PDFs rendered with PyMuPDF exactly as locally; cap at 3 pages to respect quota.

**Quota UX:** wrap GPU calls to catch the `spaces` quota-exceeded error and show
"Daily free GPU minutes used up — sign in / try tomorrow" instead of a stack trace.

---

## 4. Self-contained distribution

A `showcase/` directory in this repo that is **complete on its own** — no imports from the
`localllm` package (small modules are vendored so the Space repo can be a straight copy):

```
showcase/
  README.md            # Space card — YAML front-matter is the Space config (see below)
  app.py               # Gradio Blocks shell, tabs, theme
  interpreter.py       # FastRTC stream + @spaces.GPU STT/MT call + Piper reply
  chat.py              # streaming chat tab
  batch.py             # transcribe/translate-a-file tab
  ocr.py               # document OCR tab
  prompts.py           # vendored: tone presets, ASR/OCR/translate prompts
  piper_voices.py      # vendored: VOICE_OPTIONS + synthesize (CPU)
  endpointer.py        # vendored fallback (only used in non-WebRTC mode)
  requirements.txt     # pinned: torch, transformers, gradio, spaces, fastrtc, piper-tts, pymupdf…
  packaging: scripts/build_showcase.py  # copies/refreshes vendored files from localllm/ and runs a smoke import
```

`README.md` front-matter (this **is** the Space configuration):

```yaml
---
title: LocalLLM Voice Interpreter
emoji: 🎙️
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: "5.x"        # pin to the tested version
python_version: "3.12"
app_file: app.py
license: apache-2.0
models:
  - google/gemma-4-12b-it
  - openai/whisper-large-v3-turbo
---
```

Key implementation notes baked into the distribution:
- Models loaded at **module level on `cuda`** (ZeroGPU's emulation requires this pattern).
- `HF_TOKEN` read from Space **secrets** (Gemma is gated — the hosting account must accept
  the license once on the model page).
- Everything testable locally: `@spaces.GPU` is a no-op off-Space, so
  `cd showcase && pip install -r requirements.txt && python app.py` runs on the RTX 5060
  (12B bf16 won't fit 16 GB — a `SHOWCASE_PRESET=small` env switches to `gemma-4-4b-it`
  for local rehearsal and for the CPU plan below).

---

## 5. Publishing — step by step

1. **Account prep** (one-time)
   - Create/log into huggingface.co; subscribe to **PRO** ($9/mo) — required to select
     ZeroGPU hardware.
   - On `google/gemma-4-12b-it`: accept the license terms.
   - Create a **write token**: Settings → Access Tokens → "New token" (fine-grained, write).
2. **Create the Space**: huggingface.co/new-space → name `localllm-interpreter` →
   License Apache-2.0 → **SDK: Gradio** → **Hardware: ZeroGPU** → Private (flip Public after testing).
3. **Add the secret**: Space → Settings → Variables and secrets → `HF_TOKEN` = a **read**
   token (used at runtime to pull the gated Gemma weights).
4. **Upload** (either way works):
   ```bash
   pip install -U huggingface_hub
   hf auth login                                    # paste the write token
   hf upload <user>/localllm-interpreter ./showcase . --repo-type space
   ```
   or git: `git clone https://huggingface.co/spaces/<user>/localllm-interpreter`,
   copy `showcase/*` in, commit, push.
5. **First boot**: watch Logs tab — first build downloads ~26 GB of weights into the Space
   cache (one-time; subsequent restarts are warm). Open the app, run each tab once
   yourself (PRO quota: 40 min/day).
6. **Go public**, pin a demo video/GIF in the Space card, and link it from the GitHub README.
7. **Maintenance**: Dev Mode (PRO feature) gives SSH/live-reload into the running Space for
   debugging; `hf upload` again to redeploy.

---

## 6. Plan A0 — strictly free account ($0)

If the $9/mo is not wanted, a **CPU Basic** Space (2 vCPU / 16 GB, free, any account) can run a
degraded but honest demo from the same codebase (`SHOWCASE_PRESET=cpu`):

| Stage | Model | Expected latency (2 vCPU) |
|---|---|---|
| STT | `faster-whisper` small **int8** (CTranslate2, CPU) | ~2–5 s per 8 s utterance |
| MT | `NLLB-200-distilled-600M` int8 (CTranslate2) — or Gemma 4B Q4 via llama-cpp CPU (allowed off-ZeroGPU) | ~1–4 s |
| TTS | Piper (already CPU) | ~1–2 s |

Push-to-talk rather than continuous (CPU can't keep up with overlap), total ~5–10 s per
utterance. Not the holy grail, but a working, free, zero-cost interpreter demo — and a good
staging target before paying anything. The chat tab would use the 4B/NLLB models; OCR is too
slow on CPU and gets dropped.

---

## 7. Plan C — paid dedicated hardware (if ZeroGPU's per-call model ever chafes)

Dedicated Space GPUs run a **persistent process** (llama.cpp/GGUF allowed again, Docker SDK
allowed, no per-call quota for visitors). Billed per hour while running; auto-sleep on
inactivity is configurable. Current prices (verified on the pricing page; re-check before
committing):

| Hardware | VRAM | $/hour | Always-on /mo | Sleep-after-15-min, ~3 h active/day |
|---|---|---|---|---|
| T4 small | 16 GB | $0.40 | ~$290 | ~$36 (needs Q4 quant or 4B model) |
| A10G small | 24 GB | $1.00 | ~$720 | ~$90 |
| L4 | 24 GB | $0.80 | ~$575 | ~$72 (Q6_K 12B GGUF fits — closest to local parity) |
| L40S | 48 GB | $1.80 | ~$1,300 | ~$162 (bf16 12B, roomy) |
| A100 | 80 GB | $2.50 | ~$1,800 | ~$225 (overkill) |

Plus PRO-free alternatives that are *not* Spaces: Inference Endpoints (model API without the
demo UI — wrong tool for a showcase) and ZeroGPU credit top-ups ($1 per 10 GPU-minutes past
the daily quota — useful for a demo *event* day on Plan A).

**Recommendation: Plan A (ZeroGPU + PRO, $9/mo flat).** It is purpose-built for exactly this
kind of showcase; Plan C's cheapest sensible option (L4 with aggressive sleep) costs ~8× more
and adds cold-boot delays for visitors anyway. Choose Plan C only if you must demonstrate the
*actual* llama.cpp/GGUF + gateway architecture rather than the experience.

---

## 8. Risks & validation checklist

| Risk | Mitigation / check |
|---|---|
| FastRTC × ZeroGPU integration quirks | Known-good combination per HF docs/blog, but validate in a **private** Space first; fallback to `gr.Audio(streaming=True)` + vendored endpointer is coded in from day one |
| Cloudflare TURN free tier (10 GB/mo) | Mono 16 kHz Opus ≈ 12 MB/hour/direction → 10 GB ≈ hundreds of demo hours; monitor in Cloudflare dash; can switch to `gr.Audio` mode if ever exhausted |
| Gemma gated license | Hosting account accepts once; `HF_TOKEN` secret for runtime pulls |
| Cold start (~26 GB weights) | One-time per build; keep `sdk_version`/requirements stable to avoid rebuilds; consider `gemma-4-4b-it` preset if first-token latency annoys |
| Visitor quota exhaustion mid-demo | Catch quota errors → friendly banner; demo-day option: $1/10 min credits on the *visiting* account |
| ZeroGPU concurrency queues at peak | Short `duration=` hints improve queue priority; keep GPU calls lean (STT+MT only, TTS on CPU) |
| Local rehearsal can't fit 12B bf16 on RTX 5060 | `SHOWCASE_PRESET=small` (4B) for local runs; the Space runs the 12B preset |

---

## 9. Milestones

| # | Deliverable | Est. effort |
|---|---|---|
| S0 | `showcase/` skeleton + vendoring script; all tabs working locally with `SHOWCASE_PRESET=small`; `@spaces.GPU` no-op path verified | 1–2 days |
| S1 | Private ZeroGPU Space live: interpreter tab end-to-end via FastRTC, quota-error UX, logs clean | 1–2 days (incl. Space iteration loop) |
| S2 | Remaining tabs ported, dark theme polish, Space card (GIF, description), flip Public | 1 day |
| S3 | (optional) Plan A0 CPU variant published from the same tree | 0.5–1 day |

**Definition of done:** a stranger with no HF account opens the Space, speaks two sentences,
and hears the translation in another language within ~8 s each — on hardware that costs you
$9/month.

---

Sources: [ZeroGPU docs](https://huggingface.co/docs/hub/spaces-zerogpu) ·
[HF pricing](https://huggingface.co/pricing) ·
[FastRTC](https://fastrtc.org/) ·
[FastRTC × Cloudflare announcement](https://huggingface.co/blog/fastrtc-cloudflare)
