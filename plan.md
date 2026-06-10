# LocalLLM — Completion Plan

**Date:** 2026-06-10 · **Baseline:** v0.2.0, 72/72 unit tests passing.
**Vision:** a dense, capable, fully-offline local LLM suite — chat, OCR documents, translate documents, transcribe speech, and **streaming voice-to-voice translation** (the holy grail).

Three workstreams: **A — Harden & fix bugs**, **B — Streaming voice-to-voice translation**, **C — UI overhaul**. Sequencing at the end.

---

## Workstream A — Hardening & bug exposure

> **Status (2026-06-10): EXECUTED** — all items below are fixed with regression tests
> (103 tests passing), **except A-P0 #1 (hf_token.txt), which the user chose to skip**,
> and the `stream: true` proxy gap, which is deferred to B-1 by design.
> Tooling (ruff/mypy/bandit/pip-audit configs + GitHub Actions CI) is in place.

Findings from a code audit, ordered by severity. Each one should become a fix + a regression test.

### A-P0 — Security / data-loss (fix first)

1. **`hf_token.txt` is tracked in git and is the live token path.**
   The file currently holds a placeholder, but [secrets.py](localllm/secrets.py) reads the *real* token from this exact path, `.gitignore` does **not** exclude it, and commit `8cdecf1 "Added token"` shows the commit-the-token workflow has already happened once. One real token + `git add .` + push = leaked credential.
   **Fix:** `git rm --cached hf_token.txt`, add `hf_token.txt` to `.gitignore`, ship `hf_token.txt.example` instead, and prefer `HF_TOKEN` env / `.env.local`. Extend [test_secrets.py](tests/test_secrets.py) with a guard test that fails if `hf_token.txt` is ever tracked (`git ls-files` check).

2. **Agent file tools can exfiltrate secrets; `write_note` path check is bypassable.**
   - [tools.py:56](localllm/agents/tools.py:56) uses `str(dest).startswith(str(out_dir))` — a sibling dir like `outputs2/` passes the prefix check (`"...\outputs2\x".startswith("...\outputs")` is true). Use `Path.relative_to()` like `_safe_path` does.
   - `read_file` is project-root-confined but happily reads `hf_token.txt` / `.env` — and the agent's tool dispatch is heuristic JSON parsing of model text ([graph.py](localllm/agents/graph.py)), so a prompt-injected document can trigger reads. **Fix:** denylist secret files in `_safe_path`, and add an allowlist of readable extensions.

3. **Gateway (`service/app.py`) is a thin proxy with no guardrails.**
   - No error handling around the upstream call ([app.py:93-98](localllm/service/app.py:93)) — if llama-server is down or times out, the client gets a raw 500 with traceback instead of a clean JSON error.
   - No request-size limit on `await request.json()` — a huge base64 audio body OOMs the gateway before llama-server ever sees it. Cap at config value (e.g. 64 MB).
   - The semaphore has no queue timeout — when 2 requests are in flight, every later caller hangs up to `timeout_sec=600`. Return `503 busy` after a bounded wait.
   - No auth. Fine while bound to `127.0.0.1`, but Phase 8 mentions LAN deploy — add an optional bearer-token check now (off by default) so it exists before the bind address ever changes.
   - **`stream: true` is silently broken** — the proxy buffers the whole upstream response. This is also a blocker for Workstream B; see B-1.

### A-P1 — Functional bugs

4. **`_concat_wav` silently discards audio** ([sentence_queue.py:61-72](localllm/tts/sentence_queue.py:61)): on sample-rate/channel mismatch it returns only `b`, throwing away everything spoken so far; it also hardcodes 16-bit width instead of reading it. Should resample or raise — never silently drop.
5. **TTS fallback voices produce garbage for ja/ko** ([piper.py:53-62](localllm/tts/piper.py:53)): Japanese text is synthesized with a *Chinese* voice and Korean with an *English* voice. Phoneme mapping fails → gibberish audio presented as a working translation. Disable TTS for unsupported languages with a visible "no local voice for X" notice instead.
6. **Live staging files are never cleaned up** ([streamlit_translate.py:36](apps/streamlit_translate.py:36), `_live_chunk_staging_dir`): every session writes recordings under the shared temp dir and nothing deletes them — disk leak, and recordings persist in a world-readable location. Add session cleanup + startup sweep of stale dirs.
7. **`_load_voice` caching ignores config** ([piper.py:110-116](localllm/tts/piper.py:110)): `lru_cache` keyed only on voice name, so `tts.use_cuda` changes are ignored until restart; and `_ensure_voice_files` downloads models *during a request*, stalling the UI and contradicting the offline guarantee. Pre-download via a `localllm-download --voices` step; fail fast offline.
8. **VAD threshold logic is fragile** ([vad.py:38-40](localllm/media/vad.py:38)): threshold = `max(abs, 5% of peak)` means silence-only audio (peak = noise floor) classifies *everything* as speech, while one loud clap suppresses quiet speech. Also `chunk_audio_vad` (the "Phase 2 VAD" path) is dead code — the live UI actually uses fixed windows (`chunk_audio_live`), while the UI text claims "VAD splits audio". Fix the claim or wire real VAD (see B-2, where this gets replaced anyway).
9. **`merge_transcripts` overlap heuristic can eat repeated phrases** ([audio.py:53-67](localllm/media/audio.py:53)): a genuine repetition ≥ 9 chars at a chunk boundary is deduplicated away. Acceptable for now, but add tests documenting the behavior; B-2's boundary handling supersedes it.
10. **Config default mismatch:** `GenerationConfig.enable_thinking = True` in [config.py:86](localllm/config.py:86) but the project decision (AGENT_HANDOFF §2) is *disabled* — anyone constructing `AppSettings()` without the YAML gets empty replies. Flip the code default to `False`.
11. **Trailing-audio drop in `chunk_audio`** ([audio.py:43-50](localllm/media/audio.py:43)): a final piece under 0.5 s is silently discarded — the last word of an utterance can vanish. Merge short tails into the previous chunk.

### A-P2 — Engineering hygiene

- **Tooling:** add `ruff` (lint+format) and `mypy` to dev deps; `pip-audit` + `bandit` in CI; pin a lockfile (`uv lock` or `pip-compile`).
- **CI:** GitHub Actions — pytest matrix (3.10–3.14, Windows + macOS), lint, secret-scanning gate (gitleaks).
- **New tests to write while fixing the above:** gateway error paths (upstream down / busy / oversized body), path-traversal attempts on agent tools, `_concat_wav` mismatch, unsupported-TTS-language behavior, stale staging cleanup.
- **Docs:** merge `README-run_instruct.md` (currently untracked) into `README.md`; refresh `AGENT_HANDOFF.md` after each milestone below.

---

## Workstream B — Streaming voice-to-voice translation (the holy grail)

**Goal:** continuous microphone input → segment → transcribe (Gemma audio) → translate (Gemma) → synthesize (Piper) → speak, all local, with the translated voice trailing the speaker by a few seconds.

### Why the current stack can't do it yet

- `st.audio_input` is record-*then*-stop; Streamlit has no continuous mic stream and its rerun model fights long-lived pipelines.
- The pipeline is strictly serial per chunk (STT → MT → TTS in [translate_chunked.py:116-149](localllm/pipelines/translate_chunked.py:116)) — no overlap between stages.
- The gateway can't stream responses (A-P0 #3), so incremental output is impossible through it.

### Latency budget (from your measurements: 30 s speech → STT < 5 s, MT < 5 s, TTS 5–8 s)

Scaling linearly to a **6–10 s speech segment** (the unit the live pipeline works in):

| Stage | per 8 s segment | Notes |
|---|---|---|
| Segmentation (VAD endpoint) | ~0.3–0.8 s | silence-hangover detection |
| STT (Gemma audio) | ~1.3 s | 5 s / 30 s × 8 s |
| MT (Gemma text) | ~1.3 s | short text, faster |
| TTS (Piper, per sentence) | ~1.5–2 s | starts on first completed sentence |
| **First translated audio after segment end** | **~4–5 s** | |

With stage pipelining (STT of segment *N+1* overlaps MT/TTS of segment *N* — the gateway's `max_concurrent_requests: 2` already allows it), steady-state lag stays **5–8 s behind the speaker** and never grows. That meets the target comfortably; everything below is engineering, not research.

### Architecture

A new **live translate service** — extend the existing FastAPI gateway (or a sibling app on `:8092`) with a WebSocket endpoint; clients are thin.

```
Browser (mic)                       FastAPI live service                    Gateway :8090
┌──────────────────┐   16 kHz PCM   ┌─────────────────────────────┐
│ AudioWorklet     ├───frames(WS)──►│ Ring buffer → VAD segmenter │
│ capture          │                │   (silero-vad ONNX, local)  │
│                  │                │ Segment queue ──► STT worker├──audio──► Gemma
│ Web Audio        │                │ Transcript q  ──► MT worker ├──text───► Gemma
│ playback queue   │◄──audio+json──┤ Sentence queue ─► TTS worker│           (2 concurrent)
└──────────────────┘    (WS)        │   (Piper, in-process)       │
                                    └─────────────────────────────┘
```

**Build steps:**

1. **B-1 Gateway streaming (prereq).** SSE pass-through in `chat_completions`: when body has `stream: true`, use `httpx` streaming + `StreamingResponse`. Also unblocks token-streaming chat (roadmap Phase 4) for free.
2. **B-2 Segmenter.** Replace the energy heuristic with **silero-vad** (tiny ONNX model, fully offline, battle-tested) for endpointing: emit a segment after ~600 ms of trailing silence, with min 2 s / max 12 s caps and 0.5 s pre-roll. Keep [vad.py](localllm/media/vad.py) energy mask as fallback. This replaces findings A-8/A-9/A-11.
3. **B-3 Async pipeline.** `localllm/pipelines/translate_live.py`: three asyncio workers (STT, MT, TTS) connected by queues; reuse `_transcribe_chunk`, `translate_text`, `synthesize_speech`, and the existing `SentenceQueue` for sentence-complete TTS gating. MT keeps a rolling context window (last 2 source/target sentences) for pronoun/tense coherence across segments.
4. **B-4 WebSocket endpoint + CLI harness.** `/ws/translate` accepting binary PCM frames in, emitting JSON events out (`partial_transcript`, `translation`, `tts_chunk` base64 or binary frames, `timing`). Before touching any browser code, build `localllm-live-bench`: feeds a WAV file through the WS *in real time* and reports per-stage latency + steady-state lag. This is the regression harness for the latency budget above.
5. **B-5 Browser client MVP.** One static page served by FastAPI: AudioWorklet capture → WS; playback via a buffered Web Audio queue. Push-to-talk first, then continuous mode. **Half-duplex by default** (pause mic while TTS plays) to prevent the translated voice from feeding back into the pipeline; headphone mode unlocks full duplex.
6. **B-6 Quality passes.** Boundary dedup using segment overlap; optional **faster-whisper** (small/medium, CTranslate2, offline) as a pluggable STT backend if Gemma's audio quality on short segments disappoints — the disabled `whisper_client.py` scaffold already anticipates this. Per-stage timing surfaced live in the UI.

**Risks:** Gemma audio accuracy on 2–4 s fragments (mitigation: min-segment floor + whisper fallback); GPU contention between STT and MT requests (mitigation: the semaphore already serializes; measure with B-4 harness before optimizing); echo/feedback (mitigation: half-duplex default).

---

## Workstream C — A beautiful, appealing interface

### Recommendation: a local-first web app — **React + TypeScript + Vite**, served as a static bundle by the FastAPI service

Streamlit got the project this far, but it cannot deliver continuous audio, polished visuals, or sub-second interactivity. The replacement should still honor the core promise: **no internet, no Node at runtime** — Node is build-time only; the compiled bundle ships in the repo/wheel and FastAPI serves it from `localhost`.

**Stack:**

| Layer | Choice | Why |
|---|---|---|
| Framework | React 19 + TypeScript + Vite | huge ecosystem, fast dev loop, static export |
| Styling | Tailwind CSS v4 | rapid, consistent, themeable |
| Components | shadcn/ui (Radix primitives) | accessible, beautiful defaults, owned code — no runtime dependency on a component cloud |
| Icons / motion | lucide-react + Framer Motion | subtle animated transitions, live waveforms feel alive |
| State | Zustand + TanStack Query | tiny, fits WS + REST mix |
| Audio | Web Audio API + AudioWorklet | mic capture & gapless playback for Workstream B |
| Visualization | custom canvas VU meter / waveform (or wavesurfer.js) | live feedback while speaking |
| Transport | WebSocket (live translate) + SSE (streaming chat) + REST (OCR/files) | matches B-1/B-4 |

**App shell (single dark-themed SPA, light toggle):**
- **Chat** — streaming tokens, image/audio/document attachments, markdown.
- **Translate · Live** — the showcase: dual waveform (you / translated voice), rolling bilingual transcript, language & voice pickers, per-stage latency chips (STT/MT/TTS), push-to-talk and continuous modes.
- **Translate · Documents** — drop a PDF/DOCX, side-by-side original/translation, export.
- **Transcribe** — batch audio → text with chunk progress.
- **OCR** — image/PDF → structured text.
- **Status** — model loaded, VRAM, gateway health, per-feature latency history.

**Lower-effort alternative (worth naming):** **Gradio 5** — Python-only, has built-in streaming audio in/out components, decent themes. It would get a live-translate demo running faster, but caps out on visual polish and custom interaction. If the priority order is "holy grail first, beauty second," start the B-5 MVP on a plain static page, then build the React shell; Gradio is the middle path only if you want to avoid a frontend toolchain entirely.

Migration: keep the Streamlit apps working until the SPA reaches feature parity, then deprecate.

---

## Sequencing & milestones

| # | Milestone | Contents | Est. effort |
|---|---|---|---|
| M0 | **Secure the repo** | A-P0 items 1–3 (token, agent tools, gateway guardrails) + regression tests | 1–2 days |
| M1 | **Gateway streaming** | B-1 SSE pass-through + streaming chat in CLI | 1 day |
| M2 | **Bug sweep** | A-P1 items 4–11 + tests; ruff/mypy/CI bootstrap (A-P2) | 2–3 days |
| M3 | **Live pipeline core** | B-2 silero-vad, B-3 async pipeline, B-4 WS endpoint + `localllm-live-bench` harness proving the 5–8 s lag target | 3–5 days |
| M4 | **Voice-to-voice MVP** | B-5 browser client (push-to-talk → continuous, half-duplex) | 2–3 days |
| M5 | **New UI shell** | C: Vite/React app — Translate·Live page first, then Chat/Docs/OCR/Status parity | 1–2 weeks, incremental |
| M6 | **Polish & package** | B-6 quality passes, voice pre-download command, one-command start (`localllm-up`), Docker refresh, AGENT_HANDOFF/README updates | ongoing |

**Definition of done for the holy grail (M4):** speak 30 s continuously into the mic → translated speech begins within ~8 s of the first sentence completing, steady-state lag ≤ 8 s, no internet connection, all timing visible in the UI, `localllm-live-bench` green in CI (mocked LLM) and on the RTX 5060 (real).
