# LocalLLM — Voice Translation Plan (Gemma-native)

> Real-time **voice translation** using **Gemma 4 multimodal audio** + **edge-tts** on **16 GB VRAM** (RTX 5060) and **MacBook Pro**.

**Status:** Phase 1 complete (batch translator in Sage UI)  
**Last updated:** 2026-06-08

---

## 1. Goals

| Goal | Description |
|------|-------------|
| **Single service** | `localllm-serve` (:8090) — Gemma handles audio in one pass |
| **Translator UI** | **Translate** tab in `localllm-streamlit` (merged with Chat / Agent) |
| **Pipeline** | Voice → Gemma (transcript + translation) → TTS playback |
| **UI** | Split screen: transcript (left) · translation (right) |

**North star:** continuous voice in → translated voice out with minimal perceived delay.

**Decision (2026-06):** Whisper split pipeline **disabled**. Gemma unified audio quality is sufficient for validation; simpler stack, one VRAM consumer.

---

## 2. Current building blocks

### LocalLLM (Gemma)
- Gateway: `http://127.0.0.1:8090`
- Inference: llama-server `:8080`, Gemma 4 12B multimodal
- Quants: **q6_k** (default), **q5_k** (optional, lower VRAM)
- Audio: native `input_audio` in chat completions
- TTS: **Piper** (local offline), warmed voice cache + voice picker

### Ports
| Service | Port |
|---------|------|
| llama-server (internal) | 8080 |
| LocalLLM gateway | 8090 |
| Sage Streamlit | 8501 |

### UI entry points
| Command | Opens |
|---------|-------|
| `localllm-streamlit` | Chat · Agent · **Translate** |
| `localllm-translate-streamlit` | Same app, Translate mode pre-selected |

---

## 3. VRAM (16 GB RTX)

| Profile | Quant | GPU VRAM est. | Notes |
|---------|-------|---------------|-------|
| **default** | Q6_K | ~10 GB | Recommended |
| **compact** | Q5_K | ~8 GB | Headroom for long context / future features |

```bash
# Start gateway only:
localllm-serve
# or
.\scripts\start_translation_stack.ps1 -Quantization q6_k
```

Mac: Metal + `q6_k` typical; no dual-service VRAM split needed.

---

## 4. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Sage Streamlit (Chat | Agent | Translate)                  │
│  · mic / file upload                                        │
│  · tone + voice picker (sidebar)                            │
│  · split UI: transcript | translation                       │
│  · TTS playback (edge-tts, warmed)                          │
└────────────────────────────┬────────────────────────────────┘
                             │ audio + prompt
                             ▼
┌────────────────────────────────────────────────────────────┐
│  LocalLLM Gateway :8090                                    │
│  POST /v1/chat/completions (multimodal)                    │
└────────────────────────────┬───────────────────────────────┘
                             ▼
                    llama-server + Gemma 4
                    (transcript + translation in one pass)
                             │
                             ▼
                      edge-tts → speaker
```

### Gemma prompt format
```
TRANSCRIPT:
<source language text>
TRANSLATION:
<target language text>
```

### Tone presets (UI sidebar)
- **Exact** — literal, dry/professional
- **Professional** — polished business
- **Friendly** — warm conversational
- **Cordial** — polite, personable

---

## 5. Completed (Phase 1)

| Item | Status |
|------|--------|
| Gemma unified `translate_audio()` pipeline | ✅ |
| Translate tab in `streamlit_chat.py` | ✅ |
| Tone + voice pickers | ✅ |
| TTS warmup (`warmup_tts()` on tab load) | ✅ |
| Chat attachment list + per-file detach | ✅ |
| Whisper code commented out | ✅ |

---

## 6. Phased roadmap (Gemma-only)

### Phase 2 — Chunked semi-real-time (2–3 weeks)
- VAD-based chunking (2–4 s windows, 0.5 s overlap) using Gemma audio chunks
- Reuse `localllm/media/audio.py` chunker (28 s max per call)
- Display partial transcript left, partial translation right on sentence boundaries
- TTS queue: speak completed sentences only (reduce jitter)

### Phase 3 — Streaming voice loop (4–6 weeks)
- **WebSocket hub** (`localllm-hub` on :8092):
  - Ingest mic stream from browser
  - Rolling Gemma context for transcript + translation
  - Stream TTS audio bytes back
- Browser: WebRTC or WebSocket mic capture (replace Streamlit `audio_input` latency)
- Target: **< 2 s** perceived delay for short phrases

### Phase 4 — Production polish
- Piper TTS offline option (replace edge-tts for latency + offline)
- systemd / Windows Service for gateway
- Language-pair presets, glossary injection (domain terms)
- Token streaming in Translate tab (live partial text)

---

## 7. TTS (local Piper)

| Item | Notes |
|------|-------|
| **Engine** | `piper-tts` — fully offline after voice model download |
| **Models** | Cached under `models/piper/` (one-time download per voice) |
| **Warmup** | Voice loaded when Translate tab opens |
| **Voices** | 2–3 Piper voices per language (sidebar radio) |

No internet required for TTS at runtime once voices are cached.

---

## 8. Whisper (archived)

Whisper split pipeline code is **commented out**, not deleted:
- `localllm/client/whisper_client.py`
- `localllm/pipelines/translate.py` (split branch)
- `config/default.yaml` whisper section

Re-enable only if Gemma WER regresses on a target language pair.

---

## 9. Success criteria

| Milestone | Test |
|-----------|------|
| Phase 1 ✅ | Upload 1 min WAV → transcript + translation + TTS with tone/voice |
| Phase 2 | Live mic chunks → partial updates < 4 s behind speech |
| Phase 3 | Browser stream → translated audio < 3 s behind speaker |
| VRAM | `nvidia-smi` stable under Q6_K + 8k context |

---

## 10. Immediate next steps

1. **Phase 2 spike:** VAD chunker + incremental Gemma calls on recorded mic stream
2. **TTS:** Evaluate Piper vs edge-tts latency on your network
3. **Streaming:** Prototype FastAPI WebSocket hub for mic bytes → Gemma → TTS stream
4. **Quality:** Build a small eval set (10 clips × 3 language pairs) for regression testing