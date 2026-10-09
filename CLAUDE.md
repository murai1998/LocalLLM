# LocalLLM — project instructions

## Cost-tiered delegation policy

The main session model is the expensive orchestrator/judge. Keep its context
lean and its output short; push volume work down to the cheap project agents
in `.claude/agents/`:

| Agent | Model | Route these tasks to it |
|---|---|---|
| `scout` | Haiku | "Where is X?", "does Y exist?", summarize a module, read logs/diffs and report the gist |
| `runner` | Haiku | Run pytest/ruff/builds/long commands; report distilled pass/fail + error lines only |
| `mechanic` | Haiku | Precisely specified mechanical edits: lint fixes, renames, version bumps, table entries, constant syncs |
| `writer` | Sonnet | READMEs, docs, HF Space cards, comparison pages, changelogs |

Routing rules:
- **Delegate** when the task is self-contained and describable in one prompt
  paragraph, or when the raw output (file contents, test logs) would bloat the
  orchestrator's context.
- **Don't delegate** work that needs the conversation's accumulated context,
  tight multi-step coupling, debugging of unclear failures, architecture or
  API design, or anything security-sensitive — the orchestrator does those
  itself.
- When delegating, write a self-contained prompt: exact file paths, exact
  expected change/answer, and the verification command. Subagents start cold.
- Verification of delegated edits: trust the agent's reported ruff/pytest
  output; spot-check only when the change touches shared contracts
  (prompts.py/piper_voices.py sync, public function signatures).

## Verification commands

- Lint (matches CI): `python -m ruff check localllm apps scripts tests`
- Showcase tests: `python -m pytest tests/test_showcase.py -q`
- Full suite: `python -m pytest -q`
- CI also runs bandit: `bandit -c pyproject.toml -r localllm apps scripts --severity-level medium`

## Project facts agents get wrong

- The showcase (`showcase/`) has three backends selected by `SHOWCASE_BACKEND`:
  `llama` (local default — llama-server + GGUF), `transformers` (HF ZeroGPU
  Space), `fake` (UI rehearsal). Don't "fix" one backend by breaking parity
  with the others' public API (`models.py` interface).
- `showcase/prompts.py` and `showcase/piper_voices.py` are vendored copies of
  `localllm/pipelines/translate.py` / `localllm/tts/piper.py` tables — tests
  assert they stay in sync; always change both sides.
- Local models are presets in `localllm/model/presets.py` (`gemma-4-12b` default,
  `gemma-4-31b`, `qwen3.8-27b`), picked via `model.name`, `LOCALLLM_PROFILE` or
  `localllm-serve --model`. Only Gemma 4 12B (and E2B/E4B) accept audio; for the
  others the gateway runs an audio sidecar (12B on :8081) and routes requests with
  `input_audio` parts there. Don't send audio to the 31B/Qwen directly.
- Live translation has two engines on `/ws/translate`: local pipeline and Gemini
  3.5 Live Translate (`localllm/live/gemini.py`, key from `GEMINI_API_KEY` /
  `.env.local`). Both emit the same event schema.
- Mermaid renders on GitHub, not on Hugging Face's file viewer.
- Shell is PowerShell 7 on Windows; `&&` works, but `$env:VAR="x"` must be a
  separate statement (not `cd x && $env:V=1`).
