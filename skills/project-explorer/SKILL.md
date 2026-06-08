---
name: project-explorer
description: Explore the LocalLLM project structure and summarize key files.
---

When the user asks about the codebase, project layout, or "what's in this repo":

1. Call `list_directory` on `"."` first.
2. Read `README.md` and `AGENT_HANDOFF.md` when summarizing architecture.
3. Mention which apps exist (`localllm-chat`, `localllm-streamlit`, `localllm-agent`, etc.).
4. Keep answers concise — bullet lists over long prose.
