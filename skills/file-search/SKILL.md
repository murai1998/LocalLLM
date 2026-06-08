---
name: file-search
description: Search the project codebase for filenames and text matches.
---

Use this skill when the user asks to find code, config, docs, or strings inside the LocalLLM project.

Workflow:

1. Start with `search_project` using the user's keywords.
2. Use `read_file` on the best matches to show exact context.
3. Prefer path + line references in your answer.

Examples:
{"name": "search_project", "arguments": {"query": "ServiceManager", "path": ".", "max_results": 10}}
<|tool_call>call:search_project{query: "fetch_url", max_results: 15}<tool_call|>