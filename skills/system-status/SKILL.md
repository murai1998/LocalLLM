---
name: system-status
description: Current date/time, local platform, and LLM gateway health checks.
---

Use this skill when the user asks:
- what day or time it is (today, now, current date)
- how many days until a future date
- whether the local LLM service is running
- which GPU/platform is active

Workflow:

1. For ANY date/time question, call `get_current_datetime` first — never guess the clock.
2. Use the returned `local.date`, `local.time`, and `local.weekday` in your answer.
3. For "days until &lt;date&gt;" questions, compute the difference from `local.date`.
4. For service health, call `get_system_status` and explain `gateway` vs `inference`.

Examples:
<|tool_call>call:get_current_datetime{}<tool_call|>
{"name": "get_current_datetime", "arguments": {}}
<|tool_call>call:get_system_status{}<tool_call|>