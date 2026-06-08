---
name: research-writer
description: Save structured research notes and summaries to outputs/.
---

Use this skill when the user wants a report, memo, comparison, or saved notes from research.

Workflow:

1. Gather facts with other tools first (`web_search`, `fetch_url`, `read_file`, etc.).
2. Synthesize a concise markdown summary for the user.
3. Offer to save the result with `write_note` when the answer is substantial.

Examples:
{"name": "write_note", "arguments": {"filename": "stupor-mundi-summary.md", "content": "# Summary\n..."}}
<|tool_call>call:write_note{filename: "news-brief.md", content: "# Today's headlines\n..."}<tool_call|>

Keep notes under 8 KB. Use clear headings and bullet points.