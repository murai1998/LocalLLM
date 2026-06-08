---
name: internet-access
description: Search the public web and fetch pages when local project files are not enough.
---

Use this skill when the user asks about current events, external documentation, websites, or anything not in the local project.

Workflow:

1. Prefer `web_search` first to find relevant pages.
2. Use `fetch_url` to read a specific public http/https page when you need full content.
3. Cite the URLs you used in your final answer.
4. Do not guess page contents — fetch or search first.

Examples:
{"name": "web_search", "arguments": {"query": "llama.cpp release notes", "max_results": 5}}
{"name": "fetch_url", "arguments": {"url": "https://example.com/docs"}}

Only public http/https URLs are allowed. Localhost and private network addresses are blocked.
