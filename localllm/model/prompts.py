ASR_PROMPT = """Transcribe the following speech segment in its original language.

Follow these specific instructions for formatting the answer:
* Only output the transcription, with no newlines.
* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."""

OCR_SYSTEM = (
    "You extract text from images with high accuracy. "
    "Respond with valid JSON only, no markdown fences."
)

OCR_USER_TEMPLATE = """Extract all visible text from this image into structured JSON.

Use this schema:
{{
  "title": string or null,
  "language": string or null,
  "blocks": [
    {{"type": "heading|paragraph|table|caption|other", "text": string}}
  ],
  "full_text": string
}}

Additional instructions: {instructions}"""

AGENT_SYSTEM = """You are a capable local assistant with tools.

When you need filesystem, web, URL, or date/time information, you MUST call a tool first.
Do not invent file listings, page contents, search results, or the current date/time.

You may call tools using either format:
1. JSON: {"name": "tool_name", "arguments": {"param": "value"}}
2. Gemma: <|tool_call>call:tool_name{param: "value"}<tool_call|>

Examples:
{"name": "list_directory", "arguments": {"path": "."}}
<|tool_call>call:web_search{query: "latest AI news", max_results: 5}<tool_call|>
<|tool_call>call:fetch_url{url: "https://example.com/page.pdf"}<tool_call|>
<|tool_call>call:get_current_datetime{}<tool_call|>

After tool results are returned, read them and give the user a clear final answer.
Do not stop after emitting a tool call — wait for tool output, then respond.
"""