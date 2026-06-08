from localllm.agents.parsing import normalize_tool_args, parse_tool_calls


def test_parse_single_tool_call_with_nested_arguments():
    text = '{"name": "list_directory", "arguments": {"path": "."}}'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "list_directory"
    assert calls[0]["arguments"] == {"path": "."}


def test_parse_fenced_json_array():
    text = """Here is the tool call:
```json
[{"name": "read_file", "arguments": {"path": "README.md"}}]
```"""
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "read_file"
    assert calls[0]["arguments"]["path"] == "README.md"


def test_parse_stringified_arguments():
    text = '{"name": "list_directory", "arguments": "{\\"path\\": \\".\\"}"}'
    calls = parse_tool_calls(text)
    assert calls[0]["arguments"] == {"path": "."}


def test_normalize_tool_args_handles_none():
    assert normalize_tool_args(None) == {}


def test_parse_gemma_fetch_url_tool_call():
    text = (
        '<|tool_call>call:fetch_url'
        '{url: "https://example.com/rules.pdf"}<tool_call|>'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "fetch_url"
    assert calls[0]["arguments"]["url"] == "https://example.com/rules.pdf"


def test_parse_gemma_web_search_with_special_quotes():
    text = '<|tool_call>call:web_search{query:<|"|>main news today<|"|>}<tool_call|>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "web_search"
    assert calls[0]["arguments"]["query"] == "main news today"


def test_parse_gemma_tool_call_without_arguments():
    text = "<|tool_call>call:get_current_datetime{}<tool_call|>"
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "get_current_datetime"
    assert calls[0]["arguments"] == {}