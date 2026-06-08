from __future__ import annotations

import json
import re
from typing import Any


def normalize_tool_args(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _balanced_json_objects(text: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        start = i
        for j in range(i, len(text)):
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start : j + 1]
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        obj = None
                    if isinstance(obj, dict) and "name" in obj:
                        objects.append(obj)
                    i = j + 1
                    break
        else:
            i += 1
    return objects


def _normalize_call(obj: dict[str, Any], index: int) -> dict[str, Any] | None:
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = normalize_tool_args(obj.get("arguments", obj.get("args")))
    call_id = obj.get("id")
    if not isinstance(call_id, str) or not call_id:
        call_id = f"call_{index}"
    return {"name": name, "arguments": args, "id": call_id}


_GEMMA_TOOL_CALL_RE = re.compile(
    r"<\|tool_call>call:(?P<name>\w+)(?:\{(?P<args>[^}]*)\})?<(?:\|)?tool_call\|>",
    re.IGNORECASE,
)


def _parse_gemma_arg_string(raw: str) -> dict[str, Any]:
    cleaned = raw.strip().replace('<|"|>', '"').replace("<|'|>", "'")
    args: dict[str, Any] = {}
    pattern = re.compile(
        r'(\w+)\s*:\s*(?:"([^"]*)"|\'([^\']*)\'|([^,]+))'
    )
    for match in pattern.finditer(cleaned):
        key = match.group(1)
        value = match.group(2) or match.group(3) or (match.group(4) or "").strip()
        if key == "max_results":
            try:
                value = int(value)
            except (TypeError, ValueError):
                pass
        args[key] = value
    return args


def _parse_gemma_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for index, match in enumerate(_GEMMA_TOOL_CALL_RE.finditer(text)):
        name = match.group("name")
        raw_args = match.group("args") or ""
        args = _parse_gemma_arg_string(raw_args)
        calls.append({"name": name, "arguments": args, "id": f"gemma_call_{index}"})
    return calls


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse tool calls from JSON or Gemma native <|tool_call> format."""
    calls: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_call(obj: dict[str, Any]) -> None:
        normalized = _normalize_call(obj, len(calls))
        if normalized is None:
            return
        key = json.dumps(
            {"name": normalized["name"], "arguments": normalized["arguments"]},
            sort_keys=True,
        )
        if key in seen:
            return
        seen.add(key)
        calls.append(normalized)

    for gemma_call in _parse_gemma_tool_calls(text):
        add_call(gemma_call)

    for obj in _balanced_json_objects(text):
        add_call(obj)

    for fence in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE):
        stripped = fence.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            for obj in _balanced_json_objects(stripped):
                add_call(obj)
            continue
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    add_call(item)
        elif isinstance(parsed, dict):
            if "name" in parsed:
                add_call(parsed)
            elif "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
                for item in parsed["tool_calls"]:
                    if isinstance(item, dict):
                        add_call(item)

    return calls