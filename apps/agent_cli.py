#!/usr/bin/env python3
"""LangGraph agent CLI over local Gemma 4."""

from __future__ import annotations

import argparse
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from localllm.agents import (
    build_agent_graph,
    build_multi_agent_graph,
    invoke_agent,
    invoke_multi_agent,
    resolve_skills,
)
from localllm.secrets import apply_hf_token


def _print_tool_call(tc: dict) -> None:
    args = tc.get('args', {})
    print(f"🔍 [Tool Call] {tc['name']}({json.dumps(args, ensure_ascii=False)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph agent (local Gemma 4)")
    parser.add_argument("task", nargs="?", help="One-shot task prompt")
    parser.add_argument("--no-server", action="store_true", help="Assume llama-server is already running")
    parser.add_argument("--verbose", action="store_true", help="Show detailed trace")
    parser.add_argument("--skill", action="append", dest="skills", help="Enable specific skill(s)")
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Multi-agent mode: parallel researchers → analyst → critic",
    )

    args = parser.parse_args()

    apply_hf_token()
    skills = resolve_skills(args.skills if args.skills else ["internet-access"])
    if args.multi:
        graph = build_multi_agent_graph(autostart_server=not args.no_server, skills=skills)
        invoke = lambda g, state: invoke_multi_agent(g, state["messages"])  # noqa: E731
    else:
        graph = build_agent_graph(autostart_server=not args.no_server, skills=skills)
        invoke = invoke_agent

    if not args.task:
        print("Agent ready. Type /quit to exit.")
        while True:
            try:
                task = input("\nTask> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if task in {"/quit", "/exit", "/q"}:
                break
            if not task:
                continue
            _run_agent(graph, task, args.verbose, invoke=invoke)
        return

    # One-shot mode
    _run_agent(graph, args.task, args.verbose, invoke=invoke)


def _run_agent(graph, task: str, verbose: bool = False, *, invoke=invoke_agent):
    print(f"\n🔎 Agent thinking about: {task}")

    result = invoke(graph, {"messages": [HumanMessage(content=task)]})
    messages = result["messages"]

    if verbose:
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    _print_tool_call(tc)
            elif isinstance(msg, ToolMessage):
                print(f"📥 [Tool Result] {msg.name}\n{msg.content[:500]}...\n")

    final_reply = ""
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage) or not msg.content:
            continue
        if getattr(msg, "tool_calls", None):
            continue
        if "<|tool_call>" in msg.content:
            continue
        final_reply = msg.content
        break

    if final_reply:
        print(f"\n🧘 Sage> {final_reply}")
    else:
        print("\n🧘 Sage> (no final answer received — try --verbose to inspect tool steps)")


if __name__ == "__main__":
    main()