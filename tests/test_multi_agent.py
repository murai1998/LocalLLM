"""Multi-agent orchestration graph — runs entirely against a fake LLM client."""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from localllm.agents import extract_final_reply
from localllm.agents.multi_agent import (
    ANALYST,
    CRITIC,
    DEFAULT_RESEARCHERS,
    build_multi_agent_graph,
    invoke_multi_agent,
)


def _role_of(system_prompt: str) -> str:
    if "Primary Researcher" in system_prompt:
        return "researcher_primary"
    if "Contrarian Researcher" in system_prompt:
        return "researcher_contrarian"
    if "You are the Analyst" in system_prompt:
        return ANALYST.name
    if "You are the Critic" in system_prompt:
        return CRITIC.name
    return "unknown"


class _FakeClient:
    """Each role answers in one turn; records the order it was called in."""

    def __init__(self, calls: list[str], barrier: threading.Barrier | None = None):
        self.calls = calls
        self.barrier = barrier
        self._lock = threading.Lock()

    def chat(self, history, max_tokens=None):
        role = _role_of(history[0]["content"])
        # Researchers wait on a shared barrier — this only completes if they
        # truly run concurrently, proving the fan-out is parallel.
        if self.barrier is not None and role.startswith("researcher_"):
            self.barrier.wait(timeout=5)
        with self._lock:
            self.calls.append(role)
        if role == CRITIC.name:
            return "FINAL: the answer is 42."
        return f"[{role}] contribution"


def _build(client):
    with (
        patch("localllm.chat.engine.create_llm_client", return_value=client),
        patch("localllm.agents.multi_agent.ServiceManager"),
    ):
        return build_multi_agent_graph(autostart_server=False, skills=None)


def test_pipeline_order_and_final_reply():
    calls: list[str] = []
    graph = _build(_FakeClient(calls))
    out = invoke_multi_agent(graph, [HumanMessage(content="what is the answer?")])

    # Both researchers run, then analyst, then critic last.
    assert set(calls[:2]) == {"researcher_primary", "researcher_contrarian"}
    assert calls[2] == ANALYST.name
    assert calls[3] == CRITIC.name

    # The client receives the critic's output as the final answer.
    assert extract_final_reply(out["messages"]) == "FINAL: the answer is 42."
    # Final message is an AIMessage with no pending tool calls.
    assert isinstance(out["messages"][-1], AIMessage)
    assert not out["messages"][-1].tool_calls


def test_researchers_run_in_parallel():
    # Barrier(2) only releases when both researchers reach it at once. If the
    # fan-out ran serially, the first would block and time out → BrokenBarrier.
    barrier = threading.Barrier(2)
    graph = _build(_FakeClient([], barrier=barrier))
    out = invoke_multi_agent(graph, [HumanMessage(content="go")])
    assert extract_final_reply(out["messages"]).startswith("FINAL")


def test_trace_attributes_steps_to_roles():
    calls: list[str] = []
    graph = _build(_FakeClient(calls))
    out = invoke_multi_agent(graph, [HumanMessage(content="task")])

    summaries = {
        m.name
        for m in out["messages"]
        if isinstance(m, ToolMessage) and m.name and m.name.endswith("(contribution)")
    }
    expected = {f"{r.name} (contribution)" for r in DEFAULT_RESEARCHERS}
    expected |= {f"{ANALYST.name} (contribution)", f"{CRITIC.name} (contribution)"}
    assert summaries == expected


def test_input_messages_preserved():
    graph = _build(_FakeClient([]))
    out = invoke_multi_agent(graph, [HumanMessage(content="remember me")])
    assert any(
        isinstance(m, HumanMessage) and m.content == "remember me" for m in out["messages"]
    )


def test_custom_researcher_roster():
    from localllm.agents.multi_agent import Role

    calls: list[str] = []
    solo = (Role(name="researcher_primary", preamble="You are the Primary Researcher."),)
    with (
        patch("localllm.chat.engine.create_llm_client", return_value=_FakeClient(calls)),
        patch("localllm.agents.multi_agent.ServiceManager"),
    ):
        graph = build_multi_agent_graph(
            autostart_server=False, skills=None, researchers=solo
        )
    invoke_multi_agent(graph, [HumanMessage(content="x")])
    assert calls[0] == "researcher_primary"
    assert "researcher_contrarian" not in calls


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
