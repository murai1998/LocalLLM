"""Multi-agent orchestration over the local Gemma model.

A single task is worked by several role specialists arranged as a hybrid
fan-out / pipeline graph:

    START -> prepare
                 |-> researcher_primary    ┐ (run concurrently)
                 |-> researcher_contrarian ┘
                       -> analyst -> critic -> END

The researchers gather facts **in parallel** (each is a full tool-using agent
with its own lens). The analyst fans them in and synthesises a structured
answer; the critic red-teams that answer and produces the final, polished
result returned to the client.

Each role is just the existing single-agent tool loop (`build_agent_graph`)
driven by a role-specific system preamble — so all the tool-calling, sandboxing
and parsing behaviour is shared and already tested.

The graph keeps the *same* invocation contract as the single agent:
``invoke_agent(graph, {"messages": [HumanMessage(task)]})`` returns a dict whose
``messages`` ends with the final answer and contains the full tool trace, so the
CLI and Web UI consume it unchanged.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from localllm.agents.graph import (
    AGENT_RECURSION_LIMIT,
    build_agent_graph,
    extract_final_reply,
    invoke_agent,
)
from localllm.agents.skills import Skill
from localllm.config import AppSettings, get_settings
from localllm.service.manager import ServiceManager

# A multi-agent run chains several tool loops; give it more head-room than one.
MULTI_AGENT_RECURSION_LIMIT = AGENT_RECURSION_LIMIT * 3


@dataclass(frozen=True)
class Role:
    """A specialist seat in the multi-agent graph."""

    name: str
    preamble: str


# --- default roster -------------------------------------------------------

DEFAULT_RESEARCHERS: tuple[Role, ...] = (
    Role(
        name="researcher_primary",
        preamble=(
            "You are the Primary Researcher on a team answering one task. "
            "Use your tools to gather the core facts, figures, and authoritative "
            "sources that directly answer it. Prefer primary sources and include "
            "URLs. You are not writing the final answer — report what you found as "
            "concise, sourced bullet points for your teammates."
        ),
    ),
    Role(
        name="researcher_contrarian",
        preamble=(
            "You are the Contrarian Researcher on a team answering one task. "
            "Use your tools to surface what the obvious answer would miss: "
            "alternative viewpoints, counter-evidence, recent changes, caveats, "
            "and edge cases. Prefer primary sources and include URLs. You are not "
            "writing the final answer — report what you found as concise, sourced "
            "bullet points for your teammates."
        ),
    ),
)

ANALYST = Role(
    name="analyst",
    preamble=(
        "You are the Analyst. You are given a task and your researchers' findings. "
        "Synthesise them into a clear, well-structured answer that directly "
        "addresses the task. Weigh conflicting sources, state your reasoning, and "
        "flag any remaining uncertainty. Only call a tool if a critical fact is "
        "missing from the findings."
    ),
)

CRITIC = Role(
    name="critic",
    preamble=(
        "You are the Critic and final editor. You are given the task, the research "
        "findings, and the Analyst's draft answer. Rigorously check the draft for "
        "factual errors, unsupported claims, missing angles, and weak reasoning — "
        "verify key claims with tools if needed. Then write the FINAL answer for "
        "the user: correct, complete, and well-organised. Output only that final "
        "answer, with no meta-commentary about the critique process."
    ),
)


class MultiAgentState(TypedDict):
    task: str
    # Researchers run in parallel; both reducers below are commutative appends,
    # which is what makes concurrent writes safe.
    findings: Annotated[list[dict[str, str]], operator.add]
    trace: Annotated[list[BaseMessage], operator.add]
    # Only the critic writes here, once, at the end.
    messages: Annotated[list[BaseMessage], add_messages]


def _task_from_messages(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and m.content:
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _relabel_trace(role: str, messages: list[BaseMessage]) -> list[BaseMessage]:
    """Extract the tool-call / tool-result steps from a role's sub-run and tag
    them with the role, so the merged trace shows who did what. A role summary
    is appended as a synthetic tool result for readability."""
    out: list[BaseMessage] = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            out.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": f"{role}:{tc['name']}",
                            "args": tc.get("args") or {},
                            "id": tc.get("id", ""),
                        }
                        for tc in m.tool_calls
                    ],
                )
            )
        elif isinstance(m, ToolMessage):
            out.append(
                ToolMessage(
                    content=m.content,
                    tool_call_id=m.tool_call_id,
                    name=f"{role}:{m.name}",
                )
            )
    return out


def _role_summary(role: str, answer: str) -> ToolMessage:
    """A trace entry surfacing a role's contribution even when it used no tools."""
    return ToolMessage(content=answer, tool_call_id=f"{role}-summary", name=f"{role} (contribution)")


def build_multi_agent_graph(
    *,
    autostart_server: bool = True,
    skills: list[Skill] | None = None,
    researchers: tuple[Role, ...] | None = None,
    settings: AppSettings | None = None,
):
    """Compile the hybrid multi-agent graph. Roles reuse the single-agent tool
    loop; sub-agents never autostart the server (the orchestrator does it once)."""
    settings = settings or get_settings()
    researchers = researchers or DEFAULT_RESEARCHERS
    if autostart_server and settings.llm.provider == "local":
        ServiceManager.shared(settings).ensure_running()

    # One sub-graph (and its ChatEngine) per role, built eagerly so each role's
    # client is created now — and reused across invocations. Roles run on
    # distinct engine instances, so parallel researchers never share state.
    all_roles = (*researchers, ANALYST, CRITIC)
    sub_graphs: dict[str, Any] = {
        role.name: build_agent_graph(
            autostart_server=False,
            skills=skills,
            role_preamble=role.preamble,
        )
        for role in all_roles
    }

    def _run_role(role: Role, prompt: str) -> tuple[str, list[BaseMessage]]:
        result = invoke_agent(
            sub_graphs[role.name],
            {"messages": [HumanMessage(content=prompt)]},
            recursion_limit=AGENT_RECURSION_LIMIT,
        )
        msgs = result["messages"]
        answer = extract_final_reply(msgs) or "(no answer produced)"
        trace = _relabel_trace(role.name, msgs) + [_role_summary(role.name, answer)]
        return answer, trace

    def prepare(state: MultiAgentState) -> dict[str, Any]:
        return {"task": _task_from_messages(state["messages"])}

    def make_researcher(role: Role):
        def node(state: MultiAgentState) -> dict[str, Any]:
            answer, trace = _run_role(role, state["task"])
            return {
                "findings": [{"role": role.name, "answer": answer}],
                "trace": trace,
            }

        return node

    def analyst_node(state: MultiAgentState) -> dict[str, Any]:
        prompt = _compose(state["task"], state["findings"])
        answer, trace = _run_role(ANALYST, prompt)
        return {
            "findings": [{"role": ANALYST.name, "answer": answer}],
            "trace": trace,
        }

    def critic_node(state: MultiAgentState) -> dict[str, Any]:
        analysis = next(
            (f["answer"] for f in state["findings"] if f["role"] == ANALYST.name), ""
        )
        research = [f for f in state["findings"] if f["role"] != ANALYST.name]
        prompt = _compose(state["task"], research, analysis=analysis)
        final, critic_trace = _run_role(CRITIC, prompt)
        # Assemble the contract-compatible output in one write: full team trace
        # first, then the final answer as the last AIMessage.
        merged = state["trace"] + critic_trace + [AIMessage(content=final)]
        return {"messages": merged}

    graph = StateGraph(MultiAgentState)
    graph.add_node("prepare", prepare)
    graph.add_edge(START, "prepare")
    for role in researchers:
        graph.add_node(role.name, make_researcher(role))
        graph.add_edge("prepare", role.name)
        graph.add_edge(role.name, "analyst")
    graph.add_node("analyst", analyst_node)
    graph.add_node("critic", critic_node)
    graph.add_edge("analyst", "critic")
    graph.add_edge("critic", END)
    return graph.compile()


def _compose(task: str, findings: list[dict[str, str]], *, analysis: str = "") -> str:
    parts = [f"# Task\n{task}", ""]
    if findings:
        parts.append("# Research findings")
        for f in findings:
            parts.append(f"\n## From {f['role']}\n{f['answer']}")
    if analysis:
        parts.append(f"\n# Analyst's draft answer\n{analysis}")
    return "\n".join(parts).strip()


def invoke_multi_agent(
    graph,
    messages: list[BaseMessage],
    *,
    recursion_limit: int = MULTI_AGENT_RECURSION_LIMIT,
):
    """Run a multi-agent graph with the single-agent invocation contract."""
    return graph.invoke(
        {"messages": messages, "findings": [], "trace": []},
        config={"recursion_limit": recursion_limit},
    )
