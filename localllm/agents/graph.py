from __future__ import annotations

import json
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from localllm.agents.parsing import parse_tool_calls
from localllm.agents.skills import Skill, format_skills_for_prompt
from localllm.agents.tool_registry import tools_for_skills
from localllm.chat.engine import ChatEngine
from localllm.model.prompts import AGENT_SYSTEM


AGENT_RECURSION_LIMIT = 12


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _tool_specs(tools) -> str:
    specs = []
    for t in tools:
        specs.append(
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.args_schema.model_json_schema() if t.args_schema else {},
            }
        )
    return json.dumps(specs, indent=2)


def _build_system_prompt(skills: list[Skill] | None = None) -> str:
    tools = tools_for_skills(skills)
    prompt = AGENT_SYSTEM + "\n\nAvailable tools:\n" + _tool_specs(tools)
    skill_block = format_skills_for_prompt(skills or [])
    if skill_block:
        prompt += "\n\n" + skill_block
    return prompt


def build_agent_graph(
    engine: ChatEngine | None = None,
    *,
    autostart_server: bool = True,
    skills: list[Skill] | None = None,
):
    tools = tools_for_skills(skills)
    tool_by_name = {t.name: t for t in tools}
    system_prompt = _build_system_prompt(skills)
    engine = engine or ChatEngine(
        system_prompt=system_prompt,
        autostart_server=autostart_server,
    )

    def run_tools(state: AgentState) -> dict[str, list[BaseMessage]]:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"messages": []}
        tool_messages: list[ToolMessage] = []
        for tc in last.tool_calls:
            name = tc["name"]
            args = tc.get("args") or {}
            tool = tool_by_name.get(name)
            if tool is None:
                content = f"Unknown tool: {name}"
            else:
                content = str(tool.invoke(args))
            tool_messages.append(
                ToolMessage(content=content, tool_call_id=tc["id"], name=name)
            )
        return {"messages": tool_messages}

    def agent_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        prior = state["messages"]
        api_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        for m in prior:
            if isinstance(m, HumanMessage):
                api_messages.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                api_messages.append({"role": "assistant", "content": m.content})
            elif isinstance(m, ToolMessage):
                api_messages.append(
                    {
                        "role": "user",
                        "content": f"Tool {m.name} returned:\n{m.content}",
                    }
                )

        engine.history = api_messages
        reply = engine.client.chat(engine.history)
        engine.history.append({"role": "assistant", "content": reply})

        calls = parse_tool_calls(reply)
        if calls:
            return {
                "messages": [
                    AIMessage(
                        content=reply,
                        tool_calls=[
                            {
                                "name": c["name"],
                                "args": c.get("arguments") or {},
                                "id": c.get("id", f"call_{i}"),
                            }
                            for i, c in enumerate(calls)
                        ],
                    )
                ]
            }
        return {"messages": [AIMessage(content=reply)]}

    def route(state: AgentState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", run_tools)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def invoke_agent(graph, state: AgentState, *, recursion_limit: int = AGENT_RECURSION_LIMIT):
    """Run the agent graph with a safe recursion limit (LangGraph invoke config)."""
    return graph.invoke(state, config={"recursion_limit": recursion_limit})