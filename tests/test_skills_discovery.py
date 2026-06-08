from localllm.agents.skills import discover_skills
from localllm.agents.tool_registry import tools_for_skills


def test_discover_skills_includes_phase1_agents():
    names = {skill.name for skill in discover_skills()}
    assert "internet-access" in names
    assert "project-explorer" in names
    assert "file-search" in names
    assert "research-writer" in names
    assert "system-status" in names


def test_tools_for_system_status_includes_datetime():
    from localllm.agents.skills import resolve_skills

    tools = tools_for_skills(resolve_skills(["system-status"]))
    tool_names = {tool.name for tool in tools}
    assert "get_current_datetime" in tool_names
    assert "get_system_status" in tool_names


def test_tools_for_internet_and_file_search():
    from localllm.agents.skills import resolve_skills

    tools = tools_for_skills(resolve_skills(["internet-access", "file-search"]))
    tool_names = {tool.name for tool in tools}
    assert "fetch_url" in tool_names
    assert "web_search" in tool_names
    assert "search_project" in tool_names