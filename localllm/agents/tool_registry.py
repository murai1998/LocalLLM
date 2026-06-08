from __future__ import annotations

from langchain_core.tools import BaseTool

from localllm.agents.internet import fetch_url, web_search
from localllm.agents.media_tools import (
    convert_audio_file,
    extract_document_text,
    extract_pdf_text,
    render_pdf_page_images,
)
from localllm.agents.research_tools import get_current_datetime, get_system_status, search_project
from localllm.agents.skills import Skill
from localllm.agents.tools import list_directory, read_file, write_note

BASE_TOOLS: list[BaseTool] = [read_file, list_directory, write_note]

SKILL_TOOLS: dict[str, list[BaseTool]] = {
    "internet-access": [fetch_url, web_search],
    "file-search": [search_project],
    "system-status": [get_current_datetime, get_system_status],
    "media-convert": [
        convert_audio_file,
        extract_pdf_text,
        extract_document_text,
        render_pdf_page_images,
    ],
    # research-writer uses base write_note; skill adds workflow instructions only
}


def tools_for_skills(skills: list[Skill] | None = None) -> list[BaseTool]:
    tools: list[BaseTool] = list(BASE_TOOLS)
    enabled = {skill.name for skill in (skills or [])}
    for skill_name, extra_tools in SKILL_TOOLS.items():
        if skill_name in enabled:
            tools.extend(extra_tools)
    return tools
