from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from localllm.config import ROOT

SKILLS_DIR = ROOT / "skills"
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL | re.MULTILINE)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    content: str


def parse_skill_file(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    meta: dict = {}
    body = text.strip()

    match = FRONTMATTER_RE.match(text)
    if match:
        loaded = yaml.safe_load(match.group(1))
        meta = loaded if isinstance(loaded, dict) else {}
        body = text[match.end() :].strip()

    name = str(meta.get("name") or path.parent.name)
    description = str(meta.get("description") or "")
    return Skill(name=name, description=description, path=path, content=body)


@lru_cache
def discover_skills(skills_dir: str | None = None) -> tuple[Skill, ...]:
    root = Path(skills_dir) if skills_dir else SKILLS_DIR
    if not root.is_dir():
        return ()

    skills: list[Skill] = []
    for skill_file in sorted(root.glob("*/SKILL.md")):
        skills.append(parse_skill_file(skill_file))
    return tuple(skills)


def resolve_skills(names: list[str] | None = None) -> list[Skill]:
    """Return skills matching names, or all skills when names is None."""
    available = {skill.name: skill for skill in discover_skills()}
    if names is None:
        return list(available.values())

    resolved: list[Skill] = []
    missing: list[str] = []
    for name in names:
        skill = available.get(name)
        if skill is None:
            missing.append(name)
        else:
            resolved.append(skill)

    if missing:
        known = ", ".join(sorted(available)) or "(none installed)"
        raise ValueError(f"Unknown skill(s): {', '.join(missing)}. Available: {known}")
    return resolved


def format_skills_for_prompt(skills: list[Skill]) -> str:
    if not skills:
        return ""

    parts = ["# Active skills", ""]
    for skill in skills:
        parts.append(f"## Skill: {skill.name}")
        if skill.description:
            parts.append(skill.description)
        if skill.content:
            parts.append("")
            parts.append(skill.content)
        parts.append("")
    return "\n".join(parts).strip()
