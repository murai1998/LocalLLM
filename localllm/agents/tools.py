from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

from localllm.config import ROOT

# Files that may hold credentials — never exposed to the agent.
SECRET_FILENAMES = {"hf_token.txt", ".env", ".env.local", ".env.example"}
# Directories the agent must not traverse into.
BLOCKED_DIRS = {".git"}
# Only plain-text project files are readable ("" allows extensionless files
# like LICENSE / Dockerfile).
READABLE_SUFFIXES = {
    "",
    ".cfg",
    ".css",
    ".csv",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".ts",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _safe_path(path: str) -> Path:
    candidate = Path(path)
    p = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    if not _is_within_root(p, ROOT):
        raise ValueError("Path must stay inside the project directory.")
    if any(part.lower() in BLOCKED_DIRS for part in p.parts):
        raise ValueError("Path is not accessible.")
    if p.name.lower() in SECRET_FILENAMES:
        raise ValueError("Path is not accessible.")
    return p


@tool
def read_file(path: str) -> str:
    """Read a text file inside the project (relative or absolute within project)."""
    p = _safe_path(path)
    if p.suffix.lower() not in READABLE_SUFFIXES:
        return f"Error: file type not readable: {path}"
    if not p.is_file():
        return f"Error: file not found: {path}"
    return p.read_text(encoding="utf-8", errors="replace")[:8000]


@tool
def list_directory(path: str = ".") -> str:
    """List files and folders in a project directory (default: project root)."""
    p = _safe_path(path)
    if not p.is_dir():
        return f"Error: not a directory: {path}"
    entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    rel_root = p.resolve().relative_to(ROOT.resolve())
    header = f"Directory: {rel_root.as_posix() if str(rel_root) != '.' else '.'}"
    lines = [f"{'[dir]' if e.is_dir() else '[file]'} {e.name}" for e in entries[:100]]
    body = "\n".join(lines) or "(empty)"
    return f"{header}\n{body}"


@tool
def write_note(filename: str, content: str) -> str:
    """Write a short note to outputs/ inside the project."""
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    dest = (out_dir / filename).resolve()
    if dest == out_dir.resolve() or not _is_within_root(dest, out_dir):
        return "Error: invalid filename"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    return json.dumps({"written": str(dest)})
