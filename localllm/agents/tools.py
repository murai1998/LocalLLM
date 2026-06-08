from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

from localllm.config import ROOT


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
    return p


@tool
def read_file(path: str) -> str:
    """Read a text file inside the project (relative or absolute within project)."""
    p = _safe_path(path)
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
    if not str(dest).startswith(str(out_dir.resolve())):
        return "Error: invalid filename"
    dest.write_text(content, encoding="utf-8")
    return json.dumps({"written": str(dest)})