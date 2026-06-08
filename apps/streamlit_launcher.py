#!/usr/bin/env python3
"""Thin launcher for the Streamlit app (no Streamlit imports)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def launch() -> None:
    """Start streamlit_chat.py in a fresh Python process."""
    script = Path(__file__).resolve().parent / "streamlit_chat.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script),
        "--server.fileWatcherType",
        "none",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(cmd))