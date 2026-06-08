#!/usr/bin/env python3
"""Launcher for Sage Streamlit with Translate mode pre-selected."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def launch() -> None:
    os.environ["LOCALLLM_STREAMLIT_MODE"] = "Translate"
    script = Path(__file__).resolve().parent / "streamlit_chat.py"
    print("=" * 60)
    print("  Sage — Translate mode")
    print("  App:   streamlit_chat.py (Mode: Translate)")
    print("  URL:   http://127.0.0.1:8501")
    print("=" * 60)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script),
        "--server.address",
        "127.0.0.1",
        "--server.fileWatcherType",
        "none",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(cmd))