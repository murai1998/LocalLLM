#!/usr/bin/env python3
"""Deprecated standalone translator — use streamlit_chat.py (Translate mode)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    os.environ["LOCALLLM_STREAMLIT_MODE"] = "Translate"
    script = Path(__file__).resolve().parent / "streamlit_chat.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(script), "--server.fileWatcherType", "none"]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()