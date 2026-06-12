"""Local dev launcher: real llama.cpp backend by default + preview-friendly port.

Set SHOWCASE_BACKEND=fake for UI-only rehearsal without starting llama-server.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("SHOWCASE_BACKEND", "llama")

import app  # noqa: E402

app.warm_backend()
app.demo.launch(
    server_name="127.0.0.1",
    server_port=int(os.environ.get("PORT", "7860")),
)
