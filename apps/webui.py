#!/usr/bin/env python3
"""localllm-webui — serve the React web UI + REST API on :8095."""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="LocalLLM web UI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8095)
    args = parser.parse_args()

    print(f"LocalLLM web UI: http://{args.host}:{args.port}")
    print("Inference gateway must run separately: localllm-serve")
    uvicorn.run("localllm.webui.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
