#!/usr/bin/env python3
"""localllm-webui — serve the React web UI + REST API on :8095."""

from __future__ import annotations

import argparse
import platform
import socket
import sys

import uvicorn


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def main() -> None:
    parser = argparse.ArgumentParser(description="LocalLLM web UI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8095)
    args = parser.parse_args()

    if not _port_is_free(args.host, args.port):
        if platform.system() == "Windows":
            free_cmd = f"  .\\scripts\\stop_stale_ports.ps1 -Ports {args.port}"
        else:
            free_cmd = f"  ./scripts/stop_stale_ports.sh {args.port}"
        print(
            f"ERROR: port {args.port} is already in use — a previous web UI (or another\n"
            "process) is still listening. Free it with:\n\n"
            f"{free_cmd}\n\n"
            f"or start on another port:  localllm-webui --port {args.port + 1}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"LocalLLM web UI: http://{args.host}:{args.port}")
    print("Inference gateway must run separately: localllm-serve")
    uvicorn.run("localllm.webui.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
