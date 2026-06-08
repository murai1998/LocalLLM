#!/usr/bin/env python3
"""Run the LocalLLM FastAPI gateway (single shared inference entry point)."""

from __future__ import annotations

import argparse
import os

import uvicorn

from localllm.config import get_settings
from localllm.secrets import apply_hf_token
from localllm.service.preflight import ensure_gateway_port_available


def main() -> None:
    parser = argparse.ArgumentParser(description="LocalLLM gateway service")
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="Start gateway only; assume llama-server is already running",
    )
    args = parser.parse_args()
    if args.no_inference:
        os.environ["LOCALLLM_SERVICE__AUTOSTART_LLAMA_SERVER"] = "false"
    apply_hf_token()

    settings = get_settings()
    ensure_gateway_port_available(
        host=settings.service.host,
        port=settings.service.port,
        base_url=settings.service.base_url,
    )
    print(f"[localllm] Gateway listening on {settings.service.base_url}")
    print(f"[localllm] Inference backend: {settings.llama_server.base_url}")
    uvicorn.run(
        "localllm.service.app:app",
        host=settings.service.host,
        port=settings.service.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()