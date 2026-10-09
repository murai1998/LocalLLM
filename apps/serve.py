#!/usr/bin/env python3
"""Run the LocalLLM FastAPI gateway (single shared inference entry point)."""

from __future__ import annotations

import argparse
import os

import uvicorn

from localllm.config import get_settings
from localllm.devices.memory import InsufficientMemoryError, check_memory_fits
from localllm.model.presets import list_models
from localllm.secrets import apply_hf_token
from localllm.service.preflight import ensure_gateway_port_available


def main() -> None:
    parser = argparse.ArgumentParser(description="LocalLLM gateway service")
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="Start gateway only; assume llama-server is already running",
    )
    parser.add_argument(
        "--model",
        choices=list_models(),
        help="Model preset to serve (overrides config; e.g. gemma-4-31b, qwen3.8-27b)",
    )
    parser.add_argument(
        "--quant",
        help="Quantization within the model preset (default: the preset's default)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the check that the model (plus audio sidecar) fits in memory",
    )
    args = parser.parse_args()
    if args.force:
        os.environ["LOCALLLM_MODEL__SKIP_MEMORY_CHECK"] = "true"
    if args.model:
        os.environ["LOCALLLM_MODEL__NAME"] = args.model
        # A --model switch without --quant means "that preset's default quant".
        os.environ["LOCALLLM_MODEL__QUANTIZATION"] = args.quant or ""
    elif args.quant:
        os.environ["LOCALLLM_MODEL__QUANTIZATION"] = args.quant
    if args.no_inference:
        os.environ["LOCALLLM_SERVICE__AUTOSTART_LLAMA_SERVER"] = "false"
    apply_hf_token()

    settings = get_settings()
    preset = settings.model.preset()
    preset.quant(settings.model.quantization)  # fail fast on a bad --quant
    if not args.no_inference:
        try:
            need = check_memory_fits(settings)
        except InsufficientMemoryError as exc:
            raise SystemExit(f"[localllm] {exc}") from None
        print(f"[localllm] Estimated GPU memory: ~{need.total_gb:.1f} GB")
    ensure_gateway_port_available(
        host=settings.service.host,
        port=settings.service.port,
        base_url=settings.service.base_url,
    )
    print(f"[localllm] Gateway listening on {settings.service.base_url}")
    print(f"[localllm] Model: {preset.label} [{settings.model.resolved_gguf_file()}]")
    print(f"[localllm] Inference backend: {settings.llama_server.base_url}")
    if settings.audio_sidecar_active():
        audio = settings.audio_model_settings()
        print(
            f"[localllm] Audio sidecar: {audio.model.preset().label} "
            f"[{audio.model.resolved_gguf_file()}] on {audio.llama_server.base_url}"
        )
    uvicorn.run(
        "localllm.service.app:app",
        host=settings.service.host,
        port=settings.service.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()