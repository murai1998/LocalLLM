#!/usr/bin/env python3
"""Download Gemma 4 12B GGUF (quant selectable) and mmproj from Hugging Face."""

from __future__ import annotations

import argparse

from localllm.model.quantization import QUANT_PRESETS, list_quantizations
from localllm.secrets import apply_hf_token


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Gemma 4 GGUF + mmproj")
    parser.add_argument(
        "--quant",
        choices=list_quantizations(),
        help="Quantization preset (overrides config for this download)",
    )
    args = parser.parse_args()

    apply_hf_token()

    from localllm.config import get_settings
    from localllm.model.download import ensure_gguf_assets

    settings = get_settings()
    quant = args.quant or settings.model.quantization
    label = QUANT_PRESETS[quant]["label"]
    print(f"Quantization: {quant} ({label})")

    gguf, mmproj = ensure_gguf_assets(settings, quantization=quant)
    print(f"GGUF:   {gguf}")
    print(f"mmproj: {mmproj}")
    print("Done. Run: localllm-serve")


if __name__ == "__main__":
    main()