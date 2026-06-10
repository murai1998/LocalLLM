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
    parser.add_argument(
        "--voices",
        nargs="*",
        metavar="LANG",
        help=(
            "Pre-download Piper TTS voices so runtime stays offline. "
            "Pass language codes (e.g. --voices en de ru) or no codes for all."
        ),
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip the GGUF/mmproj download (useful with --voices)",
    )
    args = parser.parse_args()

    apply_hf_token()

    from localllm.config import get_settings

    settings = get_settings()

    if not args.skip_model:
        from localllm.model.download import ensure_gguf_assets

        quant = args.quant or settings.model.quantization
        label = QUANT_PRESETS[quant]["label"]
        print(f"Quantization: {quant} ({label})")

        gguf, mmproj = ensure_gguf_assets(settings, quantization=quant)
        print(f"GGUF:   {gguf}")
        print(f"mmproj: {mmproj}")

    if args.voices is not None:
        from localllm.tts import download_voices

        languages = args.voices or None
        print(f"Piper voices: {', '.join(languages) if languages else 'all supported languages'}")
        for voice in download_voices(languages):
            print(f"voice:  {voice}")

    print("Done. Run: localllm-serve")


if __name__ == "__main__":
    main()
