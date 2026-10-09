#!/usr/bin/env python3
"""Download a model preset's GGUF (quant selectable) and mmproj from Hugging Face."""

from __future__ import annotations

import argparse

from localllm.devices.memory import InsufficientMemoryError
from localllm.model.presets import get_preset, list_models
from localllm.secrets import apply_hf_token


def main() -> None:
    parser = argparse.ArgumentParser(description="Download model GGUF + mmproj")
    parser.add_argument(
        "--model",
        choices=list_models(),
        help="Model preset (default: model.name from config)",
    )
    parser.add_argument(
        "--quant",
        help="Quantization within the preset (default: config, else the preset default)",
    )
    parser.add_argument(
        "--with-audio-sidecar",
        action="store_true",
        help="Also fetch the audio sidecar model when the chosen model has no audio input",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even if the model (plus audio sidecar) won't fit in this machine's memory",
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

        model_cfg = settings.model
        if args.model and args.model != model_cfg.name:
            model_cfg = model_cfg.model_copy(update={"name": args.model, "quantization": ""})
        if args.quant:
            model_cfg = model_cfg.model_copy(update={"quantization": args.quant})
        if args.force:
            model_cfg = model_cfg.model_copy(update={"skip_memory_check": True})
        settings = settings.model_copy(update={"model": model_cfg})
        preset = get_preset(model_cfg.name)
        quant = preset.quant(model_cfg.quantization)
        print(f"Model: {preset.label}")
        print(f"Quantization: {model_cfg.quantization or preset.default_quantization} "
              f"({quant.label}, ~{quant.size_gb:.1f} GB)")

        try:
            gguf, mmproj = ensure_gguf_assets(
                settings, quantization=model_cfg.quantization or None
            )
        except InsufficientMemoryError as exc:
            raise SystemExit(f"Not downloading: {exc}") from None
        print(f"GGUF:   {gguf}")
        print(f"mmproj: {mmproj}")

        if args.with_audio_sidecar and settings.audio_sidecar_active():
            audio = settings.audio_model_settings()
            print(f"Audio sidecar: {audio.model.preset().label} ({audio.model.quantization})")
            gguf, mmproj = ensure_gguf_assets(audio)
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
