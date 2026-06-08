#!/usr/bin/env python3
"""Batch speech-to-text using Gemma 4 native audio (llama.cpp)."""

from __future__ import annotations

import argparse
from pathlib import Path

from localllm.pipelines.stt_batch import transcribe_to_file
from localllm.secrets import apply_hf_token


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch STT (local Gemma 4)")
    parser.add_argument("inputs", nargs="+", type=Path, help="Audio file(s)")
    parser.add_argument("-o", "--output-dir", type=Path, help="Output directory for .txt files")
    args = parser.parse_args()
    apply_hf_token()

    out_dir = args.output_dir
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for inp in args.inputs:
        out = (out_dir / inp.with_suffix(".txt").name) if out_dir else None
        path = transcribe_to_file(inp, out)
        print(f"{inp} -> {path}")


if __name__ == "__main__":
    main()