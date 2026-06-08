#!/usr/bin/env python3
"""OCR / text extraction: PyMuPDF for PDFs, Gemma vision for images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from localllm.pipelines.ocr import process_path
from localllm.secrets import apply_hf_token


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR bot (local Gemma 4 vision + PyMuPDF)")
    parser.add_argument("input", type=Path, help="Image or PDF path")
    parser.add_argument("-o", "--output", type=Path, help="JSON output path")
    parser.add_argument(
        "--instructions",
        default="",
        help="Extra extraction instructions",
    )
    args = parser.parse_args()
    apply_hf_token()

    result = process_path(args.input, instructions=args.instructions)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()