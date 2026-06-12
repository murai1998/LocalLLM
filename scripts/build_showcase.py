#!/usr/bin/env python3
"""Verify the showcase/ distribution is self-contained and in sync.

Checks:
1. Vendored constants (tones, languages, Piper voices) match the localllm source.
2. The vendored endpointer logic matches localllm/live/endpointer.py.
3. GPU-free showcase modules import cleanly without torch/transformers.
4. No showcase module imports the `localllm` package.

Run before every `hf upload`:  python scripts/build_showcase.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOWCASE = ROOT / "showcase"
sys.path.insert(0, str(SHOWCASE))
sys.path.insert(0, str(ROOT))

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name}{(' - ' + detail) if detail and not ok else ''}")
    if not ok:
        failures.append(name)


def main() -> int:
    print("Sync checks:")
    import prompts as sc_prompts

    from localllm.pipelines import translate as ll_translate
    from localllm.tts import piper as ll_piper

    check("TONE_PRESETS match", sc_prompts.TONE_PRESETS == dict(ll_translate.TONE_PRESETS))
    check("LANGUAGE_LABELS match", sc_prompts.LANGUAGE_LABELS == ll_translate.LANGUAGE_LABELS)
    check("DEFAULT_TONE match", sc_prompts.DEFAULT_TONE == ll_translate.DEFAULT_TONE)

    import piper_voices as sc_voices

    check("VOICE_OPTIONS match", sc_voices.VOICE_OPTIONS == dict(ll_piper.VOICE_OPTIONS))

    # Translate prompt parity (without context, the messages must be identical).
    local_msgs = ll_translate.build_translate_messages(
        "hola", source_lang="es", target_lang="en", tone="friendly"
    )
    showcase_msgs = sc_prompts.build_translate_messages(
        "hola", source_lang="es", target_lang="en", tone="friendly"
    )
    check("translate prompt parity", local_msgs == showcase_msgs)

    # Endpointer parity: the class body must match the localllm source.
    def class_body(path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        match = re.search(r"class StreamingEndpointer.*", text, re.DOTALL)
        return match.group(0) if match else ""

    check(
        "endpointer logic parity",
        class_body(SHOWCASE / "endpointer.py")
        == class_body(ROOT / "localllm" / "live" / "endpointer.py"),
    )

    print("Self-containment:")
    for module in ("prompts", "piper_voices", "zerogpu", "endpointer", "interpreter"):
        try:
            __import__(module)
            check(f"import {module} (no torch needed)", True)
        except Exception as exc:
            check(f"import {module}", False, str(exc))

    for py in SHOWCASE.glob("*.py"):
        bad = bool(
            re.search(r"^\s*(from|import)\s+localllm", py.read_text(encoding="utf-8"), re.M)
        )
        check(f"{py.name} does not import localllm", not bad)

    if failures:
        print(f"\n{len(failures)} check(s) FAILED — fix before uploading.")
        return 1
    print("\nAll checks passed. Upload with:")
    print("  hf upload <user>/<space-name> ./showcase . --repo-type space")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
