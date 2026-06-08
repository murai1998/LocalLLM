#!/usr/bin/env python3
"""Terminal chat with Gemma 4 12B (llama.cpp) - Enhanced & Fully Compatible"""

from __future__ import annotations

import argparse
import itertools
import shlex
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from localllm.chat import ChatEngine, UserTurn
from localllm.secrets import apply_hf_token


def _parse_attachments(argv: list[str]) -> tuple[list[str], list[Path], Path | None]:
    images: list[Path] = []
    audio: Path | None = None
    text_parts: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--image" and i + 1 < len(argv):
            images.append(Path(argv[i + 1]))
            i += 2
        elif argv[i] == "--audio" and i + 1 < len(argv):
            audio = Path(argv[i + 1])
            i += 2
        else:
            text_parts.append(argv[i])
            i += 1
    return text_parts, images, audio


def ensure_conv_dir() -> Path:
    conv_dir = Path.home() / ".localllm" / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    return conv_dir


def save_conversation(history: list, name: str | None = None) -> Path:
    conv_dir = ensure_conv_dir()
    if not name:
        name = f"gemma4_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path = conv_dir / f"{name}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Sage Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        for msg in history:
            role = "🧑‍💻 User" if msg["role"] == "user" else "🧘 Sage"
            f.write(f"### {role}\n{msg.get('content', '')}\n\n")
    print(f"💾 Saved to: {path}")
    return path


def show_thinking_spinner(stop_event: threading.Event):
    for c in itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']):
        if stop_event.is_set():
            break
        sys.stdout.write(f'\r🧘 Sage is thinking... {c} ')
        sys.stdout.flush()
        time.sleep(0.08)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced Local Sage Chat")
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Assume the LLM gateway is already running (localllm-serve)",
    )
    args = parser.parse_args()

    apply_hf_token()
    engine = ChatEngine(autostart_server=not args.no_server)

    # === Welcome Message ===
    print("🧘 Hello! I am **Sage** — your wise and helpful companion.")
    print("Ready to think deeply, crack jokes, or assist you ✨\n")
    print("Type /help for commands\n")

    history = []

    while True:
        try:
            line = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n🧘 Farewell! May wisdom guide your path. 🌿")
            break

        if not line:
            continue

        if line.startswith("/"):
            cmd_parts = line.split()
            cmd = cmd_parts[0].lower()

            if cmd in {"/quit", "/exit", "/q"}:
                print("🧘 Sage says: Until next time... 🌿")
                break
            elif cmd == "/reset":
                engine.reset()
                history.clear()
                print("🧹 Conversation reset. A clear mind is a wise mind. ✨")
                continue
            elif cmd == "/history":
                for i, msg in enumerate(history[-10:], 1):
                    role = "You" if msg["role"] == "user" else "🧘 Sage"
                    preview = (msg["content"][:80] + "...") if len(msg["content"]) > 80 else msg["content"]
                    print(f"{i:2d}. {role}: {preview}")
                continue
            elif cmd == "/save":
                name = cmd_parts[1] if len(cmd_parts) > 1 else None
                save_conversation(history, name)
                continue
            elif cmd == "/help":
                print("""Available commands:
  /help          Show this help
  /reset         Clear conversation
  /history       Show last 10 messages
  /save [name]   Save chat as markdown
  /quit          Exit""")
                continue

        parts = shlex.split(line)
        text_parts, images, audio = _parse_attachments(parts)
        text = " ".join(text_parts)

        if not text and not images and not audio:
            print("Enter a message or use --image / --audio")
            continue

        if images or audio:
            print("📎 Attached:")
            for img in images:
                print(f"   🖼️  {img.name}")
            if audio:
                print(f"   🎤 {audio.name}")

        turn = UserTurn(text=text, image_paths=images, audio_path=audio)
        history.append({"role": "user", "content": text or "(multimodal)"})

        print("\n🧘 Sage> ", end="", flush=True)

        # Spinner
        stop_spinner = threading.Event()
        spinner_thread = threading.Thread(target=show_thinking_spinner, args=(stop_spinner,), daemon=True)
        spinner_thread.start()

        try:
            reply = engine.send(turn)
        finally:
            stop_spinner.set()
            spinner_thread.join(timeout=0.3)

        print("\r" + " " * 50, end="")
        print(reply)
        history.append({"role": "assistant", "content": reply})

    if history:
        save_conversation(history, "last_session")


if __name__ == "__main__":
    main()