from pathlib import Path

import pytest

from localllm.media.attachments import (
    AttachmentError,
    format_user_error,
    prepare_agent_context,
    prepare_chat_turn,
    validate_extension,
)


def test_chat_rejects_m4a_at_validation():
    err = validate_extension("clip.m4a", "chat")
    assert err is not None
    assert "WAV" in err
    assert "Agent" in err


def test_agent_accepts_m4a():
    assert validate_extension("clip.m4a", "agent") is None


def test_format_user_error_attachment():
    msg = format_user_error(AttachmentError("bad file"))
    assert "bad file" in msg


def test_prepare_chat_rejects_m4a_file(tmp_path: Path):
    audio = tmp_path / "x.m4a"
    audio.write_bytes(b"fake")
    with pytest.raises(AttachmentError):
        prepare_chat_turn("hi", [audio])


def test_prepare_agent_context_includes_attachment_note(tmp_path: Path):
    doc = tmp_path / "notes.txt"
    doc.write_text("hello agent", encoding="utf-8")
    text, images, audio = prepare_agent_context("summarize this", [doc])
    assert "Attached files:" in text
    assert "notes.txt" in text
    assert "hello agent" in text
    assert images == []
    assert audio is None