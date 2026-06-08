from pathlib import Path
from unittest.mock import patch

from localllm.chat.engine import ChatEngine
from localllm.chat.schema import UserTurn
from localllm.client import media


def test_user_content_orders_image_before_text():
    engine = ChatEngine(autostart_server=False)
    turn = UserTurn(text="describe", image_paths=[Path("x.png")])
    with patch.object(
        engine.client,
        "image_part",
        return_value={"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
    ):
        content = engine._build_user_content(turn)
    assert isinstance(content, list)
    assert content[0]["type"] == "image_url"
    assert content[1]["type"] == "text"


def test_audio_part_structure():
    with patch.object(Path, "read_bytes", return_value=b"\x00\x01"):
        part = media.audio_part(Path("a.wav"))
    assert part["type"] == "input_audio"
    assert "data" in part["input_audio"]