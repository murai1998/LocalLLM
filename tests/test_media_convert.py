import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from localllm.agents.media_tools import convert_audio_file, extract_document_text
from localllm.agents.tool_registry import tools_for_skills
from localllm.agents.skills import resolve_skills
from localllm.media.convert import convert_audio_to_wav, ffmpeg_available, safe_media_path


def _write_wav(path: Path, seconds: float = 0.25, sr: int = 16000) -> None:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, audio, sr)


def test_convert_wav_roundtrip(tmp_path: Path):
    src = tmp_path / "tone.wav"
    _write_wav(src)
    out = convert_audio_to_wav(src, out_dir=tmp_path / "out")
    assert out.is_file()
    assert out.suffix == ".wav"


def test_m4a_requires_ffmpeg_when_unavailable(tmp_path: Path):
    fake_m4a = tmp_path / "clip.m4a"
    fake_m4a.write_bytes(b"not real m4a")
    with patch("localllm.media.convert.ffmpeg_available", return_value=False):
        with pytest.raises(RuntimeError, match="ffmpeg"):
            convert_audio_to_wav(fake_m4a, out_dir=tmp_path / "out")


def test_safe_media_path_allows_upload_cache(tmp_path: Path, monkeypatch):
    import localllm.media.convert as convert_mod

    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    file_path = upload_root / "clip.m4a"
    file_path.write_bytes(b"x")
    monkeypatch.setattr(convert_mod, "UPLOAD_CACHE_ROOT", upload_root)
    assert safe_media_path(str(file_path)) == file_path.resolve()


def test_convert_audio_file_tool_on_wav(tmp_path: Path, monkeypatch):
    import localllm.media.convert as convert_mod

    upload_root = tmp_path / "cache"
    upload_root.mkdir()
    monkeypatch.setattr(convert_mod, "UPLOAD_CACHE_ROOT", upload_root)
    src = upload_root / "a.wav"
    _write_wav(src)
    result = json.loads(convert_audio_file.invoke({"path": str(src)}))
    assert result["wav_path"]
    assert Path(result["wav_path"]).is_file()


def test_extract_document_text_tool(tmp_path: Path, monkeypatch):
    import localllm.media.convert as convert_mod

    upload_root = tmp_path / "cache"
    upload_root.mkdir()
    monkeypatch.setattr(convert_mod, "UPLOAD_CACHE_ROOT", upload_root)
    doc = upload_root / "note.txt"
    doc.write_text("hello convert", encoding="utf-8")
    result = json.loads(extract_document_text.invoke({"path": str(doc)}))
    assert "hello convert" in result["text"]


def test_media_convert_skill_registers_tools():
    skills = resolve_skills(["media-convert"])
    names = {tool.name for tool in tools_for_skills(skills)}
    assert "convert_audio_file" in names
    assert "extract_pdf_text" in names


def test_ffmpeg_available_is_bool():
    assert isinstance(ffmpeg_available(), bool)