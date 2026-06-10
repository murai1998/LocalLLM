import json

import pytest

from localllm.agents.tools import _safe_path, list_directory, read_file, write_note
from localllm.config import ROOT


def test_safe_path_blocks_secret_files():
    for name in ("hf_token.txt", ".env", ".env.local"):
        with pytest.raises(ValueError):
            _safe_path(name)


def test_safe_path_blocks_git_directory():
    with pytest.raises(ValueError):
        _safe_path(".git/config")


def test_read_file_cannot_read_secrets():
    with pytest.raises(ValueError):
        read_file.invoke({"path": "hf_token.txt"})


def test_read_file_rejects_unknown_extensions():
    # A binary-ish extension inside the project should be refused (suffix check
    # runs before existence check, so this holds even without the model file).
    result = read_file.invoke({"path": "models/gemma-4-12b-it-Q6_K.gguf"})
    assert result.startswith("Error: file type not readable")


def test_read_file_allows_plain_text():
    content = read_file.invoke({"path": "pyproject.toml"})
    assert "localllm" in content


def test_list_directory_blocks_git():
    with pytest.raises(ValueError):
        list_directory.invoke({"path": ".git"})


def test_write_note_rejects_sibling_directory_escape():
    # `outputs2` passes a naive startswith("…/outputs") check — must be rejected.
    result = write_note.invoke({"filename": "../outputs2/evil.txt", "content": "x"})
    assert result == "Error: invalid filename"
    assert not (ROOT / "outputs2").exists()


def test_write_note_rejects_parent_escape():
    result = write_note.invoke({"filename": "../../evil.txt", "content": "x"})
    assert result == "Error: invalid filename"


def test_write_note_writes_inside_outputs():
    result = write_note.invoke({"filename": "test_note_security.txt", "content": "ok"})
    payload = json.loads(result)
    dest = ROOT / "outputs" / "test_note_security.txt"
    assert dest.is_file()
    assert payload["written"] == str(dest)
    dest.unlink()
