import pytest

from localllm.agents.tools import _is_within_root, _safe_path
from localllm.config import ROOT


def test_safe_path_allows_project_root_listing():
    path = _safe_path(".")
    assert path == ROOT.resolve()
    assert path.is_dir()


def test_safe_path_rejects_escape():
    with pytest.raises(ValueError):
        _safe_path("..")


def test_is_within_root_windows_safe():
    child = (ROOT / "README.md").resolve()
    assert _is_within_root(child, ROOT)
    outside = ROOT.parent.resolve()
    assert not _is_within_root(outside, ROOT)