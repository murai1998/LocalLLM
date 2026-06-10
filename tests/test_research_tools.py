from unittest.mock import patch

from localllm.agents.research_tools import get_current_datetime, get_system_status, search_project


def test_search_project_finds_text(tmp_path, monkeypatch):
    monkeypatch.setattr("localllm.agents.research_tools.ROOT", tmp_path)
    monkeypatch.setattr("localllm.agents.tools.ROOT", tmp_path)
    sample = tmp_path / "sample.py"
    sample.write_text("def fetch_url_example():\n    return 1\n", encoding="utf-8")

    result = search_project.invoke({"query": "fetch_url", "path": ".", "max_results": 5})
    assert "fetch_url" in result
    assert "sample.py" in result


def test_get_current_datetime_returns_local_fields():
    import json

    payload = json.loads(get_current_datetime.invoke({}))
    assert "local" in payload
    assert "date" in payload["local"]
    assert "weekday" in payload["local"]


def test_get_system_status_returns_json():
    with (
        patch("localllm.agents.research_tools.httpx.get") as mock_get,
        patch("localllm.agents.research_tools.detect_platform", return_value="cuda"),
    ):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"status": "ok", "inference_ready": True}
        payload = get_system_status.invoke({})

    assert "cuda" in payload
    assert "gateway" in payload
    assert "inference" in payload
