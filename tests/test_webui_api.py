import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import localllm.webui.server as server


@pytest.fixture
def client():
    server._llm_client = MagicMock()
    server._llm_client.is_ready.return_value = True
    server._uploads.clear()
    yield TestClient(server.create_app())
    for stored in server._uploads.values():
        import shutil

        shutil.rmtree(stored.path.parent, ignore_errors=True)
    server._uploads.clear()
    server._llm_client = None


def test_health_reports_gateway_and_tts(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    payload = res.json()
    assert payload["gateway_ready"] is True
    assert "model" in payload
    assert "live_chunking" in payload


def test_meta_lists_languages_tones_voices(client):
    res = client.get("/api/meta")
    assert res.status_code == 200
    payload = res.json()
    assert payload["languages"]["en"] == "English"
    assert any(t["id"] == "professional" for t in payload["tones"])
    assert "es" in payload["voices"]
    assert "ja" not in payload["voices"]  # no garbage cross-language fallback


def test_chat_returns_reply_and_timing(client):
    server._llm_client.chat.return_value = "hello there"
    res = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert res.status_code == 200
    payload = res.json()
    assert payload["reply"] == "hello there"
    assert payload["elapsed_sec"] >= 0
    # System prompt is prepended server-side.
    sent = server._llm_client.chat.call_args.args[0]
    assert sent[0]["role"] == "system"


def test_chat_maps_gateway_failure_to_502(client):
    server._llm_client.chat.side_effect = RuntimeError("connection refused")
    res = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert res.status_code == 502


def test_chat_requires_messages(client):
    res = client.post("/api/chat", json={"messages": []})
    assert res.status_code == 422


def test_translate_text(client):
    fake = MagicMock(
        transcript="hola",
        translation="hello",
        target_language="en",
        llm_elapsed_sec=0.5,
        tone="professional",
    )
    with patch("localllm.webui.server.retranslate_transcript", return_value=fake):
        res = client.post(
            "/api/translate/text",
            json={"transcript": "hola", "source_lang": "es", "target_lang": "en"},
        )
    assert res.status_code == 200
    assert res.json()["translation"] == "hello"


def test_tts_unsupported_language_is_422(client):
    with patch("localllm.webui.server.PIPER_AVAILABLE", True):
        res = client.post("/api/tts", json={"text": "konnichiwa", "language": "ja"})
    assert res.status_code == 422


def test_upload_rejects_unknown_extension(client):
    res = client.post(
        "/api/transcribe",
        files={"file": ("evil.exe", b"MZ...", "application/octet-stream")},
    )
    assert res.status_code == 415


def test_upload_rejects_empty_file(client):
    res = client.post(
        "/api/transcribe",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert res.status_code == 400


def test_translate_audio_streams_ndjson_chunks_and_result(client):
    def fake_chunked(audio_path, *, on_progress=None, **kwargs):
        item = MagicMock(
            index=0,
            transcript="hola mundo",
            translation="hello world",
            elapsed_sec=1.2,
            start_sec=0.0,
            end_sec=8.0,
        )
        if on_progress:
            on_progress(1, 1, item)
        return MagicMock(
            transcript="hola mundo",
            translation="hello world",
            source_language="es",
            target_language="en",
            chunk_count=1,
            llm_elapsed_sec=1.2,
            tone="professional",
        )

    with patch("localllm.webui.server.translate_audio_chunked", side_effect=fake_chunked):
        res = client.post(
            "/api/translate/audio",
            files={"file": ("clip.wav", b"RIFF....WAVE", "audio/wav")},
            data={"source_lang": "es", "target_lang": "en", "tone": "professional"},
        )

    assert res.status_code == 200
    events = [json.loads(line) for line in res.text.strip().splitlines()]
    assert events[0]["type"] == "chunk"
    assert events[0]["translation"] == "hello world"
    assert events[-1]["type"] == "result"
    assert events[-1]["chunk_count"] == 1


def test_translate_audio_reports_pipeline_error(client):
    with patch(
        "localllm.webui.server.translate_audio_chunked",
        side_effect=RuntimeError("gateway down"),
    ):
        res = client.post(
            "/api/translate/audio",
            files={"file": ("clip.wav", b"RIFF....WAVE", "audio/wav")},
        )
    events = [json.loads(line) for line in res.text.strip().splitlines()]
    assert events[-1]["type"] == "error"
    assert "gateway down" in events[-1]["message"]


def test_skills_endpoint_lists_installed_skills(client):
    res = client.get("/api/skills")
    assert res.status_code == 200
    names = {s["name"] for s in res.json()}
    assert "file-search" in names  # ships with the repo


def test_upload_attach_and_detach(client):
    res = client.post(
        "/api/uploads",
        files=[
            ("files", ("a.png", b"\x89PNG fake", "image/png")),
            ("files", ("notes.txt", b"hello world", "text/plain")),
        ],
    )
    assert res.status_code == 200
    items = res.json()
    assert [i["kind"] for i in items] == ["image", "document"]
    assert all(server._uploads[i["id"]].path.is_file() for i in items)

    deleted = client.delete(f"/api/uploads/{items[0]['id']}")
    assert deleted.status_code == 200
    assert items[0]["id"] not in server._uploads
    assert client.delete(f"/api/uploads/{items[0]['id']}").status_code == 404


def test_upload_rejects_unsupported_attachment_type(client):
    res = client.post(
        "/api/uploads",
        files=[("files", ("evil.exe", b"MZ", "application/octet-stream"))],
    )
    assert res.status_code == 415


def test_chat_with_image_attachments_builds_multimodal_content(client, tmp_path):
    from PIL import Image

    img_bytes_path = tmp_path / "pic.png"
    Image.new("RGB", (4, 4), "red").save(img_bytes_path)

    uploaded = client.post(
        "/api/uploads",
        files=[
            ("files", ("pic1.png", img_bytes_path.read_bytes(), "image/png")),
            ("files", ("pic2.png", img_bytes_path.read_bytes(), "image/png")),
        ],
    ).json()

    server._llm_client.chat.return_value = "they differ"
    server._llm_client.image_part = MagicMock(
        side_effect=lambda p: {"type": "image_url", "path": str(p)}
    )
    server._llm_client.text_part = MagicMock(
        side_effect=lambda t: {"type": "text", "text": t}
    )

    res = client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "compare these"}],
            "attachment_ids": [u["id"] for u in uploaded],
        },
    )
    assert res.status_code == 200
    assert res.json()["reply"] == "they differ"
    sent = server._llm_client.chat.call_args.args[0]
    parts = sent[-1]["content"]
    assert isinstance(parts, list)
    assert sum(1 for p in parts if p["type"] == "image_url") == 2


def test_chat_with_unknown_attachment_id_is_404(client):
    res = client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "attachment_ids": ["nope"],
        },
    )
    assert res.status_code == 404


def test_chat_rejects_non_wav_audio_attachment_in_chat_mode(client):
    uploaded = client.post(
        "/api/uploads",
        files=[("files", ("song.mp3", b"ID3 fake mp3", "audio/mpeg"))],
    ).json()
    res = client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "transcribe"}],
            "attachment_ids": [uploaded[0]["id"]],
        },
    )
    assert res.status_code == 422
    assert "WAV" in res.json()["detail"]


def test_agent_mode_runs_graph_and_returns_steps(client):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    fake_result = {
        "messages": [
            HumanMessage(content="list files"),
            AIMessage(
                content="<|tool_call>…",
                tool_calls=[{"name": "list_directory", "args": {"path": "."}, "id": "c1"}],
            ),
            ToolMessage(content="[dir] apps", tool_call_id="c1", name="list_directory"),
            AIMessage(content="The project contains apps/ …"),
        ]
    }
    fake_graph = MagicMock()
    with (
        patch("localllm.webui.server._agent_graph", return_value=fake_graph),
        patch("localllm.agents.invoke_agent", return_value=fake_result),
    ):
        res = client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "list files"}],
                "mode": "agent",
                "skills": ["file-search"],
            },
        )
    assert res.status_code == 200
    payload = res.json()
    assert payload["mode"] == "agent"
    assert payload["reply"].startswith("The project contains")
    kinds = [s["kind"] for s in payload["steps"]]
    assert kinds == ["call", "result"]


def test_agent_mode_unknown_skill_is_422(client):
    res = client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "mode": "agent",
            "skills": ["does-not-exist"],
        },
    )
    assert res.status_code == 422


def test_spa_served_when_bundle_exists(client):
    if not server.DIST_DIR.is_dir():
        pytest.skip("webui bundle not built")
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    # Unknown client-side routes fall back to the SPA shell.
    deep = client.get("/translate/some/route")
    assert deep.status_code == 200
    assert "text/html" in deep.headers["content-type"]
