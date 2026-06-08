import socket
from unittest.mock import MagicMock, patch

import httpx
import pytest

from localllm.service.preflight import (
    ensure_gateway_port_available,
    gateway_is_healthy,
    port_is_free,
)
from localllm.stack.health import service_health


def test_port_is_free_detects_listener():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        _, port = sock.getsockname()
        assert port_is_free("127.0.0.1", port) is False


def test_gateway_is_healthy_true():
    response = httpx.Response(200, json={"status": "ok"}, request=httpx.Request("GET", "http://test"))
    with patch("httpx.get", return_value=response):
        assert gateway_is_healthy("http://127.0.0.1:8090") is True


def test_ensure_gateway_port_available_exits_when_healthy():
    with patch("localllm.service.preflight.gateway_is_healthy", return_value=True):
        with pytest.raises(SystemExit) as exc:
            ensure_gateway_port_available(host="127.0.0.1", port=8090, base_url="http://127.0.0.1:8090")
    assert exc.value.code == 0


def test_service_health_reports_gateway():
    response = httpx.Response(
        200,
        json={"status": "ok", "inference_ready": True},
        request=httpx.Request("GET", "http://127.0.0.1:8090/health"),
    )
    with patch("httpx.get", return_value=response):
        statuses = service_health()
    assert len(statuses) == 1
    assert statuses[0].healthy