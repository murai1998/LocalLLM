from __future__ import annotations

import socket

import httpx


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def gateway_is_healthy(base_url: str, *, timeout: float = 2.0) -> bool:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        response = httpx.get(f"{root}/health", timeout=timeout)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def ensure_gateway_port_available(
    *,
    host: str,
    port: int,
    base_url: str,
) -> None:
    """Exit gracefully when the gateway is already healthy; fail if the port is blocked."""
    if gateway_is_healthy(base_url):
        print(f"[localllm] Gateway already running at {base_url} — exiting.")
        raise SystemExit(0)

    if port_is_free(host, port):
        return

    raise SystemExit(
        f"[localllm] Port {host}:{port} is already in use but /health is not responding.\n"
        "Stop the stale process or run: Get-NetTCPConnection -LocalPort 8090 | "
        "Select-Object OwningProcess"
    )