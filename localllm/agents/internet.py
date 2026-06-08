from __future__ import annotations

import json
import re
import socket
from html.parser import HTMLParser
from ipaddress import ip_address, ip_network
from urllib.parse import unquote, urlparse

import httpx
import fitz
from langchain_core.tools import tool

MAX_FETCH_BYTES = 500_000
MAX_TEXT_CHARS = 12_000
FETCH_TIMEOUT = 30.0
USER_AGENT = "LocalLLM-Agent/0.2 (local research)"

_BLOCKED_NETWORKS = (
    ip_network("0.0.0.0/8"),
    ip_network("10.0.0.0/8"),
    ip_network("127.0.0.0/8"),
    ip_network("169.254.0.0/16"),
    ip_network("172.16.0.0/12"),
    ip_network("192.168.0.0/16"),
    ip_network("::1/128"),
    ip_network("fc00::/7"),
    ip_network("fe80::/10"),
)

_BLOCKED_HOSTS = frozenset(
    {
        "localhost",
        "metadata.google.internal",
        "metadata.google",
    }
)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        elif self._skip_depth == 0 and tag in {"p", "br", "div", "li", "h1", "h2", "h3", "tr"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self._parts.append(text)

    def text(self) -> str:
        raw = "\n".join(self._parts)
        return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", raw)).strip()


def _is_blocked_ip(addr: str) -> bool:
    try:
        ip = ip_address(addr)
    except ValueError:
        return True
    return any(ip in net for net in _BLOCKED_NETWORKS)


def _validate_public_http_url(url: str) -> str:
    cleaned = url.strip()
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http and https URLs are allowed.")
    host = parsed.hostname
    if not host:
        raise ValueError("Invalid URL: missing host.")
    if host.lower() in _BLOCKED_HOSTS:
        raise ValueError("Blocked host.")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host: {host}") from exc

    for info in infos:
        sockaddr = info[4]
        if _is_blocked_ip(sockaddr[0]):
            raise ValueError("Blocked host: private or local addresses are not allowed.")
    return cleaned


def _pdf_bytes_to_text(content: bytes) -> str:
    doc = fitz.open(stream=content, filetype="pdf")
    try:
        chunks: list[str] = []
        for i in range(doc.page_count):
            text = doc.load_page(i).get_text("text").strip()
            if text:
                chunks.append(f"--- Page {i + 1} ---\n{text}")
        text = "\n\n".join(chunks)
    finally:
        doc.close()
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + "\n\n[... truncated ...]"
    return text or "(no extractable text in PDF)"


def _html_to_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    text = parser.text()
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + "\n\n[... truncated ...]"
    return text or "(no extractable text)"


def _fetch_public_url(url: str) -> str:
    safe_url = _validate_public_http_url(url)
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/json,text/plain,*/*"}
    with httpx.Client(timeout=FETCH_TIMEOUT, follow_redirects=True) as client:
        response = client.get(safe_url, headers=headers)
        response.raise_for_status()
        content = response.content[:MAX_FETCH_BYTES]
        content_type = response.headers.get("content-type", "").lower()
        status_code = response.status_code

    if "application/pdf" in content_type or safe_url.lower().endswith(".pdf"):
        text = _pdf_bytes_to_text(content)
    elif "application/json" in content_type:
        try:
            payload = json.loads(content.decode("utf-8", errors="replace"))
            text = json.dumps(payload, indent=2)
        except json.JSONDecodeError:
            text = content.decode("utf-8", errors="replace")
    else:
        text = _html_to_text(content.decode("utf-8", errors="replace"))

    return json.dumps(
        {
            "url": safe_url,
            "status": status_code,
            "content_type": content_type,
            "text": text,
        },
        ensure_ascii=False,
    )


def _parse_ddg_html(html: str, max_results: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    blocks = re.split(r'<div class="result results_links[^"]*"[^>]*>', html)
    for block in blocks[1:]:
        link_match = re.search(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not link_match:
            continue
        href = unquote(link_match.group(1))
        title = re.sub(r"<[^>]+>", "", link_match.group(2))
        title = re.sub(r"\s+", " ", title).strip()
        snippet_match = re.search(
            r'class="result__snippet"[^>]*>(.*?)</(?:a|td|div)>',
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        snippet = ""
        if snippet_match:
            snippet = re.sub(r"<[^>]+>", "", snippet_match.group(1))
            snippet = re.sub(r"\s+", " ", snippet).strip()
        if title and href:
            results.append({"title": title, "url": href, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


def _web_search(query: str, max_results: int = 5) -> str:
    cleaned = query.strip()
    if not cleaned:
        return "Error: empty search query."

    max_results = max(1, min(int(max_results), 10))
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(timeout=FETCH_TIMEOUT, follow_redirects=True) as client:
        response = client.post(
            "https://html.duckduckgo.com/html/",
            data={"q": cleaned, "b": "", "kl": ""},
            headers=headers,
        )
        response.raise_for_status()

    results = _parse_ddg_html(response.text, max_results)
    if not results:
        return json.dumps({"query": cleaned, "results": [], "note": "No results found."})
    return json.dumps({"query": cleaned, "results": results}, ensure_ascii=False)


@tool
def fetch_url(url: str) -> str:
    """Fetch a public web page over http/https and return extracted text."""
    try:
        return _fetch_public_url(url)
    except ValueError as exc:
        return f"Error: {exc}"
    except httpx.HTTPError as exc:
        return f"Error fetching URL: {exc}"


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the public web and return titles, URLs, and snippets."""
    try:
        return _web_search(query, max_results=max_results)
    except httpx.HTTPError as exc:
        return f"Error searching the web: {exc}"
