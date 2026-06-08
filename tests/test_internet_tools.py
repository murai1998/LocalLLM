from unittest.mock import MagicMock, patch

from localllm.agents.internet import _fetch_public_url, _pdf_bytes_to_text


def test_pdf_bytes_to_text_extracts_pages():
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Rule 1: setup phase")
    pdf_bytes = doc.tobytes()
    doc.close()

    text = _pdf_bytes_to_text(pdf_bytes)
    assert "Rule 1" in text
    assert "Page 1" in text


def test_fetch_public_url_handles_pdf_content_type():
    pdf_bytes = b"%PDF-1.4"
    response = MagicMock()
    response.status_code = 200
    response.headers = {"content-type": "application/pdf"}
    response.content = pdf_bytes

    with (
        patch("localllm.agents.internet._validate_public_http_url", return_value="https://example.com/a.pdf"),
        patch("httpx.Client") as client_cls,
        patch("localllm.agents.internet._pdf_bytes_to_text", return_value="PDF rules text") as pdf_text,
    ):
        client = client_cls.return_value.__enter__.return_value
        client.get.return_value = response
        payload = _fetch_public_url("https://example.com/a.pdf")

    pdf_text.assert_called_once_with(pdf_bytes)
    assert "PDF rules text" in payload