from unittest.mock import patch

from localllm.media import attachments


def test_process_pdf_truncates_long_text(tmp_path):
    pdf = tmp_path / "big.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    long_text = "word " * 5000
    with patch("localllm.media.attachments.extract_text_from_pdf", return_value=long_text):
        blocks, images = attachments._process_pdf(pdf, max_pages=20)

    assert not images
    assert "truncated" in blocks[0]
    assert len(blocks[0]) < len(long_text)


def test_process_pdf_uses_vision_for_scanned(tmp_path):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    page_png = tmp_path / "page_0001.png"
    page_png.write_bytes(b"\x89PNG")

    with (
        patch("localllm.media.attachments.extract_text_from_pdf", return_value=""),
        patch("localllm.media.attachments.pdf_needs_vision_ocr", return_value=True),
        patch("localllm.media.attachments.render_pdf_pages", return_value=[page_png]),
    ):
        blocks, images = attachments._process_pdf(pdf, max_pages=20)

    assert images == [page_png]
    assert "scanned document" in blocks[0]