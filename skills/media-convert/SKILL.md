---
name: media-convert
description: Convert and extract content from audio, PDF, and document attachments.
---

Use this skill when the user attaches or references files that need conversion before analysis.

## Audio (m4a, mp3, flac, ogg, webm, aac)

1. Call `convert_audio_file` with the attachment path to produce 16 kHz mono WAV.
2. If conversion fails on m4a/mp3, tell the user to install **ffmpeg** on PATH.
3. After conversion, analyze the WAV path or rely on the pre-converted attachment in the user message.

## PDF

1. Call `extract_pdf_text` first for text-based PDFs.
2. If the result mode is `vision`, call `render_pdf_page_images` and describe page content.

## Documents (txt, md, docx, csv, json, yaml, html)

1. Call `extract_document_text` to read structured or plain text locally.

Workflow for mixed attachments: convert/extract each file, then answer using the extracted content.

Examples:
{"name": "convert_audio_file", "arguments": {"path": "C:/Users/.../localllm/streamlit_uploads/abc/clip.m4a"}}
{"name": "extract_pdf_text", "arguments": {"path": "docs/guide.pdf", "max_pages": 10}}
{"name": "extract_document_text", "arguments": {"path": "notes.md"}}