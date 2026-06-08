from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

from localllm.config import AppSettings, get_settings
from localllm.model.quantization import resolve_gguf_file
from localllm.secrets import apply_hf_token


def _gguf_filename(settings: AppSettings, quantization: str | None = None) -> str:
    if quantization:
        return resolve_gguf_file(quantization=quantization, gguf_file="")
    return settings.model.resolved_gguf_file()


def ensure_gguf_assets(
    settings: AppSettings | None = None,
    *,
    quantization: str | None = None,
) -> tuple[Path, Path]:
    """Download configured GGUF quant and mmproj if missing."""
    settings = settings or get_settings()
    apply_hf_token()
    cache = settings.model_cache_dir
    cache.mkdir(parents=True, exist_ok=True)

    gguf_name = _gguf_filename(settings, quantization)
    gguf = cache / gguf_name
    mmproj = settings.mmproj_path()
    repo = settings.model.gguf_repo

    if not gguf.is_file():
        downloaded = hf_hub_download(
            repo_id=repo,
            filename=gguf_name,
            local_dir=str(cache),
        )
        gguf = Path(downloaded)

    if not mmproj.is_file():
        downloaded = hf_hub_download(
            repo_id=repo,
            filename=settings.model.mmproj_file,
            local_dir=str(cache),
        )
        mmproj = Path(downloaded)

    return gguf, mmproj