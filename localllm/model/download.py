from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

from localllm.config import AppSettings, get_settings
from localllm.secrets import apply_hf_token


def ensure_gguf_assets(
    settings: AppSettings | None = None,
    *,
    quantization: str | None = None,
) -> tuple[Path, Path]:
    """Download the configured model preset's GGUF quant and mmproj if missing."""
    settings = settings or get_settings()
    if quantization:
        settings = settings.model_copy(
            update={
                "model": settings.model.model_copy(
                    update={"quantization": quantization, "gguf_file": ""}
                )
            }
        )
    apply_hf_token()
    target = settings.model_dir()
    repo = settings.model.resolved_repo()
    revision = settings.model.revision
    gguf = settings.gguf_path()
    mmproj = settings.mmproj_path()

    if not (gguf.is_file() and mmproj.is_file()):
        # Don't pull ~20 GB of weights onto a machine that can't run them.
        from localllm.devices.memory import check_memory_fits

        check_memory_fits(settings)
    target.mkdir(parents=True, exist_ok=True)

    if not gguf.is_file():
        downloaded = hf_hub_download(
            repo_id=repo,
            filename=gguf.name,
            revision=revision,
            local_dir=str(target),
        )
        gguf = Path(downloaded)

    if not mmproj.is_file():
        downloaded = hf_hub_download(
            repo_id=repo,
            filename=mmproj.name,
            revision=revision,
            local_dir=str(target),
        )
        mmproj = Path(downloaded)

    return gguf, mmproj
