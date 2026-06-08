from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "default.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _yaml_with_env_overrides(raw: dict[str, Any], *, env_prefix: str) -> dict[str, Any]:
    """Remove YAML nested keys that are overridden by LOCALLLM_* env vars."""
    if not raw:
        return {}

    merged = dict(raw)
    nested_prefix = f"{env_prefix.upper()}"

    for section, values in list(merged.items()):
        if not isinstance(values, dict):
            continue
        section_prefix = f"{nested_prefix}{section.upper()}__"
        overridden_fields = {
            key.split("__", 2)[-1].lower()
            for key in os.environ
            if key.upper().startswith(section_prefix)
        }
        if not overridden_fields:
            continue
        section_values = dict(values)
        for field in overridden_fields:
            section_values.pop(field, None)
        if section_values:
            merged[section] = section_values
        else:
            merged.pop(section, None)
    return merged


class ModelConfig(BaseSettings):
    gguf_repo: str = "unsloth/gemma-4-12b-it-GGUF"
    quantization: str = "q6_k"
    gguf_file: str = ""
    mmproj_file: str = "mmproj-F16.gguf"
    cache_dir: str = "models"

    def resolved_gguf_file(self) -> str:
        from localllm.model.quantization import resolve_gguf_file

        return resolve_gguf_file(
            quantization=self.quantization,
            gguf_file=self.gguf_file,
        )


class LlamaServerConfig(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8080
    context_size: int = 16384
    n_gpu_layers: int = 999
    jinja: bool = True
    startup_timeout_sec: int = 300

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class GenerationConfig(BaseSettings):
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    enable_thinking: bool = True
    stt_temperature: float = 0.1
    stt_max_tokens: int = 2048


class OcrConfig(BaseSettings):
    visual_detail: str = "high"
    max_pages: int = 20


class SttConfig(BaseSettings):
    sample_rate: int = 16000
    chunk_seconds: float = 28.0
    overlap_seconds: float = 2.0
    max_chunk_seconds: float = 30.0


# Whisper STT gateway — disabled (Gemma unified audio only).
# class WhisperClientConfig(BaseSettings):
#     base_url: str = "http://127.0.0.1:8091"
#     timeout_sec: float = 3600.0


class TranslateLiveConfig(BaseSettings):
    """Phase 2 — VAD chunking for semi-real-time translation."""

    min_chunk_seconds: float = 2.0
    max_chunk_seconds: float = 4.0
    overlap_seconds: float = 0.5
    frame_ms: int = 30
    energy_threshold: float = 0.01


class TranslateConfig(BaseSettings):
    source_language: str | None = None
    target_language: str = "es"
    max_tokens: int = 1024
    pipeline: str = "split"  # split | unified
    live: TranslateLiveConfig = Field(default_factory=TranslateLiveConfig)


class TtsConfig(BaseSettings):
    engine: str = "piper"
    model_dir: str = "models/piper"
    use_cuda: bool = False


class LLMConfig(BaseSettings):
    provider: str = "local"
    base_url: str = "http://127.0.0.1:8090/v1"
    api_key: str | None = None
    model: str = "gemma-4-12b-it"
    timeout_sec: float = 600.0


class ServiceConfig(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8090
    max_concurrent_requests: int = 2
    autostart_llama_server: bool = True
    startup_timeout_sec: int = 300

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LOCALLLM_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    model: ModelConfig = Field(default_factory=ModelConfig)
    llama_server: LlamaServerConfig = Field(default_factory=LlamaServerConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    # whisper: WhisperClientConfig = Field(default_factory=WhisperClientConfig)
    translate: TranslateConfig = Field(default_factory=TranslateConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    config_path: Path = DEFAULT_CONFIG_PATH

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> AppSettings:
        path = path or DEFAULT_CONFIG_PATH
        raw = _yaml_with_env_overrides(_load_yaml(path), env_prefix="LOCALLLM_")
        settings = cls(**raw) if raw else cls()
        return settings.model_copy(update={"config_path": path})

    @property
    def model_cache_dir(self) -> Path:
        p = Path(self.model.cache_dir)
        return p if p.is_absolute() else ROOT / p

    def gguf_path(self) -> Path:
        return self.model_cache_dir / self.model.resolved_gguf_file()

    def mmproj_path(self) -> Path:
        return self.model_cache_dir / self.model.mmproj_file


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings.from_yaml()