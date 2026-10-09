from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from localllm.model.presets import ModelPreset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "default.yaml"
PROFILES_DIR = ROOT / "config" / "profiles"
# Named overlay from config/profiles/<name>.yaml, e.g. LOCALLLM_PROFILE=gemma-4-31b.
PROFILE_ENV = "LOCALLLM_PROFILE"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


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
    # Model preset id — see localllm/model/presets.py (gemma-4-12b | gemma-4-31b | qwen3.8-27b).
    name: str = "gemma-4-12b"
    # Optional overrides; empty means "use the preset's value".
    gguf_repo: str = ""
    # Pin to a commit hash for supply-chain safety; "main" tracks the latest upload.
    revision: str = "main"
    quantization: str = ""
    gguf_file: str = ""
    mmproj_file: str = ""
    cache_dir: str = "models"
    # Bypass the "does this model fit in memory" guard (localllm/devices/memory.py).
    skip_memory_check: bool = False

    def preset(self) -> ModelPreset:
        from localllm.model.presets import get_preset

        return get_preset(self.name)

    def resolved_gguf_file(self) -> str:
        from localllm.model.quantization import resolve_gguf_file

        return resolve_gguf_file(
            quantization=self.quantization,
            gguf_file=self.gguf_file,
            model=self.name,
        )

    def resolved_repo(self) -> str:
        if self.gguf_repo.strip():
            return self.gguf_repo.strip()
        preset = self.preset()
        quant = None if self.gguf_file.strip() else preset.quant(self.quantization)
        return (quant.repo if quant and quant.repo else "") or preset.gguf_repo

    def resolved_mmproj_file(self) -> str:
        if self.mmproj_file.strip():
            return self.mmproj_file.strip()
        preset = self.preset()
        quant = None if self.gguf_file.strip() else preset.quant(self.quantization)
        return (quant.mmproj_file if quant and quant.mmproj_file else "") or preset.mmproj_file


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
    # Disabled by default: enabling without a large token budget yields empty replies.
    enable_thinking: bool = False
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

    min_chunk_seconds: float = 9.0
    max_chunk_seconds: float = 10.0
    overlap_seconds: float = 1.0
    frame_ms: int = 30
    energy_threshold: float = 0.01


class TranslateStreamConfig(BaseSettings):
    """Workstream B — streaming voice-to-voice endpointing and pipeline."""

    frame_ms: int = 30
    # Trailing silence that ends a speech segment.
    hangover_ms: int = 600
    # Audio kept from before speech onset so word starts aren't clipped.
    pre_roll_ms: int = 300
    # Segments shorter than this are treated as noise blips and dropped.
    min_segment_seconds: float = 1.2
    # Hard cut for continuous speech (Gemma audio limit is 30 s).
    max_segment_seconds: float = 12.0
    # Absolute RMS floor; adaptive threshold never goes below this.
    energy_threshold: float = 0.0035
    # On flush (stream end), emit a trailing segment if at least this long.
    min_flush_seconds: float = 0.5
    # Pad very short segments up to this length before STT.
    stt_pad_seconds: float = 2.0


class TranslateConfig(BaseSettings):
    source_language: str | None = None
    target_language: str = "es"
    max_tokens: int = 1024
    pipeline: str = "split"  # split | unified
    live: TranslateLiveConfig = Field(default_factory=TranslateLiveConfig)
    stream: TranslateStreamConfig = Field(default_factory=TranslateStreamConfig)


class TtsConfig(BaseSettings):
    engine: str = "piper"
    model_dir: str = "models/piper"
    use_cuda: bool = False


class AudioSidecarConfig(BaseSettings):
    """Second llama-server that handles audio input for models without it.

    Gemma 4 31B and Qwen3.8-27B accept text + images only. With ``mode: auto``
    the gateway starts a small audio-capable model next to them and routes any
    request carrying ``input_audio`` parts to it, so STT / live translation keep
    working unchanged.
    """

    mode: str = "auto"  # auto | on | off
    name: str = "gemma-4-12b"
    quantization: str = "q5_k"
    host: str = "127.0.0.1"
    port: int = 8081
    context_size: int = 8192
    max_concurrent_requests: int = 1

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class GeminiConfig(BaseSettings):
    """Remote Gemini Live Translate engine (comparison backend for live translation).

    The API key comes from GEMINI_API_KEY / GOOGLE_API_KEY or ``.env.local``;
    never put it in YAML.
    """

    live_translate_model: str = "gemini-3.5-live-translate-preview"
    # Also speak translated audio when the speaker already uses the target language.
    echo_target_language: bool = False


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
    # Optional bearer token for /v1/* endpoints (set before exposing beyond localhost).
    api_key: str | None = None
    # How long a request may wait for an inference slot before 503.
    queue_timeout_sec: float = 30.0
    # Reject request bodies larger than this (base64 audio/images can get big).
    max_request_bytes: int = 64 * 1024 * 1024

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
    audio_sidecar: AudioSidecarConfig = Field(default_factory=AudioSidecarConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    profile: str = ""
    config_path: Path = DEFAULT_CONFIG_PATH

    @classmethod
    def from_yaml(cls, path: Path | None = None, *, profile: str | None = None) -> AppSettings:
        path = path or DEFAULT_CONFIG_PATH
        raw = _load_yaml(path)
        profile = (profile if profile is not None else os.environ.get(PROFILE_ENV, "")).strip()
        if profile:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", profile):
                raise ValueError(f"Invalid profile name '{profile}'.")
            profile_path = PROFILES_DIR / f"{profile}.yaml"
            if not profile_path.is_file():
                available = ", ".join(sorted(p.stem for p in PROFILES_DIR.glob("*.yaml")))
                raise ValueError(f"Unknown profile '{profile}'. Available: {available}")
            raw = _deep_merge(raw, _load_yaml(profile_path))
        raw = _yaml_with_env_overrides(raw, env_prefix="LOCALLLM_")
        settings = cls(**raw) if raw else cls()
        return settings.model_copy(update={"config_path": path, "profile": profile})

    @property
    def model_cache_dir(self) -> Path:
        p = Path(self.model.cache_dir)
        return p if p.is_absolute() else ROOT / p

    def model_dir(self) -> Path:
        return self.model_cache_dir / self.model.preset().subdir

    def gguf_path(self) -> Path:
        return self.model_dir() / self.model.resolved_gguf_file()

    def mmproj_path(self) -> Path:
        return self.model_dir() / self.model.resolved_mmproj_file()

    def audio_sidecar_active(self) -> bool:
        """True when a separate audio model must run beside the primary model."""
        if self.llm.provider != "local":
            return False
        mode = self.audio_sidecar.mode.strip().lower()
        if mode == "on":
            return True
        if mode == "off":
            return False
        return not self.model.preset().has_audio

    def audio_model_settings(self) -> AppSettings:
        """Settings view whose ``model`` is the audio sidecar's preset."""
        side = self.audio_sidecar
        model = ModelConfig(
            name=side.name,
            quantization=side.quantization,
            revision=self.model.revision,
            cache_dir=self.model.cache_dir,
        )
        server = self.llama_server.model_copy(
            update={"host": side.host, "port": side.port, "context_size": side.context_size}
        )
        return self.model_copy(update={"model": model, "llama_server": server})


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings.from_yaml()
