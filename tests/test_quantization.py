
import pytest

from localllm.config import AppSettings, ModelConfig, get_settings
from localllm.model.quantization import resolve_gguf_file


def test_default_quant_is_q6_k():
    model = ModelConfig()
    assert model.resolved_gguf_file() == "gemma-4-12b-it-Q6_K.gguf"


def test_q5_k_preset():
    model = ModelConfig(quantization="q5_k")
    assert model.resolved_gguf_file() == "gemma-4-12b-it-Q5_K_M.gguf"


def test_explicit_gguf_file_overrides_quant():
    model = ModelConfig(
        quantization="q5_k",
        gguf_file="custom-model.gguf",
    )
    assert model.resolved_gguf_file() == "custom-model.gguf"


def test_unknown_quantization_raises():
    with pytest.raises(ValueError):
        resolve_gguf_file(quantization="q3_k", gguf_file="")


def test_env_overrides_yaml_quantization(monkeypatch):
    monkeypatch.setenv("LOCALLLM_MODEL__QUANTIZATION", "q5_k")
    get_settings.cache_clear()
    settings = AppSettings.from_yaml()
    assert settings.model.quantization == "q5_k"
    assert settings.model.resolved_gguf_file() == "gemma-4-12b-it-Q5_K_M.gguf"
    get_settings.cache_clear()


def test_download_cli_quant_arg(monkeypatch):
    from scripts.download_model import main

    captured: dict[str, str] = {}

    def fake_ensure(settings, *, quantization=None):
        captured["quantization"] = quantization
        return settings.model_cache_dir / "gemma-4-12b-it-Q5_K_M.gguf", settings.mmproj_path()

    monkeypatch.setattr("localllm.model.download.ensure_gguf_assets", fake_ensure)
    monkeypatch.setattr("localllm.secrets.apply_hf_token", lambda: None)
    monkeypatch.setattr("sys.argv", ["localllm-download", "--quant", "q5_k"])

    main()

    assert captured["quantization"] == "q5_k"
