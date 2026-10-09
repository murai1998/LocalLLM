from pathlib import Path

import pytest

from localllm.backends.llama_server import LlamaServerManager
from localllm.config import AppSettings, ModelConfig
from localllm.model.presets import PRESETS, get_preset


def test_every_preset_default_quant_resolves():
    for preset in PRESETS.values():
        assert preset.quant().file.endswith(".gguf")
        assert preset.served_name


def test_gemma_31b_profile_uses_qat_build_and_audio_sidecar():
    settings = AppSettings.from_yaml(profile="gemma-4-31b")
    assert settings.model.name == "gemma-4-31b"
    assert settings.model.resolved_repo() == "google/gemma-4-31B-it-qat-q4_0-gguf"
    assert settings.gguf_path().relative_to(settings.model_cache_dir) == Path(
        "gemma-4-31b/gemma-4-31B_q4_0-it.gguf"
    )
    assert settings.mmproj_path().name == "gemma-4-31B-it-mmproj.gguf"
    assert settings.audio_sidecar_active()


def test_qwen_profile_resolves_unsloth_dynamic_quant():
    settings = AppSettings.from_yaml(profile="qwen3.8-27b")
    assert settings.model.resolved_repo() == "unsloth/Qwen3.8-27B-GGUF"
    assert settings.gguf_path().name == "Qwen3.8-27B-UD-Q5_K_XL.gguf"
    # Same mmproj filename as the 12B — the per-model subdir keeps them apart.
    assert settings.mmproj_path().parent.name == "qwen3.8-27b"


def test_quant_hosted_in_another_repo_switches_repo_and_mmproj():
    model = ModelConfig(name="gemma-4-31b", quantization="q5_k")
    assert model.resolved_repo() == "unsloth/gemma-4-31B-it-GGUF"
    assert model.resolved_gguf_file() == "gemma-4-31B-it-Q5_K_M.gguf"
    assert model.resolved_mmproj_file() == "mmproj-F16.gguf"


def test_unknown_quant_for_preset_raises():
    with pytest.raises(ValueError, match="qwen3.8-27b"):
        get_preset("qwen3.8-27b").quant("q6_k")


def test_default_12b_keeps_flat_layout_and_no_sidecar():
    settings = AppSettings()
    assert settings.gguf_path() == settings.model_cache_dir / "gemma-4-12b-it-Q6_K.gguf"
    assert not settings.audio_sidecar_active()


def test_sidecar_modes():
    on = AppSettings(audio_sidecar={"mode": "on"})
    off = AppSettings(model={"name": "gemma-4-31b"}, audio_sidecar={"mode": "off"})
    remote = AppSettings(model={"name": "gemma-4-31b"}, llm={"provider": "openai"})
    assert on.audio_sidecar_active()
    assert not off.audio_sidecar_active()
    assert not remote.audio_sidecar_active()


def test_audio_model_settings_view_targets_sidecar_port():
    settings = AppSettings(model={"name": "qwen3.8-27b"})
    audio = settings.audio_model_settings()
    assert audio.model.name == "gemma-4-12b"
    assert audio.gguf_path().name == "gemma-4-12b-it-Q5_K_M.gguf"
    assert audio.llama_server.port == settings.audio_sidecar.port
    assert settings.llama_server.port != audio.llama_server.port


def test_profile_env_var_and_validation(monkeypatch):
    monkeypatch.setenv("LOCALLLM_PROFILE", "qwen3.8-27b")
    assert AppSettings.from_yaml().model.name == "qwen3.8-27b"
    with pytest.raises(ValueError, match="Invalid profile"):
        AppSettings.from_yaml(profile="../secrets")
    with pytest.raises(ValueError, match="Unknown profile"):
        AppSettings.from_yaml(profile="does-not-exist")


def test_env_override_beats_profile(monkeypatch):
    monkeypatch.setenv("LOCALLLM_MODEL__QUANTIZATION", "ud_q4_k_xl")
    settings = AppSettings.from_yaml(profile="qwen3.8-27b")
    assert settings.gguf_path().name == "Qwen3.8-27B-UD-Q4_K_XL.gguf"


def test_build_command_sets_alias_and_preset_server_args(monkeypatch):
    monkeypatch.setattr(
        "localllm.backends.llama_server.llama_server_binary", lambda: "llama-server"
    )
    settings = AppSettings(model={"name": "qwen3.8-27b"})
    cmd = LlamaServerManager(settings).build_command(Path("m.gguf"), Path("p.gguf"))
    assert cmd[cmd.index("--alias") + 1] == "qwen3.8-27b"
    assert cmd[cmd.index("--top-k") + 1] == "20"


def test_start_refuses_a_server_running_another_model(monkeypatch):
    settings = AppSettings(model={"name": "gemma-4-31b"})
    manager = LlamaServerManager(settings)
    monkeypatch.setattr(manager, "is_ready", lambda timeout=2.0: True)
    monkeypatch.setattr(manager, "served_model_ids", lambda: ["gemma-4-12b-it"])
    with pytest.raises(RuntimeError, match="already serving"):
        manager.start()
