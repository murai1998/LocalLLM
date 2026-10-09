import pytest

from localllm.config import AppSettings
from localllm.devices import memory
from localllm.devices.memory import (
    InsufficientMemoryError,
    MemoryBudget,
    check_memory_fits,
    memory_budget,
    memory_requirement,
)


def _budget(gb: float) -> MemoryBudget:
    return MemoryBudget(gb, "test")


def test_requirement_includes_audio_sidecar_for_models_without_audio():
    settings = AppSettings(model={"name": "gemma-4-31b"})
    need = memory_requirement(settings)
    assert len(need.parts) == 2
    assert need.total_gb == pytest.approx(22.9 + 10.1, abs=0.2)
    alone = memory_requirement(settings, include_sidecar=False)
    assert alone.total_gb == pytest.approx(22.9, abs=0.2)


def test_default_12b_has_no_sidecar_part():
    assert len(memory_requirement(AppSettings()).parts) == 1


def test_31b_qat_fits_48gb_mac_but_q6_does_not():
    budget = _budget(51.5 * memory.METAL_DEFAULT_FRACTION)  # 48 GiB Mac
    check_memory_fits(AppSettings(model={"name": "gemma-4-31b"}), budget=budget)
    with pytest.raises(InsufficientMemoryError, match=r"--quant q5_k"):
        check_memory_fits(
            AppSettings(model={"name": "gemma-4-31b", "quantization": "q6_k"}), budget=budget
        )


def test_big_models_refused_on_small_machine():
    budget = _budget(16 * memory.METAL_DEFAULT_FRACTION)  # 16 GB Mac
    for name in ("gemma-4-31b", "qwen3.8-27b"):
        with pytest.raises(InsufficientMemoryError, match="smaller model"):
            check_memory_fits(AppSettings(model={"name": name}), budget=budget)


def test_skip_flag_bypasses_guard():
    settings = AppSettings(model={"name": "gemma-4-31b", "skip_memory_check": True})
    check_memory_fits(settings, budget=_budget(4))


def test_undetectable_budget_does_not_block(monkeypatch):
    monkeypatch.setattr(memory, "memory_budget", lambda settings: None)
    check_memory_fits(AppSettings(model={"name": "gemma-4-31b"}))


def test_metal_budget_prefers_raised_wired_limit(monkeypatch):
    monkeypatch.setattr(memory, "total_ram_bytes", lambda: 48 * 2**30)
    monkeypatch.setattr(memory, "_metal_wired_limit_bytes", lambda: None)
    default = memory_budget(AppSettings(), platform_name="metal")
    assert default.available_gb == pytest.approx(48 * 2**30 * 0.75 / 1e9)
    monkeypatch.setattr(memory, "_metal_wired_limit_bytes", lambda: 40 * 2**30)
    raised = memory_budget(AppSettings(), platform_name="metal")
    assert "wired_limit" in raised.source


def test_cuda_budget_uses_vram_unless_cpu_only(monkeypatch):
    monkeypatch.setattr(memory, "_cuda_vram_bytes", lambda: 16 * 2**30)
    monkeypatch.setattr(memory, "total_ram_bytes", lambda: 64 * 2**30)
    assert memory_budget(AppSettings(), platform_name="cuda").source == "total NVIDIA VRAM"
    cpu = AppSettings(llama_server={"n_gpu_layers": 0})
    assert "RAM" in memory_budget(cpu, platform_name="cuda").source


def test_download_refuses_before_fetching(monkeypatch, tmp_path):
    from localllm.model import download

    calls = []
    monkeypatch.setattr(download, "hf_hub_download", lambda **kw: calls.append(kw))
    monkeypatch.setattr(memory, "memory_budget", lambda settings: _budget(8))
    settings = AppSettings(model={"name": "qwen3.8-27b", "cache_dir": str(tmp_path)})
    with pytest.raises(InsufficientMemoryError):
        download.ensure_gguf_assets(settings)
    assert calls == []
    assert not (tmp_path / "qwen3.8-27b").exists()
