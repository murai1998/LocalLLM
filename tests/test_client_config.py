from localllm.client.factory import create_llm_client
from localllm.config import AppSettings, LLMConfig, ServiceConfig


def test_default_llm_points_at_gateway():
    settings = AppSettings.from_yaml()
    assert settings.llm.provider == "local"
    assert settings.llm.base_url == "http://127.0.0.1:8090/v1"
    assert settings.service.port == 8090


def test_create_llm_client_uses_gateway_base_url():
    settings = AppSettings(
        llm=LLMConfig(base_url="http://127.0.0.1:8090/v1"),
        service=ServiceConfig(port=8090),
    )
    client = create_llm_client(settings)
    assert client.base_url == "http://127.0.0.1:8090/v1"


def test_default_translate_config():
    settings = AppSettings.from_yaml()
    assert settings.translate.target_language == "es"