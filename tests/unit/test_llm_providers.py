from app.services.llm_providers import (
    LLM_PROVIDER_PRESETS,
    get_llm_provider_preset,
    llm_api_key_env_names,
    normalize_llm_provider,
)


def test_provider_catalog_includes_cloud_local_and_custom_options():
    keys = {item.key for item in LLM_PROVIDER_PRESETS}

    assert {
        "local",
        "openai",
        "anthropic",
        "gemini",
        "mistral",
        "openrouter",
        "groq",
        "deepseek",
        "ollama",
        "custom",
    } <= keys
    assert get_llm_provider_preset("ollama").requires_api_key is False
    assert get_llm_provider_preset("openai").requires_api_key is True


def test_provider_aliases_and_environment_names_are_normalized():
    assert normalize_llm_provider("Claude") == "anthropic"
    assert normalize_llm_provider("openai-compatible") == "custom"
    assert llm_api_key_env_names("gemini") == (
        "DPLM_LLM_API_KEY",
        "LLM_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    )
