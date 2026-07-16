"""Provider presets for the optional model-backed MAS adapter.

GestureBind speaks the OpenAI Chat Completions wire format.  The presets below
only supply a display name, endpoint and sensible starting model; users may
override the model and endpoint in Settings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LlmProviderPreset:
    key: str
    label: str
    api_url: str
    default_model: str
    requires_api_key: bool = True


LLM_PROVIDER_PRESETS: tuple[LlmProviderPreset, ...] = (
    LlmProviderPreset("local", "Локальный агент", "", "", False),
    LlmProviderPreset(
        "openai",
        "OpenAI",
        "https://api.openai.com/v1/chat/completions",
        "gpt-5-mini",
    ),
    LlmProviderPreset(
        "anthropic",
        "Anthropic Claude",
        "https://api.anthropic.com/v1/chat/completions",
        "claude-sonnet-4-6",
    ),
    LlmProviderPreset(
        "gemini",
        "Google Gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "gemini-3.5-flash",
    ),
    LlmProviderPreset(
        "mistral",
        "Mistral",
        "https://api.mistral.ai/v1/chat/completions",
        "mistral-small-latest",
    ),
    LlmProviderPreset(
        "openrouter",
        "OpenRouter",
        "https://openrouter.ai/api/v1/chat/completions",
        "openai/gpt-5-mini",
    ),
    LlmProviderPreset(
        "groq",
        "Groq",
        "https://api.groq.com/openai/v1/chat/completions",
        "openai/gpt-oss-20b",
    ),
    LlmProviderPreset(
        "deepseek",
        "DeepSeek",
        "https://api.deepseek.com/chat/completions",
        "deepseek-v4-flash",
    ),
    LlmProviderPreset(
        "ollama",
        "Ollama (локально)",
        "http://127.0.0.1:11434/v1/chat/completions",
        "gpt-oss:20b",
        False,
    ),
    LlmProviderPreset(
        "custom",
        "Свой OpenAI-совместимый API",
        "",
        "",
        False,
    ),
)

_PROVIDERS_BY_KEY = {item.key: item for item in LLM_PROVIDER_PRESETS}

PROVIDER_API_KEY_ENV: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "mistral": ("MISTRAL_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
}


def normalize_llm_provider(value: str | None) -> str:
    normalized = str(value or "local").strip().lower().replace("-", "_")
    aliases = {
        "claude": "anthropic",
        "google": "gemini",
        "mistral_api": "mistral",
        "openai_compatible": "custom",
        "openai_compatible_api": "custom",
    }
    return aliases.get(normalized, normalized)


def get_llm_provider_preset(value: str | None) -> LlmProviderPreset | None:
    return _PROVIDERS_BY_KEY.get(normalize_llm_provider(value))


def llm_provider_label(value: str | None) -> str:
    preset = get_llm_provider_preset(value)
    return preset.label if preset is not None else str(value or "LLM").strip()


def llm_provider_requires_api_key(value: str | None) -> bool:
    preset = get_llm_provider_preset(value)
    return bool(preset and preset.requires_api_key)


def llm_api_key_env_names(value: str | None) -> tuple[str, ...]:
    provider = normalize_llm_provider(value)
    return (
        "DPLM_LLM_API_KEY",
        "LLM_API_KEY",
        *PROVIDER_API_KEY_ENV.get(provider, ()),
    )


__all__ = [
    "LLM_PROVIDER_PRESETS",
    "LlmProviderPreset",
    "get_llm_provider_preset",
    "llm_api_key_env_names",
    "llm_provider_label",
    "llm_provider_requires_api_key",
    "normalize_llm_provider",
]
