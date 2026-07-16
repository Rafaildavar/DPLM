"""Secure storage for user-supplied MAS provider credentials."""

from __future__ import annotations

import hashlib
from typing import Any
from urllib.parse import urlparse

from app.services.llm_providers import get_llm_provider_preset


MAS_KEYRING_SERVICE = "ai.gesturebind.mas"
MISTRAL_KEYRING_ACCOUNT = "mistral-api-key"


def _keyring_account(provider: str, api_url: str = "") -> str:
    normalized = str(provider or "custom").strip().lower().replace("-", "_")
    preset = get_llm_provider_preset(normalized)
    endpoint = str(api_url or "").strip().rstrip("/")
    preset_endpoint = str(preset.api_url if preset else "").strip().rstrip("/")
    custom_endpoint = bool(endpoint and endpoint != preset_endpoint)
    if normalized == "mistral" and not custom_endpoint:
        return MISTRAL_KEYRING_ACCOUNT
    safe = "".join(char for char in normalized if char.isalnum() or char == "_")
    account = safe or "custom"
    if custom_endpoint:
        parsed = urlparse(endpoint)
        origin = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"
        digest = hashlib.sha256(origin.encode("utf-8")).hexdigest()[:12]
        account = f"{account}-{digest}"
    return f"{account}-api-key"


class MasCredentialError(RuntimeError):
    """Raised when the operating-system credential store is unavailable."""


class MasCredentialStore:
    def __init__(self, backend: Any | None = None) -> None:
        self._backend = backend

    def _keyring(self) -> Any:
        if self._backend is not None:
            return self._backend
        try:
            import keyring
        except ImportError as exc:  # pragma: no cover - release dependency
            raise MasCredentialError("Системное хранилище ключей недоступно") from exc
        return keyring

    def get_api_key(self, provider: str, api_url: str = "") -> str:
        try:
            value = self._keyring().get_password(
                MAS_KEYRING_SERVICE,
                _keyring_account(provider, api_url),
            )
        except Exception as exc:
            raise MasCredentialError(
                "Не удалось прочитать ключ из системного хранилища"
            ) from exc
        return str(value or "").strip()

    def set_api_key(self, provider: str, api_key: str, api_url: str = "") -> None:
        value = str(api_key or "").strip()
        if not value:
            raise ValueError("API-ключ не может быть пустым")
        try:
            self._keyring().set_password(
                MAS_KEYRING_SERVICE,
                _keyring_account(provider, api_url),
                value,
            )
        except Exception as exc:
            raise MasCredentialError(
                "Не удалось сохранить ключ в системном хранилище"
            ) from exc

    def delete_api_key(self, provider: str, api_url: str = "") -> None:
        if not self.get_api_key(provider, api_url):
            return
        try:
            self._keyring().delete_password(
                MAS_KEYRING_SERVICE,
                _keyring_account(provider, api_url),
            )
        except Exception as exc:
            raise MasCredentialError(
                "Не удалось удалить ключ из системного хранилища"
            ) from exc

    # Backward-compatible helpers keep existing Mistral Keychain entries valid.
    def get_mistral_api_key(self) -> str:
        return self.get_api_key("mistral")

    def set_mistral_api_key(self, api_key: str) -> None:
        self.set_api_key("mistral", api_key)

    def delete_mistral_api_key(self) -> None:
        self.delete_api_key("mistral")


__all__ = [
    "MAS_KEYRING_SERVICE",
    "MISTRAL_KEYRING_ACCOUNT",
    "MasCredentialError",
    "MasCredentialStore",
]
