"""Secure storage for user-supplied MAS provider credentials."""
from __future__ import annotations

from typing import Any


MAS_KEYRING_SERVICE = "ai.gesturebind.mas"
MISTRAL_KEYRING_ACCOUNT = "mistral-api-key"


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
            raise MasCredentialError(
                "Системное хранилище ключей недоступно"
            ) from exc
        return keyring

    def get_mistral_api_key(self) -> str:
        try:
            value = self._keyring().get_password(
                MAS_KEYRING_SERVICE,
                MISTRAL_KEYRING_ACCOUNT,
            )
        except Exception as exc:
            raise MasCredentialError(
                "Не удалось прочитать ключ из системного хранилища"
            ) from exc
        return str(value or "").strip()

    def set_mistral_api_key(self, api_key: str) -> None:
        value = str(api_key or "").strip()
        if not value:
            raise ValueError("API-ключ не может быть пустым")
        try:
            self._keyring().set_password(
                MAS_KEYRING_SERVICE,
                MISTRAL_KEYRING_ACCOUNT,
                value,
            )
        except Exception as exc:
            raise MasCredentialError(
                "Не удалось сохранить ключ в системном хранилище"
            ) from exc

    def delete_mistral_api_key(self) -> None:
        if not self.get_mistral_api_key():
            return
        try:
            self._keyring().delete_password(
                MAS_KEYRING_SERVICE,
                MISTRAL_KEYRING_ACCOUNT,
            )
        except Exception as exc:
            raise MasCredentialError(
                "Не удалось удалить ключ из системного хранилища"
            ) from exc


__all__ = [
    "MAS_KEYRING_SERVICE",
    "MISTRAL_KEYRING_ACCOUNT",
    "MasCredentialError",
    "MasCredentialStore",
]
