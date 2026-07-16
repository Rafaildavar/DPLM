from __future__ import annotations

import pytest

from app.services.mas_credentials import (
    MAS_KEYRING_SERVICE,
    MISTRAL_KEYRING_ACCOUNT,
    MasCredentialError,
    MasCredentialStore,
)


class _MemoryKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, account: str):
        return self.values.get((service, account))

    def set_password(self, service: str, account: str, value: str) -> None:
        self.values[(service, account)] = value

    def delete_password(self, service: str, account: str) -> None:
        self.values.pop((service, account), None)


def test_mas_key_is_stored_and_deleted_through_keyring():
    backend = _MemoryKeyring()
    store = MasCredentialStore(backend)

    store.set_mistral_api_key("  user-secret  ")

    assert store.get_mistral_api_key() == "user-secret"
    assert backend.values[(MAS_KEYRING_SERVICE, MISTRAL_KEYRING_ACCOUNT)] == "user-secret"

    store.delete_mistral_api_key()
    assert store.get_mistral_api_key() == ""


def test_mas_keyring_errors_do_not_include_secret():
    secret = "sk-user-secret-value"

    class _BrokenKeyring:
        def set_password(self, *_args):
            raise RuntimeError("backend unavailable")

    with pytest.raises(MasCredentialError) as exc_info:
        MasCredentialStore(_BrokenKeyring()).set_mistral_api_key(secret)

    assert secret not in str(exc_info.value)
