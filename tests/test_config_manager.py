"""Tests for twin.config_manager — encrypted keychain and config.json management."""

import json
import os
import pytest
from pathlib import Path

from twin.config import Provider
from twin.config_manager import ConfigManager, _derive_key, _get_machine_id


# Stable test machine ID so tests never touch the real platform-specific call.
_TEST_MACHINE_ID = "test-machine-id-1234"


@pytest.fixture
def cm(tmp_path: Path) -> ConfigManager:
    """ConfigManager scoped to a temp directory with a fixed machine ID."""
    return ConfigManager(data_dir=tmp_path, _machine_id=_TEST_MACHINE_ID)


# ── Machine ID and key derivation ────────────────────────────────────────────

def test_get_machine_id_returns_nonempty_string() -> None:
    """Platform-specific machine ID should be a non-empty string."""
    machine_id = _get_machine_id()
    assert isinstance(machine_id, str)
    assert len(machine_id) > 0


def test_derive_key_returns_32_bytes() -> None:
    """Derived key must be 32 bytes (AES-256)."""
    key = _derive_key(_TEST_MACHINE_ID)
    assert len(key) == 32


def test_derive_key_is_deterministic() -> None:
    """Same machine ID always produces the same key."""
    key1 = _derive_key(_TEST_MACHINE_ID)
    key2 = _derive_key(_TEST_MACHINE_ID)
    assert key1 == key2


def test_derive_key_differs_for_different_machine_ids() -> None:
    """Different machine IDs produce different keys (non-portable by design)."""
    key1 = _derive_key("machine-a")
    key2 = _derive_key("machine-b")
    assert key1 != key2


# ── Keychain encrypt / decrypt round-trip ────────────────────────────────────

def test_set_and_get_key_round_trip(cm: ConfigManager) -> None:
    """A key set via set_key() is returned unchanged by get_key()."""
    cm.set_key(Provider.ANTHROPIC, "sk-ant-test-key")
    assert cm.get_key(Provider.ANTHROPIC) == "sk-ant-test-key"


def test_get_key_missing_returns_none(cm: ConfigManager) -> None:
    """get_key() returns None when no key has been set for a provider."""
    assert cm.get_key(Provider.OPENAI) is None


def test_set_key_multiple_providers(cm: ConfigManager) -> None:
    """Keys for different providers are stored and retrieved independently."""
    cm.set_key(Provider.ANTHROPIC, "ant-key")
    cm.set_key(Provider.OPENAI, "oai-key")
    assert cm.get_key(Provider.ANTHROPIC) == "ant-key"
    assert cm.get_key(Provider.OPENAI) == "oai-key"


def test_remove_key(cm: ConfigManager) -> None:
    """remove_key() deletes a stored key without affecting other providers."""
    cm.set_key(Provider.ANTHROPIC, "ant-key")
    cm.set_key(Provider.OPENAI, "oai-key")
    cm.remove_key(Provider.ANTHROPIC)
    assert cm.get_key(Provider.ANTHROPIC) is None
    assert cm.get_key(Provider.OPENAI) == "oai-key"


def test_remove_key_nonexistent_is_noop(cm: ConfigManager) -> None:
    """Removing a key that was never set does not raise."""
    cm.remove_key(Provider.GEMINI)  # should not raise


def test_keychain_file_is_not_plaintext(cm: ConfigManager, tmp_path: Path) -> None:
    """The keychain file must not contain the raw API key in plaintext."""
    secret = "super-secret-api-key"
    cm.set_key(Provider.ANTHROPIC, secret)
    raw_bytes = (tmp_path / "keychain.enc").read_bytes()
    assert secret.encode() not in raw_bytes


def test_wrong_machine_id_cannot_decrypt(tmp_path: Path) -> None:
    """A keychain written with one machine ID cannot be read with another."""
    cm_a = ConfigManager(data_dir=tmp_path, _machine_id="machine-a")
    cm_a.set_key(Provider.ANTHROPIC, "secret-key")

    cm_b = ConfigManager(data_dir=tmp_path, _machine_id="machine-b")
    # Should return None (silent failure) rather than raising.
    assert cm_b.get_key(Provider.ANTHROPIC) is None


# ── API key resolution order ─────────────────────────────────────────────────

def test_resolve_api_key_from_keychain(cm: ConfigManager) -> None:
    """Keychain key is returned when present."""
    cm.set_key(Provider.ANTHROPIC, "keychain-key")
    assert cm.resolve_api_key(Provider.ANTHROPIC) == "keychain-key"


def test_resolve_api_key_env_fallback(cm: ConfigManager, monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variable is used when no keychain entry exists."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    assert cm.resolve_api_key(Provider.ANTHROPIC) == "env-key"


def test_resolve_api_key_keychain_takes_priority_over_env(
    cm: ConfigManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keychain key takes priority over the environment variable."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    cm.set_key(Provider.ANTHROPIC, "keychain-key")
    assert cm.resolve_api_key(Provider.ANTHROPIC) == "keychain-key"


def test_resolve_api_key_raises_when_absent(cm: ConfigManager, monkeypatch: pytest.MonkeyPatch) -> None:
    """KeyError with an onboarding message is raised when no key exists anywhere."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(KeyError, match="twin config set-key"):
        cm.resolve_api_key(Provider.ANTHROPIC)


def test_resolve_api_key_ollama_no_key_needed(cm: ConfigManager) -> None:
    """Ollama always returns an empty string — no key is required."""
    result = cm.resolve_api_key(Provider.OLLAMA)
    assert result == ""


# ── Active provider config ───────────────────────────────────────────────────

def test_set_and_get_active_provider(cm: ConfigManager) -> None:
    """set_active_provider / get_active_provider round-trip."""
    cm.set_active_provider(Provider.OPENAI)
    assert cm.get_active_provider() == Provider.OPENAI


def test_get_active_provider_default(cm: ConfigManager) -> None:
    """Default active provider is Anthropic when nothing is configured."""
    assert cm.get_active_provider() == Provider.ANTHROPIC


def test_get_active_provider_env_override(
    cm: ConfigManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TWIN_PROVIDER env var is used when config.json has no provider set."""
    monkeypatch.setenv("TWIN_PROVIDER", "gemini")
    assert cm.get_active_provider() == Provider.GEMINI


def test_config_json_takes_priority_over_env(
    cm: ConfigManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """config.json provider takes priority over TWIN_PROVIDER env var."""
    monkeypatch.setenv("TWIN_PROVIDER", "gemini")
    cm.set_active_provider(Provider.OPENAI)
    assert cm.get_active_provider() == Provider.OPENAI


def test_invalid_provider_in_config_defaults_to_anthropic(
    cm: ConfigManager, tmp_path: Path
) -> None:
    """An unrecognised provider value in config.json silently falls back to Anthropic."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"active_provider": "not-a-real-provider"}))
    assert cm.get_active_provider() == Provider.ANTHROPIC


# ── Model config ─────────────────────────────────────────────────────────────

def test_set_and_get_model(cm: ConfigManager) -> None:
    """set_model / get_model round-trip for a provider."""
    cm.set_model(Provider.ANTHROPIC, "claude-sonnet-4-5")
    assert cm.get_model(Provider.ANTHROPIC) == "claude-sonnet-4-5"


def test_get_model_missing_returns_none(cm: ConfigManager) -> None:
    """get_model() returns None when no model has been configured."""
    assert cm.get_model(Provider.OPENAI) is None


def test_set_model_multiple_providers_independent(cm: ConfigManager) -> None:
    """Models for different providers are stored independently."""
    cm.set_model(Provider.ANTHROPIC, "claude-opus-4-8")
    cm.set_model(Provider.OPENAI, "gpt-4o")
    assert cm.get_model(Provider.ANTHROPIC) == "claude-opus-4-8"
    assert cm.get_model(Provider.OPENAI) == "gpt-4o"


# ── Vault path ───────────────────────────────────────────────────────────────

def test_set_and_get_vault_path(cm: ConfigManager, tmp_path: Path) -> None:
    """Vault path is stored and returned as a Path object."""
    vault = tmp_path / "vault"
    cm.set_vault_path(vault)
    result = cm.get_vault_path()
    assert result == vault.resolve()


def test_get_vault_path_none_when_unset(cm: ConfigManager) -> None:
    """get_vault_path() returns None when no vault has been configured."""
    assert cm.get_vault_path() is None


# ── list_config safety ───────────────────────────────────────────────────────

def test_list_config_no_key_values(cm: ConfigManager) -> None:
    """list_config() output must never contain actual API key strings."""
    secret = "sk-ant-secret-key-value"
    cm.set_key(Provider.ANTHROPIC, secret)
    info = cm.list_config()
    info_str = json.dumps(info)
    assert secret not in info_str


def test_list_config_shows_key_presence(cm: ConfigManager) -> None:
    """list_config() reports key_configured=True for a stored key."""
    cm.set_key(Provider.ANTHROPIC, "any-key")
    info = cm.list_config()
    assert info["providers"]["anthropic"]["key_configured"] is True


def test_list_config_keychain_source(cm: ConfigManager) -> None:
    """list_config() reports key_source='keychain' for a keychain-stored key."""
    cm.set_key(Provider.OPENAI, "oai-key")
    info = cm.list_config()
    assert info["providers"]["openai"]["key_source"] == "keychain"


def test_list_config_env_source(
    cm: ConfigManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """list_config() reports key_source='env' for an environment-variable key."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    info = cm.list_config()
    assert info["providers"]["openai"]["key_source"] == "env"


def test_list_config_no_key_shows_not_configured(cm: ConfigManager) -> None:
    """list_config() reports key_configured=False when no key is present."""
    info = cm.list_config()
    assert info["providers"]["gemini"]["key_configured"] is False
    assert info["providers"]["gemini"]["key_source"] is None


def test_list_config_includes_active_provider(cm: ConfigManager) -> None:
    """list_config() includes the active_provider field."""
    cm.set_active_provider(Provider.OLLAMA)
    info = cm.list_config()
    assert info["active_provider"] == "ollama"


def test_list_config_includes_model(cm: ConfigManager) -> None:
    """list_config() includes the configured model for each provider."""
    cm.set_model(Provider.ANTHROPIC, "claude-haiku-4-5-20251001")
    info = cm.list_config()
    assert info["providers"]["anthropic"]["model"] == "claude-haiku-4-5-20251001"


def test_list_config_vault_path(cm: ConfigManager, tmp_path: Path) -> None:
    """list_config() includes vault_path when set."""
    vault = tmp_path / "my_vault"
    cm.set_vault_path(vault)
    info = cm.list_config()
    assert info["vault_path"] is not None
