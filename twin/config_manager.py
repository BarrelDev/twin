"""Encrypted keychain and configuration management for Twin.

Stores API keys in an AES-256-GCM encrypted file (~/.twin/keychain.enc).
Non-sensitive config is stored as plaintext JSON (~/.twin/config.json).
The encryption key is derived from username + machine ID via PBKDF2,
making the keychain non-portable across machines by design.
"""

import json
import os
import platform
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

from twin.config import Provider

_PBKDF2_ITERATIONS = 480_000
_KEY_LENGTH = 32
_NONCE_LENGTH = 12

# Environment variable names per provider (None = no key needed).
_ENV_VARS: dict[Provider, str | None] = {
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.GEMINI: "GEMINI_API_KEY",
    Provider.OPENROUTER: "OPENROUTER_API_KEY",
    Provider.OLLAMA: None,
}


def _get_machine_id() -> str:
    """
    Read a stable, machine-specific identifier.

    Uses platform-specific sources: Windows registry MachineGuid,
    macOS IOPlatformUUID, or Linux /etc/machine-id.

    Returns:
        Machine ID string unique to this host.

    Raises:
        RuntimeError: If the machine ID cannot be determined on this platform.
    """
    system = platform.system()
    if system == "Windows":
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Cryptography",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "MachineGuid")
            return str(value)

    if system == "Darwin":
        result = subprocess.run(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            if "IOPlatformUUID" in line:
                # Line format: "IOPlatformUUID" = "XXXXXXXX-..."
                return line.split('"')[-2]
        raise RuntimeError("Could not find IOPlatformUUID in ioreg output")

    # Linux (and other POSIX)
    for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        if candidate.exists():
            return candidate.read_text().strip()

    raise RuntimeError(
        "Cannot determine machine ID: /etc/machine-id not found. "
        "Install dbus or create /etc/machine-id manually."
    )


def _derive_key(machine_id: str) -> bytes:
    """
    Derive a 32-byte AES-256 key from username + machine ID via PBKDF2-SHA256.

    The salt is a deterministic hash of the machine ID so the same key is
    reproduced on every run without storing the salt separately.

    Args:
        machine_id: Platform-specific machine identifier.

    Returns:
        32-byte encryption key.
    """
    username = os.environ.get("USERNAME") or os.environ.get("USER") or "twin"
    password = f"{username}:{machine_id}".encode()
    salt = sha256(machine_id.encode()).digest()[:16]

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=_KEY_LENGTH,
        salt=salt,
        iterations=_PBKDF2_ITERATIONS,
    )
    return kdf.derive(password)


class ConfigManager:
    """
    Manages encrypted API key storage and non-sensitive Twin configuration.

    Keys are stored in AES-256-GCM encrypted form at <data_dir>/keychain.enc.
    Non-sensitive settings (active provider, default models, vault path) are
    stored as plaintext JSON at <data_dir>/config.json.

    The keychain is machine-bound: the encryption key is derived from the
    current user's username and the platform machine ID. A keychain file
    copied to another machine cannot be decrypted.
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        _machine_id: str | None = None,
    ) -> None:
        """
        Initialize the config manager.

        Args:
            data_dir: Directory for config and keychain files. Defaults to ~/.twin.
            _machine_id: Override machine ID for testing. Not for production use.
        """
        self._data_dir = data_dir or Path.home() / ".twin"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._keychain_path = self._data_dir / "keychain.enc"
        self._config_path = self._data_dir / "config.json"
        self._machine_id = _machine_id
        self._cached_key: bytes | None = None

    def _get_key(self) -> bytes:
        """Derive and cache the encryption key for this session."""
        if self._cached_key is None:
            machine_id = self._machine_id or _get_machine_id()
            self._cached_key = _derive_key(machine_id)
        return self._cached_key

    def _load_keychain(self) -> dict[str, str]:
        """Decrypt and return the keychain, or an empty dict if missing/corrupt."""
        if not self._keychain_path.exists():
            return {}
        raw = self._keychain_path.read_bytes()
        if len(raw) <= _NONCE_LENGTH:
            return {}
        try:
            aesgcm = AESGCM(self._get_key())
            plaintext = aesgcm.decrypt(raw[:_NONCE_LENGTH], raw[_NONCE_LENGTH:], None)
            return json.loads(plaintext.decode())
        except Exception:
            return {}

    def _save_keychain(self, data: dict[str, str]) -> None:
        """Encrypt and persist the keychain to disk."""
        plaintext = json.dumps(data).encode()
        nonce = os.urandom(_NONCE_LENGTH)
        aesgcm = AESGCM(self._get_key())
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        self._keychain_path.write_bytes(nonce + ciphertext)

    def _load_config(self) -> dict[str, Any]:
        """Load non-sensitive config from JSON, returning an empty dict if missing."""
        if not self._config_path.exists():
            return {}
        try:
            return json.loads(self._config_path.read_text())
        except json.JSONDecodeError:
            return {}

    def _save_config(self, data: dict[str, Any]) -> None:
        """Persist non-sensitive config as formatted JSON."""
        self._config_path.write_text(json.dumps(data, indent=2))

    # ── API Key management ──────────────────────────────────────────────────

    def set_key(self, provider: Provider, api_key: str) -> None:
        """
        Encrypt and store an API key for a provider.

        Args:
            provider: The LLM provider.
            api_key: The API key to store. Never written to disk in plaintext.
        """
        keychain = self._load_keychain()
        keychain[provider.value] = api_key
        self._save_keychain(keychain)

    def get_key(self, provider: Provider) -> str | None:
        """
        Retrieve a stored API key for a provider from the keychain.

        Args:
            provider: The LLM provider.

        Returns:
            Decrypted API key string, or None if not present in keychain.
        """
        return self._load_keychain().get(provider.value)

    def remove_key(self, provider: Provider) -> None:
        """
        Remove a provider's API key from the keychain.

        Args:
            provider: The provider whose key should be removed.
        """
        keychain = self._load_keychain()
        keychain.pop(provider.value, None)
        self._save_keychain(keychain)

    def resolve_api_key(self, provider: Provider) -> str:
        """
        Resolve the API key for a provider using the documented priority order.

        Resolution order: keychain → environment variable → descriptive error.
        Ollama requires no key and always returns an empty string.

        Args:
            provider: The LLM provider to resolve a key for.

        Returns:
            API key string (empty string for Ollama).

        Raises:
            KeyError: If no key is found, with an onboarding message.
        """
        if provider == Provider.OLLAMA:
            return ""

        key = self.get_key(provider)
        if key:
            return key

        env_var = _ENV_VARS.get(provider)
        if env_var:
            key = os.environ.get(env_var)
            if key:
                return key

        env_hint = f"Or set {env_var} in your environment.\n" if env_var else ""
        raise KeyError(
            f"No API key found for {provider.value}.\n"
            f"Run: twin config set-key\n"
            f"{env_hint}"
            f"To use a local model with no API key: twin config set-provider ollama"
        )

    # ── Provider / model config ─────────────────────────────────────────────

    def set_active_provider(self, provider: Provider) -> None:
        """
        Set the active LLM provider in config.json.

        Args:
            provider: The provider to make active.
        """
        config = self._load_config()
        config["active_provider"] = provider.value
        self._save_config(config)

    def get_active_provider(self) -> Provider:
        """
        Return the active provider using the documented resolution order.

        Resolution order: config.json → TWIN_PROVIDER env var → Anthropic default.

        Returns:
            Active Provider enum value.
        """
        config = self._load_config()
        raw = (
            config.get("active_provider")
            or os.environ.get("TWIN_PROVIDER")
            or Provider.ANTHROPIC.value
        )
        try:
            return Provider(raw)
        except ValueError:
            return Provider.ANTHROPIC

    def set_model(self, provider: Provider, model: str) -> None:
        """
        Set the default model for a provider.

        Args:
            provider: The LLM provider.
            model: Model identifier to use as default for this provider.
        """
        config = self._load_config()
        config.setdefault("models", {})[provider.value] = model
        self._save_config(config)

    def get_model(self, provider: Provider) -> str | None:
        """
        Return the configured default model for a provider.

        Args:
            provider: The LLM provider.

        Returns:
            Model identifier string, or None if not configured.
        """
        return self._load_config().get("models", {}).get(provider.value)

    # ── Vault path ──────────────────────────────────────────────────────────

    def set_vault_path(self, path: Path) -> None:
        """
        Persist the Obsidian vault path in config.json.

        Args:
            path: Absolute path to the Obsidian vault root.
        """
        config = self._load_config()
        config["vault_path"] = str(path.resolve())
        self._save_config(config)

    def get_vault_path(self) -> Path | None:
        """
        Return the configured Obsidian vault path.

        Returns:
            Path to the vault root, or None if not configured.
        """
        raw = self._load_config().get("vault_path")
        return Path(raw) if raw else None

    # ── Status display ──────────────────────────────────────────────────────

    def list_config(self) -> dict[str, Any]:
        """
        Return a safe summary of current configuration.

        Never includes API key values — only indicates whether a key is present
        and where it was found (keychain or environment variable).

        Returns:
            Dict with active_provider, per-provider status, and vault_path.
        """
        config = self._load_config()
        keychain = self._load_keychain()

        providers: dict[str, dict[str, Any]] = {}
        for provider in Provider:
            in_keychain = provider.value in keychain
            env_var = _ENV_VARS.get(provider)
            in_env = bool(env_var and os.environ.get(env_var))
            providers[provider.value] = {
                "key_configured": in_keychain or in_env,
                "key_source": (
                    "keychain" if in_keychain
                    else ("env" if in_env else None)
                ),
                "model": config.get("models", {}).get(provider.value),
            }

        return {
            "active_provider": config.get("active_provider", Provider.ANTHROPIC.value),
            "providers": providers,
            "vault_path": config.get("vault_path"),
        }
