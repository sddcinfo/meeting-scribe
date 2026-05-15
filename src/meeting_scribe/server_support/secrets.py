"""Resolve secret references at the boundary of running a subprocess.

The WAN feature stores upstream-WiFi PSKs by *reference* in
``settings.json`` (key ``psk_ref``) and resolves the plaintext only at
the moment we hand it to ``nmcli``. The encrypted-at-rest store is the
sddcinfo monorepo's age-encrypted credentials file (``.credentials.env.age``).
We invoke ``scripts/decrypt-creds.sh`` once per ``wan_up`` call to
recover the plaintext.

This module is deliberately small and pure:
- No globals retain the PSK
- No log statement ever sees the value
- ``repr()`` of any object never includes a PSK
- The plaintext string crosses one frame (this module) and one argv
  (``nmcli con add ... wifi-sec.psk <psk>``) and then nothing here
  references it

Honest residual exposure: NetworkManager writes the PSK into
``/etc/NetworkManager/system-connections/<name>.nmconnection`` (mode
0600, root-only) until the profile is deleted. See
``docs/plans/wifi-wan-gateway.md`` Synthesis for the v1 trade-off.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# Default location of the sddcinfo monorepo's decrypt helper. Overridable
# at runtime via ``SDDCINFO_ROOT`` env var (matches the existing
# convention used by the shared AI-hooks Bash guard).
_DEFAULT_SDDCINFO_ROOT = Path.home() / "sddcinfo"

# Test seam: tests monkeypatch this to point at a stub script.
DECRYPT_SCRIPT_PATH: Path | None = None


class SecretNotFoundError(KeyError):
    """``psk_ref`` does not exist in the decrypted credentials store."""


class SecretDecryptError(RuntimeError):
    """``scripts/decrypt-creds.sh`` failed (missing age key, bad file, etc.)."""


def _resolve_script_path() -> Path:
    """Locate ``scripts/decrypt-creds.sh`` from env or the default root."""
    if DECRYPT_SCRIPT_PATH is not None:
        return DECRYPT_SCRIPT_PATH
    root_env = os.environ.get("SDDCINFO_ROOT")
    root = Path(root_env) if root_env else _DEFAULT_SDDCINFO_ROOT
    return root / "scripts" / "decrypt-creds.sh"


def _parse_decrypted_env(text: str) -> dict[str, str]:
    """Parse ``KEY=value`` lines from the decrypted env file.

    Tolerates blank lines and ``#`` comments. Values may be unquoted,
    single-quoted, or double-quoted; quotes are stripped. Whitespace
    around ``=`` is not allowed (matches the bash sourcing semantics of
    the source file).
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        # Strip surrounding matching quotes; leave unmatched alone.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        out[key] = value
    return out


def resolve_psk(psk_ref: str) -> str:
    """Return the plaintext PSK for ``psk_ref`` from the age store.

    Raises ``SecretNotFoundError`` if the key is absent;
    ``SecretDecryptError`` if the decrypt subprocess fails.
    The returned string is the caller's to dispose; this function
    retains no reference to it after returning.
    """
    if not psk_ref or not isinstance(psk_ref, str):
        raise SecretNotFoundError(f"invalid psk_ref: {psk_ref!r}")

    script = _resolve_script_path()
    if not script.is_file():
        raise SecretDecryptError(f"decrypt helper not found at {script}")

    try:
        proc = subprocess.run(
            ["bash", str(script)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        raise SecretDecryptError(f"decrypt helper failed to run: {exc}") from exc

    if proc.returncode != 0:
        # stderr may contain useful diagnostics ("No age key at ..."); pass through
        # WITHOUT including any decrypted stdout in the error message.
        raise SecretDecryptError(
            f"decrypt helper rc={proc.returncode}: {proc.stderr.strip()[:200]}"
        )

    env_map = _parse_decrypted_env(proc.stdout)
    if psk_ref not in env_map:
        raise SecretNotFoundError(f"psk_ref {psk_ref!r} not present in credentials store")
    return env_map[psk_ref]


def psk_ref_exists(psk_ref: str) -> bool:
    """Cheap presence check — used by the CLI's ``profiles add`` dry-validate.

    Returns True iff the decrypt helper succeeds AND the key is present.
    Never raises; any failure mode returns False.
    """
    try:
        resolve_psk(psk_ref)
        return True
    except SecretNotFoundError, SecretDecryptError:
        return False
