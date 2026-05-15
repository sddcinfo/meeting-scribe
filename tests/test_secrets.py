"""``server_support.secrets.resolve_psk`` — age-store PSK resolution.

The helper shells out to ``scripts/decrypt-creds.sh`` from the sddcinfo
monorepo and parses ``KEY=value`` lines. Tests use a stub script to
avoid depending on the actual age key during CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meeting_scribe.server_support import secrets


@pytest.fixture
def stub_decrypt_script(tmp_path: Path, monkeypatch):
    """Create an executable bash stub at a per-test path.

    The fixture returns a callable that writes the given body and points
    ``secrets.DECRYPT_SCRIPT_PATH`` at it.
    """
    script = tmp_path / "decrypt-creds.sh"

    def _install(body: str, *, mode: int = 0o755) -> Path:
        script.write_text("#!/usr/bin/env bash\n" + body)
        script.chmod(mode)
        monkeypatch.setattr(secrets, "DECRYPT_SCRIPT_PATH", script)
        return script

    return _install


# ─── _parse_decrypted_env ──────────────────────────────────────


def test_parse_simple_kv_pairs() -> None:
    text = "FOO=bar\nBAZ=qux\n"
    assert secrets._parse_decrypted_env(text) == {"FOO": "bar", "BAZ": "qux"}


def test_parse_skips_blank_and_comment_lines() -> None:
    text = "\n# a comment\nFOO=bar\n  \n# another\nBAZ=qux\n"
    assert secrets._parse_decrypted_env(text) == {"FOO": "bar", "BAZ": "qux"}


def test_parse_strips_matching_quotes() -> None:
    text = "FOO=\"bar with spaces\"\nBAZ='qux'\nUNQ=raw\n"
    parsed = secrets._parse_decrypted_env(text)
    assert parsed == {"FOO": "bar with spaces", "BAZ": "qux", "UNQ": "raw"}


def test_parse_preserves_equals_in_value() -> None:
    text = "PSK=secret=with=equals\n"
    assert secrets._parse_decrypted_env(text) == {"PSK": "secret=with=equals"}


def test_parse_ignores_lines_without_equals() -> None:
    text = "FOO=bar\njunkline\nBAZ=qux\n"
    assert secrets._parse_decrypted_env(text) == {"FOO": "bar", "BAZ": "qux"}


def test_parse_handles_empty_string_value() -> None:
    text = "EMPTY=\nNONEMPTY=v\n"
    assert secrets._parse_decrypted_env(text) == {"EMPTY": "", "NONEMPTY": "v"}


# ─── resolve_psk ───────────────────────────────────────────────


def test_resolve_psk_returns_value(stub_decrypt_script) -> None:
    stub_decrypt_script("echo 'YUNOMOTOCHO_PSK=hunter2'\n")
    assert secrets.resolve_psk("YUNOMOTOCHO_PSK") == "hunter2"


def test_resolve_psk_raises_secret_not_found_for_missing(stub_decrypt_script) -> None:
    stub_decrypt_script("echo 'OTHER_KEY=value'\n")
    with pytest.raises(secrets.SecretNotFoundError):
        secrets.resolve_psk("YUNOMOTOCHO_PSK")


def test_resolve_psk_raises_decrypt_error_when_helper_rc_nonzero(stub_decrypt_script) -> None:
    stub_decrypt_script("echo 'no key' >&2\nexit 1\n")
    with pytest.raises(secrets.SecretDecryptError):
        secrets.resolve_psk("ANY_KEY")


def test_resolve_psk_raises_decrypt_error_when_script_missing(monkeypatch, tmp_path) -> None:
    missing = tmp_path / "nope.sh"
    monkeypatch.setattr(secrets, "DECRYPT_SCRIPT_PATH", missing)
    with pytest.raises(secrets.SecretDecryptError):
        secrets.resolve_psk("ANY_KEY")


def test_resolve_psk_rejects_invalid_ref(stub_decrypt_script) -> None:
    """Empty / non-str refs short-circuit without spawning the decrypt helper."""
    with pytest.raises(secrets.SecretNotFoundError):
        secrets.resolve_psk("")
    with pytest.raises(secrets.SecretNotFoundError):
        secrets.resolve_psk(None)  # type: ignore[arg-type]


def test_resolve_psk_does_not_leak_decrypted_stdout_into_error(
    stub_decrypt_script,
) -> None:
    """A failing decrypt helper must not leak its stdout into the exception."""
    # Stub that prints secrets on stdout AND fails. Exception text must
    # NOT contain anything from stdout.
    stub_decrypt_script(
        "echo 'YUNOMOTOCHO_PSK=should-not-appear-in-error'\necho 'decrypt failed' >&2\nexit 2\n"
    )
    with pytest.raises(secrets.SecretDecryptError) as exc_info:
        secrets.resolve_psk("YUNOMOTOCHO_PSK")
    assert "should-not-appear-in-error" not in str(exc_info.value)


def test_psk_ref_exists_true(stub_decrypt_script) -> None:
    stub_decrypt_script("echo 'YUNOMOTOCHO_PSK=hunter2'\n")
    assert secrets.psk_ref_exists("YUNOMOTOCHO_PSK") is True


def test_psk_ref_exists_false_for_missing(stub_decrypt_script) -> None:
    stub_decrypt_script("echo 'OTHER=x'\n")
    assert secrets.psk_ref_exists("YUNOMOTOCHO_PSK") is False


def test_psk_ref_exists_false_when_helper_fails(stub_decrypt_script) -> None:
    stub_decrypt_script("exit 1\n")
    assert secrets.psk_ref_exists("YUNOMOTOCHO_PSK") is False


def test_resolve_psk_honors_sddcinfo_root_env(monkeypatch, tmp_path: Path) -> None:
    """SDDCINFO_ROOT env should locate scripts/decrypt-creds.sh."""
    root = tmp_path / "alt-root"
    (root / "scripts").mkdir(parents=True)
    script = root / "scripts" / "decrypt-creds.sh"
    script.write_text("#!/usr/bin/env bash\necho 'KEY_FROM_ALT_ROOT=v'\n")
    script.chmod(0o755)
    monkeypatch.setattr(secrets, "DECRYPT_SCRIPT_PATH", None)
    monkeypatch.setenv("SDDCINFO_ROOT", str(root))
    assert secrets.resolve_psk("KEY_FROM_ALT_ROOT") == "v"
