"""Tests for cli/trust.py — export-cert, cert-fingerprint, known-appliances
record format, fingerprint-match abort path.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from meeting_scribe.cli import cli
from meeting_scribe.cli import trust as trust_mod
from meeting_scribe.cli import _common as common_mod


def _have_openssl() -> bool:
    return shutil.which("openssl") is not None


pytestmark = pytest.mark.skipif(
    not _have_openssl(),
    reason="openssl required to mint test cert",
)


def _gen_v10_cert(tmp: Path, *, appliance_id: str = "deadbeef00112233") -> Path:
    cert = tmp / "cert.pem"
    key = tmp / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "30",
            "-nodes",
            "-subj",
            rf"/CN=meeting-scribe\/{appliance_id}",
            "-addext",
            "subjectAltName = IP:10.42.0.1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return cert


def test_export_cert_writes_pem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """export-cert copies the leaf to the requested path verbatim."""
    fake_root = tmp_path
    (fake_root / "certs").mkdir()
    src_cert = _gen_v10_cert(fake_root / "certs")
    src_cert.rename(fake_root / "certs" / "cert.pem")

    monkeypatch.setattr(common_mod, "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(trust_mod, "PROJECT_ROOT", fake_root)

    out = tmp_path / "out.pem"
    result = CliRunner().invoke(cli, ["export-cert", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert out.read_bytes() == (fake_root / "certs" / "cert.pem").read_bytes()


def test_cert_fingerprint_prints_hex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_root = tmp_path
    (fake_root / "certs").mkdir()
    src_cert = _gen_v10_cert(fake_root / "certs")
    src_cert.rename(fake_root / "certs" / "cert.pem")
    monkeypatch.setattr(common_mod, "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(trust_mod, "PROJECT_ROOT", fake_root)

    result = CliRunner().invoke(cli, ["cert-fingerprint"])
    assert result.exit_code == 0, result.output
    fp = result.output.strip()
    assert len(fp) == 64
    assert all(c in "0123456789abcdef" for c in fp)


def test_export_cert_fails_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(common_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(trust_mod, "PROJECT_ROOT", tmp_path)
    result = CliRunner().invoke(cli, ["export-cert", str(tmp_path / "out.pem")])
    assert result.exit_code != 0
    assert "leaf cert missing" in result.output


def test_trust_install_requires_a_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """trust-install with no flags rejects with a usage error."""
    result = CliRunner().invoke(cli, ["trust-install"])
    assert result.exit_code != 0
    assert "specify --from-pem" in result.output


def test_trust_install_mutex(monkeypatch: pytest.MonkeyPatch) -> None:
    result = CliRunner().invoke(
        cli,
        ["trust-install", "--from-pem", "/dev/null", "--confirm-fingerprint"],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_trust_install_rejects_non_meeting_scribe_cert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cert whose CN doesn't start with meeting-scribe/ is rejected
    BEFORE reaching the platform-install dispatcher."""
    other = tmp_path / "other.pem"
    other_key = tmp_path / "other.key"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(other_key),
            "-out",
            str(other),
            "-days",
            "30",
            "-nodes",
            "-subj",
            "/CN=not-meeting-scribe",
            "-addext",
            "subjectAltName = IP:10.42.0.1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    result = CliRunner().invoke(cli, ["trust-install", "--from-pem", str(other)])
    assert result.exit_code != 0
    assert "expected CN to start with meeting-scribe/" in result.output


def test_appliance_info_emits_empty_record_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """appliance-info on a fresh laptop returns ``{}`` JSON, not an
    error."""
    monkeypatch.setattr(
        trust_mod,
        "_KNOWN_APPLIANCES_PATH",
        tmp_path / "known-appliances.json",
    )
    result = CliRunner().invoke(cli, ["appliance-info"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {}


def test_trust_uninstall_requires_args(monkeypatch: pytest.MonkeyPatch) -> None:
    result = CliRunner().invoke(cli, ["trust-uninstall"])
    assert result.exit_code != 0
    assert "specify <appliance_id> or --all" in result.output


def test_trust_uninstall_mutex(monkeypatch: pytest.MonkeyPatch) -> None:
    result = CliRunner().invoke(
        cli,
        ["trust-uninstall", "abc123", "--all"],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_trust_install_fingerprint_mismatch_aborts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the operator types a wrong fingerprint, the install aborts
    BEFORE writing into the trust store. No cert leaks into the OS DB.
    """
    fake_pem = "-----BEGIN CERTIFICATE-----\nFAKE\n-----END CERTIFICATE-----\n"
    cert_real = _gen_v10_cert(tmp_path)

    def fake_fetch(host: str, port: int = 443) -> str:
        return cert_real.read_text(encoding="utf-8")

    monkeypatch.setattr(trust_mod, "_fetch_leaf_pem", fake_fetch)
    # Force the install dispatcher to count invocations — it must NOT be
    # called.
    install_calls: list = []

    def fake_install(cert_path, *, appliance_id):
        install_calls.append((cert_path, appliance_id))
        return True, "fake"

    monkeypatch.setattr(trust_mod, "_platform_install", fake_install)

    result = CliRunner().invoke(
        cli,
        ["trust-install", "--confirm-fingerprint", "--host", "10.42.0.1"],
        input="00deadbeef\n",
    )
    assert result.exit_code != 0
    assert "fingerprint mismatch" in result.output
    assert install_calls == []
