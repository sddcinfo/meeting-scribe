"""Tests for `meeting_scribe.runtime.cert_check` and the per-device leaf
generation path in `cli/_common.py:_ensure_admin_tls_certs`.

Both halves of the v1.0 leaf-only TLS trust anchor are exercised:

  * `assert_cert_sans` — read-only sanity check; raises `CertConfigError` when
    a required SAN is missing, never mutates the cert path.
  * `_ensure_admin_tls_certs` — provisioning-time regenerator; produces a
    cert with `Subject CN=meeting-scribe/<appliance_id>`, SAN `IP:10.42.0.1`,
    and frozen appliance-id.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from meeting_scribe.runtime.cert_check import (
    CertConfigError,
    assert_cert_sans,
    get_leaf_fingerprint,
    get_subject_cn,
)


def _have_openssl() -> bool:
    return shutil.which("openssl") is not None


pytestmark = pytest.mark.skipif(
    not _have_openssl(),
    reason="openssl binary required for cert sanity tests",
)


def _gen_cert(
    tmp: Path,
    *,
    cn: str,
    san_arg: str | None,
) -> tuple[Path, Path]:
    """Helper: produce a self-signed cert+key pair under `tmp` matching the
    real production openssl invocation (sans the atomic-rename dance)."""
    cert = tmp / "cert.pem"
    key = tmp / "key.pem"
    # OpenSSL's -subj parser splits RDN on `/`, so escape any literal slash
    # in the CN value (matches the production helper in cli/_common.py).
    cn_escaped = cn.replace("/", r"\/")
    args = [
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
        f"/CN={cn_escaped}",
    ]
    if san_arg:
        args += [
            "-addext",
            f"subjectAltName = {san_arg}",
            "-addext",
            "basicConstraints = critical, CA:FALSE",
        ]
    subprocess.run(args, capture_output=True, text=True, check=True)
    return cert, key


def test_assert_cert_sans_missing_file_raises(tmp_path: Path) -> None:
    """Absent cert path → `CertConfigError` with remediation text."""
    with pytest.raises(CertConfigError) as exc_info:
        assert_cert_sans(
            tmp_path / "no-such.pem",
            required_ips={"10.42.0.1"},
        )
    assert "meeting-scribe setup" in str(exc_info.value)


def test_assert_cert_sans_legacy_cert_no_san_raises(tmp_path: Path) -> None:
    """Pre-cutover CN=meeting-scribe leaf with no SAN extension fails the
    sanity check — the runtime must abort startup so the operator regenerates.
    """
    cert, _ = _gen_cert(tmp_path, cn="meeting-scribe", san_arg=None)
    with pytest.raises(CertConfigError) as exc_info:
        assert_cert_sans(cert, required_ips={"10.42.0.1"})
    assert "missing IP SANs" in str(exc_info.value)
    assert "meeting-scribe setup" in str(exc_info.value)


def test_assert_cert_sans_v10_leaf_passes(tmp_path: Path) -> None:
    """v1.0-format leaf with the required AP IP SAN passes silently."""
    cert, _ = _gen_cert(
        tmp_path,
        cn="meeting-scribe/abc123",
        san_arg="IP:10.42.0.1",
    )
    # Should not raise.
    assert_cert_sans(cert, required_ips={"10.42.0.1"})


def test_assert_cert_sans_dns_san_required(tmp_path: Path) -> None:
    """A required DNS SAN that is not present triggers `missing DNS SANs`."""
    cert, _ = _gen_cert(
        tmp_path,
        cn="meeting-scribe/abc123",
        san_arg="IP:10.42.0.1",
    )
    with pytest.raises(CertConfigError) as exc_info:
        assert_cert_sans(
            cert,
            required_ips={"10.42.0.1"},
            required_dns={"meeting.local"},
        )
    assert "missing DNS SANs" in str(exc_info.value)


def test_get_leaf_fingerprint_format(tmp_path: Path) -> None:
    """Fingerprint output is lowercase hex, no colons, 64 chars."""
    cert, _ = _gen_cert(
        tmp_path,
        cn="meeting-scribe/abc123",
        san_arg="IP:10.42.0.1",
    )
    fp = get_leaf_fingerprint(cert)
    assert ":" not in fp
    assert fp.lower() == fp
    assert len(fp) == 64
    assert all(ch in "0123456789abcdef" for ch in fp)


def test_get_subject_cn_extracts_appliance_id(tmp_path: Path) -> None:
    """`get_subject_cn` returns the full CN string including the
    appliance-id suffix used by the bootstrap-page identity surface."""
    cert, _ = _gen_cert(
        tmp_path,
        cn="meeting-scribe/deadbeef00112233",
        san_arg="IP:10.42.0.1",
    )
    cn = get_subject_cn(cert)
    assert cn == "meeting-scribe/deadbeef00112233"


def test_ensure_admin_tls_certs_regenerates_legacy_cert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-cutover CN=meeting-scribe leaf without SAN is detected and
    rewritten with the appliance-id + IP SAN format.
    """
    fake_root = tmp_path
    (fake_root / "certs").mkdir()
    legacy_cert, legacy_key = _gen_cert(
        fake_root / "certs",
        cn="meeting-scribe",
        san_arg=None,
    )
    legacy_cert.rename(fake_root / "certs" / "cert.pem")
    legacy_key.rename(fake_root / "certs" / "key.pem")

    from meeting_scribe.cli import _common as common_mod

    monkeypatch.setattr(common_mod, "PROJECT_ROOT", fake_root)

    ok, detail = common_mod._ensure_admin_tls_certs()
    assert ok, detail
    assert "appliance_id=" in detail or "validated" in detail

    cert = fake_root / "certs" / "cert.pem"
    # New cert has the AP IP SAN.
    assert_cert_sans(cert, required_ips={"10.42.0.1"})
    # New cert's CN includes the appliance ID.
    cn = get_subject_cn(cert)
    assert cn.startswith("meeting-scribe/"), cn
    # The appliance-id file persists for re-runs.
    appliance_path = fake_root / "certs" / "appliance-id"
    assert appliance_path.exists()
    appliance_id = appliance_path.read_text().strip()
    assert cn == f"meeting-scribe/{appliance_id}"


def test_ensure_admin_tls_certs_idempotent_when_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A v1.0-format leaf that already passes the SAN check is left alone
    (same fingerprint after a second `_ensure_admin_tls_certs`)."""
    fake_root = tmp_path
    (fake_root / "certs").mkdir()
    from meeting_scribe.cli import _common as common_mod

    monkeypatch.setattr(common_mod, "PROJECT_ROOT", fake_root)
    ok1, _ = common_mod._ensure_admin_tls_certs()
    assert ok1
    fp_before = get_leaf_fingerprint(fake_root / "certs" / "cert.pem")
    appliance_before = (fake_root / "certs" / "appliance-id").read_text().strip()

    ok2, _ = common_mod._ensure_admin_tls_certs()
    assert ok2
    fp_after = get_leaf_fingerprint(fake_root / "certs" / "cert.pem")
    appliance_after = (fake_root / "certs" / "appliance-id").read_text().strip()

    assert fp_before == fp_after, "cert was regenerated unnecessarily"
    assert appliance_before == appliance_after


def test_ensure_admin_tls_certs_appliance_id_persists_across_regen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the cert is regenerated (e.g. it expired) the appliance ID
    stays the same — operators on the laptop's known-list keep matching."""
    fake_root = tmp_path
    (fake_root / "certs").mkdir()
    from meeting_scribe.cli import _common as common_mod

    monkeypatch.setattr(common_mod, "PROJECT_ROOT", fake_root)

    ok, _ = common_mod._ensure_admin_tls_certs()
    assert ok
    appliance_first = (fake_root / "certs" / "appliance-id").read_text().strip()
    fp_first = get_leaf_fingerprint(fake_root / "certs" / "cert.pem")

    # Force regeneration by deleting the cert + key but keeping appliance-id.
    os.unlink(fake_root / "certs" / "cert.pem")
    os.unlink(fake_root / "certs" / "key.pem")

    ok2, _ = common_mod._ensure_admin_tls_certs()
    assert ok2
    appliance_second = (fake_root / "certs" / "appliance-id").read_text().strip()
    fp_second = get_leaf_fingerprint(fake_root / "certs" / "cert.pem")

    assert appliance_first == appliance_second, "appliance-id rotated unexpectedly"
    # Different fingerprint because RSA keys are fresh — confirms the cert
    # was actually regenerated rather than reused from a cache.
    assert fp_first != fp_second
