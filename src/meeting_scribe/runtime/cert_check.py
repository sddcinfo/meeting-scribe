"""Read-only TLS leaf cert sanity check.

Runs immediately before ``uvicorn.Config(...)`` builds the SSL context in
``server.main()`` (and from ``meeting-scribe doctor``). Verifies the cert at
``certfile`` carries every required Subject Alternative Name (SAN) entry — IP
literals and DNS names — and raises a ``CertConfigError`` with an actionable
remediation message if any are missing.

The function NEVER mutates the cert path. Cert (re)generation is a
provisioning-time operation owned by ``meeting-scribe setup`` (root); the
runtime simply aborts startup with a clear hint when the cert is wrong.

This is one half of the v1.0 leaf-only TLS trust anchor. The other half lives
in ``scripts/setup_certs.py`` (root) which produces the cert in the first
place.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


class CertConfigError(RuntimeError):
    """Cert exists but its SANs don't match the v1.0 trust anchor.

    Raised from ``assert_cert_sans``. The message is operator-facing and
    always names the remediation command.
    """


def _run_openssl(args: list[str]) -> str:
    """Invoke ``openssl`` with the given args, returning stdout text.

    Raises ``CertConfigError`` if openssl is absent on PATH or if the call
    fails — the caller treats both as a hard cert-config problem.
    """
    if not shutil.which("openssl"):
        raise CertConfigError(
            "openssl not on PATH — install via `apt install openssl` and "
            "re-run `meeting-scribe setup` as root."
        )
    proc = subprocess.run(
        ["openssl", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise CertConfigError(
            f"openssl {args[0]} failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[:200]}"
        )
    return proc.stdout


def _extract_sans(certfile: Path) -> tuple[set[str], set[str]]:
    """Return (ip_sans, dns_sans) extracted from ``certfile``'s subjectAltName.

    Uses ``openssl x509 -ext subjectAltName -noout`` which prints the
    extension in the form::

        X509v3 Subject Alternative Name:
            IP Address:10.42.0.1, DNS:meeting-scribe.local

    Lines lacking the extension yield empty sets; callers compare against
    their required-set and decide.
    """
    text = _run_openssl(["x509", "-in", str(certfile), "-noout", "-ext", "subjectAltName"])
    ips: set[str] = set()
    dns: set[str] = set()
    for entry in re.findall(r"(IP\s*Address|DNS):\s*([^,\s]+)", text):
        kind, value = entry
        if kind.startswith("IP"):
            ips.add(value)
        else:
            dns.add(value)
    return ips, dns


def get_leaf_fingerprint(certfile: Path) -> str:
    """Return the lowercase, colon-free SHA-256 fingerprint of the cert.

    Used by ``meeting-scribe cert-fingerprint`` and the bootstrap form's
    appliance-identity surface (Plan §A.5 / TLS Trust Anchor section).
    """
    text = _run_openssl(["x509", "-in", str(certfile), "-noout", "-fingerprint", "-sha256"])
    # Output: "sha256 Fingerprint=AB:CD:..."
    match = re.search(r"=([0-9A-F:]+)\s*$", text.strip(), re.IGNORECASE)
    if not match:
        raise CertConfigError(
            f"openssl returned unparseable fingerprint output: {text.strip()[:120]!r}"
        )
    return match.group(1).replace(":", "").lower()


def get_subject_cn(certfile: Path) -> str:
    """Return the cert's Subject CommonName (used for the appliance ID).

    For v1.0 leaves the CN is ``meeting-scribe/<appliance_id>``. Older
    pre-cutover leaves with a bare ``meeting-scribe`` CN return that string;
    the caller decides whether that is acceptable.
    """
    text = _run_openssl(["x509", "-in", str(certfile), "-noout", "-subject"])
    # Output: "subject= CN=meeting-scribe/abcdef0123456789"
    match = re.search(r"CN\s*=\s*(\S.*)", text.strip())
    if not match:
        raise CertConfigError(
            f"openssl returned unparseable subject output: {text.strip()[:120]!r}"
        )
    return match.group(1).strip()


def assert_cert_sans(
    certfile: Path,
    *,
    required_ips: set[str],
    required_dns: set[str] = frozenset(),
) -> None:
    """Read-only check: cert at ``certfile`` carries every required SAN.

    Raises ``CertConfigError`` with an operator-actionable message if any
    required IP or DNS SAN is missing, OR if the cert file itself can't be
    read. Never mutates the cert path; cert regeneration is a provisioning
    operation, not a runtime one.

    Called from ``server.main()`` before ``uvicorn.Config(...)`` builds the
    SSL context, and from ``meeting-scribe doctor``. The two callers share
    the same remediation message so the operator sees one consistent
    instruction.
    """
    if not certfile.exists():
        raise CertConfigError(
            f"TLS cert missing at {certfile}. "
            "Run `sudo meeting-scribe setup` to generate the appliance leaf."
        )
    ips, dns = _extract_sans(certfile)
    missing_ips = required_ips - ips
    missing_dns = required_dns - dns
    if missing_ips or missing_dns:
        details: list[str] = []
        if missing_ips:
            details.append(f"missing IP SANs: {sorted(missing_ips)}")
        if missing_dns:
            details.append(f"missing DNS SANs: {sorted(missing_dns)}")
        raise CertConfigError(
            f"TLS cert at {certfile} has wrong SANs ({'; '.join(details)}). "
            "Re-run `sudo meeting-scribe setup` to regenerate the appliance "
            "leaf, then restart meeting-scribe."
        )
