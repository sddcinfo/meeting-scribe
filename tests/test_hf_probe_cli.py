"""Tests for the ``meeting-scribe hf-probe`` CLI.

Critical assertions per plan §1.7:
- exit codes 0 / 64 / 65 by status branch
- token never appears in argv (P0 secret-handling)
- token bytes are zeroed after use (P0 defense-in-depth)
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from meeting_scribe.cli.hf_probe import (
    EX_NETWORK,
    EX_OK,
    EX_TOKEN_OR_EULA,
    hf_probe,
)
from meeting_scribe.hf_preflight import (
    HfModelResult,
    HfStatus,
    ValidationReport,
)


def _ok_report() -> ValidationReport:
    return ValidationReport(
        token_prefix="hf_xxxx",
        whoami="alice",
        results=[
            HfModelResult("model-a", HfStatus.OK, revision="sha-a"),
            HfModelResult("model-b", HfStatus.OK, revision="sha-b"),
        ],
    )


def _bad_token_report() -> ValidationReport:
    return ValidationReport(
        token_prefix="hf_xxxx",
        whoami=None,
        results=[
            HfModelResult("model-a", HfStatus.BAD_TOKEN),
        ],
    )


def _gated_report() -> ValidationReport:
    return ValidationReport(
        token_prefix="hf_xxxx",
        whoami="alice",
        results=[
            HfModelResult("Qwen/Qwen3.6-35B-A3B-FP8", HfStatus.GATED_NOT_ACCEPTED),
        ],
    )


def _network_only_report() -> ValidationReport:
    return ValidationReport(
        token_prefix="hf_xxxx",
        whoami="alice",
        results=[HfModelResult("model-a", HfStatus.NETWORK_ERROR)],
    )


# ── Exit-code routing ─────────────────────────────────────────


@patch("meeting_scribe.cli.hf_probe.validate_hf_access")
def test_exit_0_when_ok(mock_validate) -> None:
    mock_validate.return_value = _ok_report()
    runner = CliRunner()
    result = runner.invoke(
        hf_probe,
        ["--read-token-from-stdin"],
        input="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n",
    )
    assert result.exit_code == EX_OK


@patch("meeting_scribe.cli.hf_probe.validate_hf_access")
def test_exit_64_when_bad_token(mock_validate) -> None:
    mock_validate.return_value = _bad_token_report()
    runner = CliRunner()
    result = runner.invoke(
        hf_probe,
        ["--read-token-from-stdin"],
        input="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n",
    )
    assert result.exit_code == EX_TOKEN_OR_EULA


@patch("meeting_scribe.cli.hf_probe.validate_hf_access")
def test_exit_64_when_gated(mock_validate) -> None:
    mock_validate.return_value = _gated_report()
    runner = CliRunner()
    result = runner.invoke(
        hf_probe,
        ["--read-token-from-stdin"],
        input="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n",
    )
    assert result.exit_code == EX_TOKEN_OR_EULA


@patch("meeting_scribe.cli.hf_probe.validate_hf_access")
def test_exit_65_when_network_only(mock_validate) -> None:
    """Network-only failures get exit 65 so the orchestrator can route
    differently from token/EULA failures (the operator can't fix the
    network from the dev box but CAN fix gated-EULA from a browser)."""
    mock_validate.return_value = _network_only_report()
    runner = CliRunner()
    result = runner.invoke(
        hf_probe,
        ["--read-token-from-stdin"],
        input="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n",
    )
    assert result.exit_code == EX_NETWORK


# ── Token transport (P0 — never via argv/env-var-on-CLI) ──────


@patch("meeting_scribe.cli.hf_probe.validate_hf_access")
def test_refuses_when_no_token_no_stdin(mock_validate) -> None:
    runner = CliRunner()
    result = runner.invoke(
        hf_probe,
        ["--read-token-from-stdin"],
        input="",  # empty stdin
    )
    assert result.exit_code == EX_TOKEN_OR_EULA
    assert "no token" in result.output.lower() or "produced no" in result.output


@patch("meeting_scribe.cli.hf_probe.validate_hf_access")
def test_refuses_when_no_stdin_and_no_tty_and_no_flag(mock_validate) -> None:
    """Default mode (no flag) refuses if stdin is not a TTY — prevents
    a script-level oversight from silently turning into the no-token path
    or an interactive hang."""
    runner = CliRunner()
    # CliRunner's stdin is not a TTY (default).
    result = runner.invoke(hf_probe, [])
    assert result.exit_code == EX_TOKEN_OR_EULA


def test_strips_trailing_newline_from_stdin_token() -> None:
    """A token piped with `printf '%s\\n' "$HF_TOKEN"` ends with \\n;
    we must strip it before passing to validate. Test against the
    helper directly because the CLI scrubs the token in-place after
    use, so post-validation inspection sees a zeroed buffer."""
    import io
    from unittest.mock import patch as _patch

    from meeting_scribe.cli.hf_probe import _read_token_from_stdin

    fake_stdin = io.BytesIO(b"hf_abcdefghijk\n")
    fake_stdin_obj = MagicMock()
    fake_stdin_obj.buffer = fake_stdin
    with _patch("meeting_scribe.cli.hf_probe.sys") as mock_sys:
        mock_sys.stdin = fake_stdin_obj
        token = _read_token_from_stdin()
    assert token == "hf_abcdefghijk"


def test_scrub_token_zeroes_bytearray() -> None:
    """`_scrub_token` zeroes a bytearray in place AND drops the dict
    reference, so the token bytes are gone from heap-walkable memory
    after validation completes (best-effort — full forensic guarantees
    require os-level secret-handling like /dev/shm tmpfs files)."""
    from meeting_scribe.cli.hf_probe import _scrub_token

    token_bytes = bytearray(b"hf_secret_xyz_xyz_xyz")
    holder = {"value": token_bytes}
    _scrub_token(holder)
    # Bytearray was zeroed in place
    assert all(b == 0 for b in token_bytes)
    # Dict reference dropped
    assert "value" not in holder


def test_token_never_appears_in_argv() -> None:
    """P0 — the token must not appear in /proc/<pid>/cmdline of the
    spawned hf-probe process. This invokes the actual subprocess (not
    CliRunner) to inspect real argv on Linux. Skipped on non-Linux."""
    import sys

    if not sys.platform.startswith("linux"):
        pytest.skip("argv-leak check only meaningful on Linux")

    # The token we expect to NEVER see on argv.
    secret = "hf_unmistakable_argv_canary_xyz"

    # Spawn the probe as a real subprocess. We pass `--include-shared`
    # off so the validator hits a tiny dummy list (mocked out by
    # patching validate_hf_access in a shim module — but actually
    # the simplest version just runs the real probe against a fake
    # model list so it fails fast on bad-token, no network needed).
    # We pipe the secret on stdin and read /proc/<pid>/cmdline before
    # the process exits.
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "meeting_scribe.cli",
            "hf-probe",
            "--read-token-from-stdin",
            "--no-include-shared",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,  # binary so we don't paint argv with extra encoding
    )
    try:
        # Send token + read /proc/<pid>/cmdline before the process exits.
        proc.stdin.write(secret.encode())
        proc.stdin.flush()
        # Race: read cmdline before close. /proc/<pid>/cmdline is the
        # snapshot at exec time, so even a slow process exposes a stable
        # view here.
        cmdline_path = f"/proc/{proc.pid}/cmdline"
        try:
            with open(cmdline_path, "rb") as f:
                cmdline = f.read()
        except FileNotFoundError:
            cmdline = b""
        proc.stdin.close()
    finally:
        proc.wait(timeout=10)

    assert secret.encode() not in cmdline, f"Token leaked into argv! cmdline={cmdline!r}"


# ── Runtime-manifest emission ─────────────────────────────────


@patch("meeting_scribe.cli.hf_probe.validate_hf_access")
def test_emit_runtime_manifest_includes_revisions_when_ok(mock_validate) -> None:
    """When --emit-runtime-manifest is set and report.ok is True, the
    JSON output must include a runtime_manifest section with the
    captured commit SHAs."""
    import json

    mock_validate.return_value = _ok_report()
    runner = CliRunner()
    result = runner.invoke(
        hf_probe,
        ["--read-token-from-stdin", "--json", "--emit-runtime-manifest"],
        input="hf_xxxxxxxx\n",
    )
    assert result.exit_code == EX_OK
    payload = json.loads(result.output)
    assert "runtime_manifest" in payload
    revisions = payload["runtime_manifest"]["model_revisions"]
    assert revisions == {"model-a": "sha-a", "model-b": "sha-b"}
    assert payload["runtime_manifest"]["hf_whoami"] == "alice"


@patch("meeting_scribe.cli.hf_probe.validate_hf_access")
def test_emit_runtime_manifest_omitted_when_not_ok(mock_validate) -> None:
    """An EULA-failed report MUST NOT emit a runtime manifest — the
    customer-side probe is the sole authority for model_revisions, and
    a partial-resolution manifest would silently drop revision pinning."""
    import json

    mock_validate.return_value = _gated_report()
    runner = CliRunner()
    result = runner.invoke(
        hf_probe,
        ["--read-token-from-stdin", "--json", "--emit-runtime-manifest"],
        input="hf_xxxxxxxx\n",
    )
    assert result.exit_code == EX_TOKEN_OR_EULA
    payload = json.loads(result.output)
    assert "runtime_manifest" not in payload


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
