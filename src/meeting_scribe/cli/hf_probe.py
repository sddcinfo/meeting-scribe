"""``meeting-scribe hf-probe`` — validate HF token + EULA acceptance.

Designed to be invoked over SSH from the dev-box orchestrator
(`sddc gb10 onboard`'s stage 2.5 remote check). The token must NEVER
appear in argv or in the SSH command line — it travels via stdin
only. See plans/steady-plotting-eich.md §1.4.

Usage:
    meeting-scribe hf-probe --json --read-token-from-stdin   (orchestrator)
    meeting-scribe hf-probe                                   (interactive)
    meeting-scribe hf-probe --emit-runtime-manifest --json   (stage 2.5)

Exit codes (sysexits-aligned):
    0   report.ok — every model accessible.
    64  EX_USAGE — at least one BAD_TOKEN / GATED_NOT_ACCEPTED / NOT_FOUND row.
    65  EX_DATAERR — has_only_network_failures (transient) — the operator
        sees the same "fix DNS / proxy CA / firewall" remediation as a
        full failure but the orchestrator can distinguish the two.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import click

from meeting_scribe.cli import cli
from meeting_scribe.cli._common import PROJECT_ROOT
from meeting_scribe.hf_preflight import (
    ValidationReport,
    validate_hf_access,
)
from meeting_scribe.recipes import all_model_ids

# Exit codes — must match sysexits.h so an SSH wrapper can route by rc.
EX_OK = 0
EX_TOKEN_OR_EULA = 64
EX_NETWORK = 65


def _read_token_from_stdin(max_bytes: int = 4096) -> str:
    """Read up to `max_bytes` from stdin until newline. Strips trailing
    \\n / \\r\\n. Refuses to read more than `max_bytes` to bound the
    surface for a misbehaving caller."""
    chunks: list[bytes] = []
    total = 0
    raw = sys.stdin.buffer
    while total < max_bytes:
        b = raw.read(1)
        if not b:
            break
        if b == b"\n":
            break
        chunks.append(b)
        total += 1
    return b"".join(chunks).decode("utf-8", errors="replace").rstrip("\r")


def _read_token_from_env_file() -> str:
    """Best-effort fallback: read `HF_TOKEN=…` from `.env` next to the
    project root. Used only when stdin is a TTY (interactive operator)."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return ""
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def _scrub_token(holder: dict, key: str = "value") -> None:
    """Best-effort scrub: overwrite a bytearray in place, then drop it.

    Python `str` is immutable and the only way to overwrite its buffer
    in place (`ctypes.memmove(id(s), …)`) corrupts Python's internal
    state when the same string is later reached by GC — observed as a
    SIGSEGV in `logging.handlers` during a follow-up `importlib`
    import (2026-05-01). So instead, the caller is expected to keep
    the token in a `bytearray` (mutable, no interning, no GC trap)
    inside a dict; this function zeroes the bytearray and drops the
    reference. If the caller passes a `str`, only the dict reference
    can be dropped — the str buffer relies on GC to reclaim, which is
    the same posture as `del token`.
    """
    val = holder.pop(key, None)
    if isinstance(val, bytearray):
        for i in range(len(val)):
            val[i] = 0


def _emit_report_json(report: ValidationReport, *, runtime_manifest: bool) -> str:
    """Serialize the report (and optionally a RuntimeManifest skeleton)
    to JSON. The orchestrator parses this on stdout."""
    payload: dict = {
        "token_prefix": report.token_prefix,
        "whoami": report.whoami,
        "ok": report.ok,
        "has_only_network_failures": report.has_only_network_failures,
        "results": [
            {
                "model_id": r.model_id,
                "status": r.status.value,
                "detail": r.detail,
                "url": r.url,
                "revision": r.revision,
            }
            for r in report.results
        ],
    }
    if runtime_manifest and report.ok:
        # Build the RuntimeManifest pieces the orchestrator needs.
        # `build_manifest_sha` is filled in by the orchestrator after
        # capture; this side only knows model_revisions + whoami.
        payload["runtime_manifest"] = {
            "schema_version": 1,
            "resolved_at": datetime.now(UTC).isoformat(),
            "model_revisions": {r.model_id: r.revision for r in report.results},
            "hf_whoami": report.whoami or "",
        }
    return json.dumps(payload, indent=2, sort_keys=True)


@cli.command("hf-probe")
@click.option(
    "--read-token-from-stdin",
    is_flag=True,
    default=False,
    help="Read the HF token from stdin (one line, ≤4 KB). Required for "
    "orchestrator use — never accepts the token via argv or env.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit a JSON ValidationReport on stdout. Default emits human-rendered text on stderr.",
)
@click.option(
    "--emit-runtime-manifest",
    is_flag=True,
    default=False,
    help="When the report is ok, include a RuntimeManifest skeleton "
    "(model_revisions resolved from HF) in the JSON output. Used "
    "by `sddc gb10 onboard` stage 2.5 to capture the customer-side "
    "model SHAs for the two-manifest model.",
)
@click.option(
    "--include-shared/--no-include-shared",
    default=True,
    show_default=True,
    help="Include the autosre-shared translation model. The customer-install "
    "flow always pulls this onto the device, so the gate must check it.",
)
def hf_probe(
    read_token_from_stdin: bool,
    as_json: bool,
    emit_runtime_manifest: bool,
    include_shared: bool,
) -> None:
    """Validate HF_TOKEN against every gated model meeting-scribe needs.

    Emits a ValidationReport (human text or JSON) and exits with a
    sysexits-aligned code so a remote caller can route by rc:
    0 ok, 64 token/EULA failure, 65 network-only failure.
    """
    # --- Token transport ---
    # Held in a dict so we can pop+scrub the bytearray (when it came
    # from stdin) AFTER validation completes, without leaving a long-
    # lived str reference around. See `_scrub_token` for why we don't
    # use ctypes memmove on str buffers.
    holder: dict = {}
    if read_token_from_stdin:
        holder["value"] = _read_token_from_stdin()
        if not holder["value"]:
            click.echo(
                "ERROR: --read-token-from-stdin set but stdin produced no "
                "token. Pipe the token on stdin, e.g.:\n"
                "    printf '%s' \"$HF_TOKEN\" | meeting-scribe hf-probe ...",
                err=True,
            )
            sys.exit(EX_TOKEN_OR_EULA)
    elif sys.stdin.isatty():
        # Interactive operator path: read from .env (the canonical store).
        holder["value"] = _read_token_from_env_file()
        if not holder["value"]:
            click.echo(
                "ERROR: no HF_TOKEN found in .env and stdin is a TTY.\n"
                "Set HF_TOKEN via `meeting-scribe setup` or pipe via\n"
                "    printf '%s' \"$HF_TOKEN\" | meeting-scribe hf-probe --read-token-from-stdin",
                err=True,
            )
            sys.exit(EX_TOKEN_OR_EULA)
    else:
        click.echo(
            "ERROR: stdin is not a TTY and --read-token-from-stdin was not set.\n"
            "Pass --read-token-from-stdin and pipe the token on stdin.",
            err=True,
        )
        sys.exit(EX_TOKEN_OR_EULA)

    # --- Validation ---
    try:
        model_ids = all_model_ids(include_shared=include_shared)
        if not model_ids:
            click.echo("ERROR: no model recipes found", err=True)
            sys.exit(EX_TOKEN_OR_EULA)
        report = validate_hf_access(holder["value"], model_ids)
    finally:
        _scrub_token(holder)

    # --- Output ---
    if as_json:
        click.echo(_emit_report_json(report, runtime_manifest=emit_runtime_manifest))
    else:
        click.echo(report.render(), err=True)

    # --- Exit code ---
    if report.ok:
        sys.exit(EX_OK)
    if report.has_only_network_failures:
        sys.exit(EX_NETWORK)
    # At least one BAD_TOKEN / GATED_NOT_ACCEPTED / NOT_FOUND.
    sys.exit(EX_TOKEN_OR_EULA)
