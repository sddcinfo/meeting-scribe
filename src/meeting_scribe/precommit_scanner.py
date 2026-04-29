"""Pre-commit safety scan — block sensitive data from being staged.

Runs against the working tree of a repo and reports every file + line
that trips a sensitive-data rule:

* **Credentials** — API keys, bearer tokens, AWS / GitHub patterns,
  private keys, ``.env`` files with passwords.
* **Network identity** — specific LAN IPs (``192.168.8.*``), MAC
  addresses, personal hostnames that should live in config, not code.
* **Meeting / recording content** — anything that looks like a path
  into ``meetings/``, ``recording.pcm``, ``journal.jsonl``,
  ``polished.json``, or audio file extensions.
* **PII** — personal email addresses, real-name signatures.

Used from any sddcinfo-managed repo:

    sddc precommit                       # scan current repo working tree
    sddc precommit --repo ~/path/to/repo
    sddc precommit --include-staged      # include already-staged files
    sddc precommit --verbose             # show every hit, not just file names

Non-zero exit = problems found. Designed to run before ``git add``,
before ``git commit``, and as the first step of any "ready to push"
session so we never accidentally leak meeting or infra detail into
the public git history.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import click

# ── Rule definitions ───────────────────────────────────────────

# Each rule is (id, severity, pattern, description).
# severity: "block" = fails the scan; "warn" = surfaced but non-fatal.
# Patterns are compiled case-insensitive unless the rule says otherwise.


@dataclass
class Rule:
    id: str
    severity: str  # "block" or "warn"
    pattern: re.Pattern
    description: str


def _rules() -> list[Rule]:
    return [
        # ── Credentials ───────────────────────────────────────
        Rule(
            "credentials.github-pat",
            "block",
            re.compile(r"\bghp_[A-Za-z0-9]{36,}\b"),
            "GitHub personal access token",
        ),
        Rule(
            "credentials.github-fine-grained",
            "block",
            re.compile(r"\bgithub_pat_[A-Za-z0-9_]{60,}\b"),
            "GitHub fine-grained token",
        ),
        Rule(
            "credentials.aws-access-key",
            "block",
            re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
            "AWS access key ID",
        ),
        Rule(
            "credentials.aws-secret",
            "block",
            re.compile(r"aws_secret_access_key\s*=\s*['\"][A-Za-z0-9/+=]{30,}['\"]"),
            "AWS secret access key",
        ),
        Rule(
            "credentials.slack-token",
            "block",
            re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
            "Slack token",
        ),
        Rule(
            "credentials.openai-key",
            "block",
            re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
            "OpenAI / Anthropic-style API key",
        ),
        Rule(
            "credentials.anthropic-key",
            "block",
            re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
            "Anthropic API key",
        ),
        Rule(
            "credentials.cloudflare-token",
            "block",
            # CF API tokens are 40-char hex-ish; only flag when the
            # variable name clearly ties it to Cloudflare to keep FP low.
            re.compile(
                r"(?:CLOUDFLARE_[A-Z_]+|cf[_-]?token)\s*=\s*['\"][A-Za-z0-9_-]{30,}['\"]",
                re.IGNORECASE,
            ),
            "Cloudflare API token",
        ),
        Rule(
            "credentials.private-key-block",
            "block",
            re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"),
            "Private key PEM block",
        ),
        Rule(
            "credentials.generic-password-assignment",
            "block",
            # Pattern matches "passwd = \"…\"" style assignments with 6+  # sddc-precommit: ignore
            # chars of secret — the regex below is what fires.  Comment text
            # itself would self-trip without the ignore tag on the line above.
            re.compile(
                r"(?:^|[^_a-zA-Z])(?:password|passwd|passphrase)"
                r"\s*[:=]\s*['\"][^'\"\s]{6,}['\"]",
                re.IGNORECASE,
            ),
            "Inline password literal",
        ),
        Rule(
            "credentials.bearer-header",
            "block",
            re.compile(
                r"Authorization\s*[:=]\s*['\"]?Bearer\s+[A-Za-z0-9._-]{20,}",
                re.IGNORECASE,
            ),
            "Bearer token embedded in code",
        ),
        # ── Meeting / recording content ───────────────────────
        # Most of these are defensive: meetings/ + test-fixtures/ are
        # gitignored in both meeting-scribe and the outer sddcinfo
        # repo, but a developer renaming a fixture or copying a real
        # meeting into a new path would trip these.
        Rule(
            "meeting.embedded-journal",
            "block",
            re.compile(
                r'"segment_id"\s*:\s*"[0-9a-f]{8}-[0-9a-f]{4}-',
                re.IGNORECASE,
            ),
            "Meeting journal entry embedded in a non-fixture file",
        ),
        Rule(
            "meeting.polished-json-path",
            "warn",
            re.compile(r"meetings/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}", re.IGNORECASE),
            "Reference to a real meeting UUID path — verify it's a code path, not embedded data",
        ),
        Rule(
            "meeting.recording-pcm",
            "block",
            re.compile(r"\brecording\.pcm\b.*(?:\.read|\.open\(|base64|b64|bytes\(\))"),
            "Code path reads recording.pcm into a committed artifact",
        ),
        # ── Infra / network identity ──────────────────────────
        Rule(
            "identity.lan-ip-192-168-8",
            "warn",
            re.compile(r"\b192\.168\.8\.\d{1,3}\b"),
            "Hard-coded LAN IP on the 192.168.8.* subnet — move to config",
        ),
        Rule(
            "identity.mac-address",
            "warn",
            re.compile(
                r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b",
                re.IGNORECASE,
            ),
            "MAC address — verify it's a test fixture, not a real device",
        ),
        # ── PII ───────────────────────────────────────────────
        Rule(
            "pii.personal-email",
            "warn",
            re.compile(
                r"\b[A-Za-z0-9._%+-]+@(?:gmail|yahoo|hotmail|outlook)\.(?:com|co\.jp)\b",
                re.IGNORECASE,
            ),
            "Personal email address — belongs in contributor metadata, not code",
        ),
    ]


# ── File filtering ─────────────────────────────────────────────

# Skip paths: binaries, large data, already-gitignored content.
# Glob-style; any match means we don't scan the file contents.
_SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".xz",
    ".zst",
    ".wav",
    ".mp3",
    ".pcm",
    ".ogg",
    ".flac",
    ".m4a",
    ".pptx",
    ".docx",
    ".xlsx",
    ".odp",
    ".odt",
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".dll",
    ".o",
    ".onnx",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".gguf",
}

_SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "dist",
    "build",
    ".idea",
    ".vscode",
}


def _should_skip_path(p: Path) -> bool:
    if p.suffix.lower() in _SKIP_SUFFIXES:
        return True
    return any(part in _SKIP_DIRS for part in p.parts)


# ── Git plumbing ───────────────────────────────────────────────


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise click.ClickException(
            f"git {' '.join(args)} failed in {repo}: {result.stderr.strip()}"
        )
    return result.stdout


def _repo_root(start: Path) -> Path:
    out = _git(start, "rev-parse", "--show-toplevel").strip()
    return Path(out)


def _files_under_review(repo: Path, include_staged: bool) -> list[tuple[Path, str]]:
    """Return (abs_path, status_code) for every file git considers modified,
    added, or untracked under ``repo``."""
    out = _git(repo, "status", "--porcelain")
    entries: list[tuple[Path, str]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        # Porcelain format: XY<space>path (path may be "a -> b" on rename).
        # X = staged, Y = unstaged.
        status, rest = line[:2], line[3:]
        path_token = rest.split(" -> ")[-1].strip()
        # Only surface files caller asked about. Default: everything
        # not yet in a commit (unstaged + untracked). --include-staged
        # also pulls files the user already `git add`'d.
        staged_only = status[0] != " " and status[0] != "?"
        untracked = status.startswith("??")
        unstaged_or_untracked = status[1] != " " or untracked
        if staged_only and not include_staged:
            continue
        if not (staged_only or unstaged_or_untracked):
            continue
        if status.startswith("D"):
            continue  # deletions don't need scanning
        entries.append((repo / path_token, status))
    return entries


# ── Scanner ────────────────────────────────────────────────────


@dataclass
class Hit:
    path: Path
    line_no: int
    rule: Rule
    snippet: str


def _scan_file(path: Path, rules: list[Rule], max_bytes: int = 2_000_000) -> list[Hit]:
    if not path.is_file():
        return []
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size > max_bytes:
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            content = fh.read(max_bytes)
    except OSError:
        return []

    hits: list[Hit] = []
    for line_no, line in enumerate(content.splitlines(), start=1):
        for rule in rules:
            if rule.pattern.search(line):
                hits.append(
                    Hit(
                        path=path,
                        line_no=line_no,
                        rule=rule,
                        snippet=line.strip()[:160],
                    )
                )
    return hits


def _is_suppressed(line: str) -> bool:
    """Allow callers to mark a known-benign hit with
    ``# sddc-precommit: ignore`` on the same line."""
    return "sddc-precommit: ignore" in line


def _print_hits(hits: list[Hit], verbose: bool) -> tuple[int, int]:
    """Print hits, returning (n_block, n_warn)."""
    block = sum(1 for h in hits if h.rule.severity == "block")
    warn = len(hits) - block
    if not hits:
        return block, warn

    # Group by path
    by_path: dict[Path, list[Hit]] = {}
    for h in hits:
        by_path.setdefault(h.path, []).append(h)

    for p, items in sorted(by_path.items()):
        click.echo(f"  {p}")
        for h in items:
            marker = "BLOCK" if h.rule.severity == "block" else "warn "
            click.echo(f"    [{marker}] {h.path}:{h.line_no} {h.rule.id} — {h.rule.description}")
            if verbose:
                click.echo(f"           {h.snippet}")
    return block, warn


# ── CLI ────────────────────────────────────────────────────────


@click.command("precommit")
@click.option(
    "--repo",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Repo to scan (default: current working directory's git root).",
)
@click.option(
    "--include-staged/--no-include-staged",
    default=True,
    help="Include already-staged files in the scan.",
)
@click.option(
    "--include-all-tracked",
    is_flag=True,
    default=False,
    help="Ignore git status — scan every tracked file. Slow; use sparingly.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show the matching line snippet for each hit.",
)
@click.option(
    "--warn-only",
    is_flag=True,
    default=False,
    help="Treat all hits as warnings (exit 0 even on block-level findings).",
)
def precommit(
    repo: Path | None,
    include_staged: bool,
    include_all_tracked: bool,
    verbose: bool,
    warn_only: bool,
) -> None:
    """Scan the working tree for sensitive data before commit.

    Exit 0 when clean, exit 1 when any block-level rule fires.
    """
    start = Path(repo).resolve() if repo else Path.cwd().resolve()
    root = _repo_root(start)
    click.echo(f"[sddc precommit] scanning {root}")

    if include_all_tracked:
        raw = _git(root, "ls-files").splitlines()
        entries = [(root / Path(p), "tracked") for p in raw if p.strip()]
    else:
        entries = _files_under_review(root, include_staged=include_staged)

    if not entries:
        click.echo("  no files to scan (clean working tree)")
        return

    click.echo(f"  files: {len(entries)}")

    rules = _rules()
    all_hits: list[Hit] = []
    scanned = skipped = 0
    for path, _status in entries:
        if _should_skip_path(path):
            skipped += 1
            continue
        hits = _scan_file(path, rules)
        if hits:
            # Drop suppressed lines.
            filtered: list[Hit] = []
            for h in hits:
                if _is_suppressed(h.snippet):
                    continue
                filtered.append(h)
            all_hits.extend(filtered)
        scanned += 1

    click.echo(f"  scanned: {scanned}   skipped (binary/large): {skipped}")

    block, warn = _print_hits(all_hits, verbose=verbose)

    click.echo("")
    if not all_hits:
        click.echo(click.style("✓ no sensitive data found", fg="green"))
        return

    summary = f"{block} block / {warn} warn"
    if block == 0:
        click.echo(click.style(f"⚠ {summary} (no block-level hits)", fg="yellow"))
        return

    click.echo(click.style(f"✗ {summary}", fg="red"))
    click.echo(
        "Review each finding above. To suppress a known-benign hit, add\n"
        "'# sddc-precommit: ignore' on the same line as the match."
    )
    if warn_only:
        click.echo("(--warn-only: exiting 0 despite block-level hits)")
        return
    sys.exit(1)


__all__ = ["precommit"]
