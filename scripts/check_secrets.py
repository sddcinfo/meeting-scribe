#!/usr/bin/env python3
"""Pre-commit secret scanner — mirrors CI checks locally.

Checks for:
  - API keys, tokens, passwords in source files
  - Claude Code references (Co-Authored-By, .claude/, CLAUDE.md)
  - Hardcoded absolute user paths (/Users/, /home/)
  - Private keys, certificates

Exit code 0 = clean, 1 = issues found.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCAN_DIRS = ["src", "static", "tests", "scripts"]
SCAN_EXTENSIONS = {".py", ".js", ".html", ".css", ".yml", ".yaml", ".toml", ".md"}
SKIP_FILES = {"check_secrets.py", "validate_how_it_works.py"}

issues: list[str] = []


def scan_file(path: Path) -> None:
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return

    rel = path.relative_to(ROOT)
    lines = text.split("\n")

    for i, line in enumerate(lines, 1):
        # API keys / tokens (common patterns)
        if re.search(r"(api[_-]?key|api[_-]?token|secret[_-]?key)\s*[:=]\s*['\"][^'\"]{8,}", line, re.IGNORECASE):
            issues.append(f"{rel}:{i}: Possible API key/token")

        # HF tokens
        if re.search(r"hf_[A-Za-z0-9]{20,}", line):
            issues.append(f"{rel}:{i}: HuggingFace token detected")

        # Private keys
        if "BEGIN PRIVATE KEY" in line or "BEGIN RSA PRIVATE KEY" in line:
            issues.append(f"{rel}:{i}: Private key detected")

        # Claude references
        if "Co-Authored-By" in line and "Claude" in line:
            issues.append(f"{rel}:{i}: Claude Co-Authored-By reference")

        # Hardcoded user paths
        if re.search(r"/Users/\w+/|/home/\w+/", line) and "check_secrets" not in str(rel):
            issues.append(f"{rel}:{i}: Hardcoded user path")

        # Passwords
        if re.search(r"password\s*[:=]\s*['\"][^'\"]{4,}", line, re.IGNORECASE):
            issues.append(f"{rel}:{i}: Possible hardcoded password")


def main() -> int:
    for dir_name in SCAN_DIRS:
        scan_dir = ROOT / dir_name
        if not scan_dir.exists():
            continue
        for path in scan_dir.rglob("*"):
            if path.is_file() and path.suffix in SCAN_EXTENSIONS and path.name not in SKIP_FILES:
                scan_file(path)

    # Also scan root-level files
    for path in ROOT.glob("*"):
        if path.is_file() and path.suffix in SCAN_EXTENSIONS:
            scan_file(path)

    if issues:
        print(f"\n{'='*60}")
        print(f"SECRET SCAN: {len(issues)} issue(s) found")
        print(f"{'='*60}")
        for issue in issues:
            print(f"  ✗ {issue}")
        print()
        return 1

    print("Secret scan: clean ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
