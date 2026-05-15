#!/usr/bin/env python3
"""Generate ``src/meeting_scribe/data/eff-wordlist.txt`` from a
system word dictionary.

The recovery code (``meeting_scribe.terminal.recovery``) draws six
words from this list to mint a ~66-bit recovery phrase. The EFF's
public-domain 2048-word large-wordlist is the gold standard, but
pulling it at build time would require internet at provisioning
which the v1.0 plan rules out.

This generator curates a 2048-word list locally from
``/usr/share/dict/american-english`` (or any text file given on
the command line) by filtering for memorable, low-typo words:

  * 4 ≤ length ≤ 8 characters
  * all lowercase ASCII letters only
  * no possessives / apostrophes / accents
  * deduped, alphabetized, capped at 2048
  * deterministic — same input produces the same wordlist, so
    re-running on a different image doesn't rotate the recovery
    surface unnecessarily

Usage:
  python3 scripts/build-eff-wordlist.py
  python3 scripts/build-eff-wordlist.py /path/to/source-dict.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "src" / "meeting_scribe" / "data" / "eff-wordlist.txt"
DEFAULT_SOURCE = Path("/usr/share/dict/american-english")
TARGET_COUNT = 2048

ALLOWED = re.compile(r"^[a-z]{4,8}$")


def main() -> int:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    if not source.exists():
        print(f"source dictionary missing: {source}", file=sys.stderr)
        return 2

    words: set[str] = set()
    for raw in source.read_text(encoding="utf-8", errors="replace").splitlines():
        w = raw.strip().lower()
        if ALLOWED.fullmatch(w):
            words.add(w)

    if len(words) < TARGET_COUNT:
        print(
            f"only {len(words)} eligible words found in {source}; need >= {TARGET_COUNT}",
            file=sys.stderr,
        )
        return 3

    chosen = sorted(words)[:TARGET_COUNT]
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    print(f"wrote {len(chosen)} words to {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
