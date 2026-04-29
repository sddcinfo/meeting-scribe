"""Post-fix language tagging on B1 bench output.

The B1 harness wrote ``language: "ja"`` for every per-sample row
regardless of source corpus.  This script re-derives the language
from the corpus_id prefix (`fleurs_ja_jp_*` → ja, `fleurs_en_us_*`
→ en) so the reducer's per-language bucketing is correct.

In-place edit of both qwen3_asr.json and cohere_transcribe.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def relabel(path: Path) -> tuple[int, int]:
    data = json.loads(path.read_text())
    changed = 0
    for row in data.get("per_sample", []):
        cid = row.get("id", "")
        if cid.startswith("fleurs_ja_"):
            new_lang = "ja"
        elif cid.startswith("fleurs_en_"):
            new_lang = "en"
        else:
            continue
        if row.get("language") != new_lang:
            row["language"] = new_lang
            changed += 1
    path.write_text(json.dumps(data, indent=2))
    return len(data.get("per_sample", [])), changed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, nargs="+", required=True)
    args = p.parse_args()
    for path in args.input:
        total, changed = relabel(path)
        print(f"{path}: {changed}/{total} rows relabelled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
