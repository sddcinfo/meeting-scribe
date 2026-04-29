#!/usr/bin/env python3
"""Smoke-test the multilingual self-introduction name extractor."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meeting_scribe.speaker.name_extraction import extract_name as extract

CASES: list[tuple[str, str | None]] = [
    # English
    ("My name is Brad.", "Brad"),
    ("Hi, my name is Sarah.", "Sarah"),
    ("Hi, I'm Sarah", "Sarah"),
    ("I'm Brad", "Brad"),
    ("I am Brad", "Brad"),
    ("Call me Mike", "Mike"),
    ("This is Lisa speaking", "Lisa"),
    ("Brad here", "Brad"),
    ("Brad", "Brad"),
    ("Brad. Brad.", "Brad"),
    # Spanish
    ("Me llamo Carlos", "Carlos"),
    ("Mi nombre es Carlos", "Carlos"),
    ("Soy Carlos.", "Carlos"),
    # French
    ("Je m'appelle Pierre", "Pierre"),
    ("Mon nom est Pierre", "Pierre"),
    ("Moi, c'est Pierre", "Pierre"),
    ("Je suis Pierre.", "Pierre"),
    # German
    ("Ich heiße Klaus", "Klaus"),
    ("Ich heisse Klaus", "Klaus"),
    ("Mein Name ist Klaus", "Klaus"),
    ("Ich bin Klaus", "Klaus"),
    # Italian
    ("Mi chiamo Marco", "Marco"),
    ("Il mio nome è Marco", "Marco"),
    ("Sono Marco", "Marco"),
    # Portuguese
    ("Me chamo João", "João"),
    ("Eu me chamo João", "João"),
    ("Meu nome é João", "João"),
    ("Eu sou João", "João"),
    # Russian (Cyrillic) — BGN/PCGN romanization
    ("Меня зовут Иван", "Ivan"),
    ("Моё имя Иван", "Ivan"),
    ("Я Иван", "Ivan"),
    # Japanese — names are romanized to Latin, honorifics stripped
    ("私は田中です", "Tanaka"),
    ("わたしは田中です", "Tanaka"),
    ("僕は田中です", "Tanaka"),
    ("田中です", "Tanaka"),
    ("田中。田中。", "Tanaka"),
    ("田中と申します", "Tanaka"),
    ("田中さん", "Tanaka"),
    ("カワサキさん", "Kawasaki"),
    ("川崎", "Kawasaki"),
    # Korean — Revised Romanization (jamo-based, deterministic)
    ("저는 박지민입니다", "Bakjimin"),
    ("박지민입니다", "Bakjimin"),
    ("제 이름은 박지민입니다", "Bakjimin"),
    # Chinese — kanji-only path falls through to Japanese readings; documented
    # limitation until pypinyin is available. Skip strict assertion.
    # Negatives
    ("the meeting is about to start", None),
    ("hello everyone", None),
    ("I'm going to the store", None),
    ("good morning", None),
]


def _norm(value: str | None) -> str | None:
    """Drop a trailing punctuation-only difference for assertion robustness."""
    return value.strip() if isinstance(value, str) else value


def main() -> int:
    failures = 0
    for text, expected in CASES:
        got = extract(text)
        ok = got == expected
        marker = "✓" if ok else "✗"
        print(f"{marker} {text!r:55} → {got!r:15}  (expected {expected!r})")
        if not ok:
            failures += 1
    print()
    print(f"{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
