#!/usr/bin/env python3
"""Validate static/hardware-scaling.html numeric claims.

The page makes external claims (NVIDIA / Dell / Lenovo hardware
specs) that can't be cross-checked against repo source code. Instead,
every numeric value carries a ``data-claim`` attribute that resolves
to a dotted path in static/data/hardware-scaling-claims.json. This
validator keeps the prose and the claims manifest in lockstep.

Checks:
  1. Bilingual parity: lang-en vs lang-ja counts match.
  2. Required-claim coverage: every [data-spec-value] element carries
     a non-empty data-claim attribute.
  3. Numeric correctness: rendered text on each [data-claim] matches
     the value at the dotted path in the claims JSON (after unit
     normalisation).
  4. Tier-key completeness: every numeric leaf under ``tiers`` in the
     claims JSON has at least one data-claim referencing it on the
     page (keeps prose and JSON in lockstep both ways).

Exit 0 if all checks pass, 1 if any fail.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
HARDWARE_HTML = ROOT / "static" / "hardware-scaling.html"
CLAIMS_JSON = ROOT / "static" / "data" / "hardware-scaling-claims.json"


# ── helpers ──────────────────────────────────────────────────────────


def _read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text()


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _resolve_dotted(d: dict, path: str):
    """tiers.spark.fast_memory_gb → walk the nested dict.

    Also accepts short paths like ``spark.fast_memory_gb`` and looks
    them up under ``d["tiers"]`` if the first segment isn't a
    top-level key. Page authors keep paths short for readability;
    the JSON groups numbers under ``tiers`` for hygiene.
    """
    parts = path.split(".")
    if parts and parts[0] not in d and "tiers" in d and parts[0] in d.get("tiers", {}):
        parts = ["tiers", *parts]
    cur = d
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _iter_spec_value_tags(html: str):
    """Yield every element carrying data-spec-value.

    Returns dicts with: tag (full tag), attrs (dict), inner_text.
    """
    for m in re.finditer(
        r"<(\w+)\b([^>]*\bdata-spec-value\b[^>]*)>(.*?)</\1>",
        html,
        re.DOTALL,
    ):
        attrs_text = m.group(2)
        inner = m.group(3)
        # Strip inner HTML tags for the rendered-text comparison
        inner_text = re.sub(r"<[^>]+>", "", inner).strip()
        attrs: dict[str, str] = {}
        for am in re.finditer(r'\b([\w-]+)\s*=\s*"([^"]*)"', attrs_text):
            attrs[am.group(1)] = am.group(2)
        yield {"tag": m.group(0), "attrs": attrs, "inner_text": inner_text}


# Unit-normalisation: extract a numeric value from rendered prose and
# compare to a JSON value. The JSON stores raw numbers (e.g. 128 for
# "128 GB") so we strip the unit suffix before float comparison.
def _normalise_number(rendered: str):
    """Return a float / int / str representation suitable for comparing
    against the JSON value."""
    s = rendered.strip()
    # Strip leading "~", "USD ", "= ", commas, and trailing units.
    s = re.sub(r"^[~=]\s*", "", s)
    s = re.sub(r"^USD\s+", "", s)
    s = s.replace(",", "")
    # Match a number prefix
    m = re.match(r"(-?\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1))
    return s  # string-equality fallback for things like form-factor labels


def _claim_value_for_compare(value):
    if isinstance(value, (int, float)):
        return float(value)
    return value


# ── checks ───────────────────────────────────────────────────────────


def check_bilingual_parity(html: str) -> list[str]:
    en_count = len(re.findall(r'class="[^"]*\blang-en\b', html))
    ja_count = len(re.findall(r'class="[^"]*\blang-ja\b', html))
    if en_count != ja_count:
        return [f"Bilingual mismatch: {en_count} lang-en vs {ja_count} lang-ja elements"]
    return []


def check_required_claim_coverage(html: str) -> list[str]:
    """Every [data-spec-value] element must carry a non-empty data-claim."""
    errors: list[str] = []
    for element in _iter_spec_value_tags(html):
        if not element["attrs"].get("data-claim", "").strip():
            errors.append(f"missing data-claim on spec value {element['inner_text']!r}")
    return errors


def check_numeric_correctness(html: str, claims: dict) -> list[str]:
    """For every [data-claim], resolve into the claims JSON and confirm
    the rendered text matches."""
    errors: list[str] = []
    for element in _iter_spec_value_tags(html):
        attrs = element["attrs"]
        claim_path = attrs.get("data-claim", "").strip()
        if not claim_path:
            continue  # already flagged by required-claim check
        expected = _resolve_dotted(claims, claim_path)
        if expected is None:
            errors.append(f"data-claim={claim_path!r}: not present in hardware-scaling-claims.json")
            continue

        rendered = element["inner_text"]
        rendered_norm = _normalise_number(rendered)
        expected_norm = _claim_value_for_compare(expected)

        # If both are numeric, compare with a tiny tolerance.
        if isinstance(rendered_norm, float) and isinstance(expected_norm, float):
            if abs(rendered_norm - expected_norm) > 1e-9:
                errors.append(
                    f"data-claim={claim_path!r}: rendered {rendered!r} → {rendered_norm} "
                    f"vs JSON {expected_norm}"
                )
        else:
            # String comparison (e.g. form-factor labels)
            if str(rendered).strip() != str(expected).strip():
                errors.append(
                    f"data-claim={claim_path!r}: rendered {rendered!r} vs JSON {expected!r}"
                )
    return errors


def check_tier_completeness(html: str, claims: dict) -> list[str]:
    """Every numeric leaf under tiers in the claims JSON must be referenced
    by at least one data-claim on the page (so the JSON and prose stay
    in lockstep both ways)."""
    errors: list[str] = []
    page_claims = set(re.findall(r'data-claim="([^"]+)"', html))

    tiers = claims.get("tiers", {})
    referenced: set[str] = set()

    def visit(prefix: str, value):
        if isinstance(value, dict):
            for k, v in value.items():
                if k.startswith("_"):
                    continue
                visit(f"{prefix}.{k}" if prefix else k, v)
        else:
            referenced.add(prefix)

    visit("tiers", tiers)
    # Strip the "tiers." prefix to match the claim-path convention used on the page.
    expected_paths = {p[len("tiers.") :] for p in referenced if p.startswith("tiers.")}

    # Some entries in the claims JSON are labels in stage titles (not
    # exposed as a data-claim spec value) or are pure metadata. Skip
    # those legitimately.
    SKIP_PATHS = {
        "spark.name",
        "b200.name",
        "nvl72.name",
        "spark.cpu_short",
        "b200.cpu_short",
        "nvl72.grace_cpu_count",
    }
    expected_paths -= SKIP_PATHS

    missing = sorted(expected_paths - page_claims)
    if missing:
        errors.append(f"claims JSON has keys with no data-claim reference on the page: {missing}")
    return errors


# ── main ─────────────────────────────────────────────────────────────


def validate() -> list[str]:
    html = _read(HARDWARE_HTML)
    if not html:
        return [f"hardware-scaling.html not found at {HARDWARE_HTML}"]

    claims = _load_json(CLAIMS_JSON)
    if not claims:
        return [f"hardware-scaling-claims.json not found at {CLAIMS_JSON}"]

    errors: list[str] = []
    checks = [
        ("bilingual parity", lambda: check_bilingual_parity(html)),
        ("required claim coverage", lambda: check_required_claim_coverage(html)),
        ("numeric correctness", lambda: check_numeric_correctness(html, claims)),
        ("tier completeness", lambda: check_tier_completeness(html, claims)),
    ]

    for name, check_fn in checks:
        try:
            results = check_fn()
        except Exception as exc:
            results = [f"check crashed: {exc}"]
        for e in results:
            errors.append(f"[{name}] {e}")

    return errors


def main() -> None:
    errors = validate()
    if errors:
        print(
            f"hardware-scaling validation FAILED "
            f"({len(errors)} issue{'s' if len(errors) != 1 else ''}):"
        )
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("hardware-scaling validation: all checks passed ✓")
        print("  bilingual parity, required claim coverage,")
        print("  numeric correctness, tier completeness")


if __name__ == "__main__":
    main()
