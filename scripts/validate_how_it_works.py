#!/usr/bin/env python3
"""Validate how-it-works.html claims against the actual codebase.

Checks:
  1. Structural: pipeline nodes ↔ stage detail sections, sequential numbering
  2. Bilingual parity: lang-en vs lang-ja element counts
  3. Model names: every model mentioned in the HTML appears in a recipe or compose
  4. Ports: port numbers in the HTML match recipes, compose, and config
  5. Memory figures: GPU memory in the resource table matches recipe estimates
  6. Language count: "10 supported languages" matches LANGUAGE_REGISTRY size
  7. TTS replica count: matches config.py default tts_vllm_url pool size
  8. Homepage landing: checks index.html claims against the same sources

Exit 0 if all checks pass, 1 if any fail.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOW_IT_WORKS = ROOT / "static" / "how-it-works.html"
INDEX_HTML = ROOT / "static" / "index.html"
COMPOSE = ROOT / "docker-compose.gb10.yml"
CONFIG_PY = ROOT / "src" / "meeting_scribe" / "config.py"
LANGUAGES_PY = ROOT / "src" / "meeting_scribe" / "languages.py"
RECIPES_DIR = ROOT / "src" / "meeting_scribe" / "recipes"


# ── helpers ──────────────────────────────────────────────────────────


def _read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text()


def _recipe_yamls() -> dict[str, str]:
    """Return {filename: content} for all recipe YAML files."""
    if not RECIPES_DIR.is_dir():
        return {}
    return {p.name: p.read_text() for p in RECIPES_DIR.glob("*.yaml")}


def _count_language_registry(src: str) -> int | None:
    """Count entries in LANGUAGE_REGISTRY dict from languages.py source."""
    block = re.search(r"LANGUAGE_REGISTRY.*?=\s*\{(.*?)\}", src, re.DOTALL)
    if not block:
        return None
    return len(re.findall(r'^\s+"[a-z]{2}":\s+Language\(', block.group(1), re.MULTILINE))


def _tts_replica_count_from_config(src: str) -> int | None:
    """Count comma-separated URLs in the default tts_vllm_url."""
    m = re.search(r'tts_vllm_url:\s*str\s*=\s*"([^"]+)"', src)
    if not m:
        return None
    return len(m.group(1).split(","))


def _extract_memory_table(html: str) -> dict[str, str]:
    """Extract component -> memory string from the resource table.

    Returns e.g. {"ASR": "~5 GB", "Translation": "~24 GB", ...}.

    Row fingerprint: <tr> <td>LABEL</td> <td>...</td>
                       <td>...<span class="mv">~X GB[ trailing text]</span></td>
    The trailing text allowance is load-bearing — rows like "Translation
    (shared)" carry "~35 GB weights" in the memory cell. The old regex
    pinned </span> immediately after GB, which made the regex fall
    through to the next row's GB-cell and label, producing wrong rows.
    """
    # Capture each <tr> ... </tr> span individually, then grab the
    # first column (label) and the first .mv GB cell inside that row.
    rows: dict[str, str] = {}
    for tr in re.finditer(r"<tr>(.*?)</tr>", html, re.DOTALL):
        body = tr.group(1)
        label_m = re.search(r"<td>(.*?)</td>", body, re.DOTALL)
        mv_m = re.search(
            r'<span\s+class="mv">\s*(~[\d.]+\s*GB)\b',
            body,
        )
        if not label_m or not mv_m:
            continue
        label = re.sub(r"<[^>]+>", "", label_m.group(1)).strip()
        if not label:
            continue
        rows[label] = mv_m.group(1).replace("  ", " ")
    return rows


def _recipe_memory_estimates() -> dict[str, float]:
    """Derive expected memory from recipes (gpu_memory_utilization × 121 GB).

    For non-vLLM models (pyannote), use the inline comment.
    Returns component key → estimated GB.
    """
    gpu_total = 121.0  # GB10 usable GPU memory
    estimates: dict[str, float] = {}

    for name, content in _recipe_yamls().items():
        util_match = re.search(r"gpu_memory_utilization:\s*([\d.]+)", content)
        comment_gb = re.search(r"#.*?~(\d+(?:\.\d+)?)\s*GB", content)

        if "asr" in name:
            key = "ASR"
        elif "translation" in name or "35b" in name:
            key = "Translation"
        elif "sortformer" in name or "diariz" in name:
            key = "Diarization"
        elif "tts" in name and "vllm" not in name:
            key = "TTS"
        else:
            continue

        if key == "Diarization" and comment_gb:
            # pyannote isn't vLLM — use the comment estimate
            estimates[key] = float(comment_gb.group(1))
        elif util_match:
            estimates[key] = round(float(util_match.group(1)) * gpu_total, 1)

    return estimates


def _compose_tts_ports(src: str) -> list[int]:
    """Extract TTS_PORT values from non-profile services in docker-compose."""
    ports: list[int] = []
    # Only count services without profiles: (i.e., the default stack)
    # Split by service blocks and check for TTS_PORT
    for m in re.finditer(r"TTS_PORT=(\d+)", src):
        port = int(m.group(1))
        # 8022 is the vllm-omni migration target (profile-gated), skip it
        if port not in (8022,):
            ports.append(port)
    return ports


# ── checks ───────────────────────────────────────────────────────────


def check_structure(html: str) -> list[str]:
    """Pipeline nodes ↔ stage sections, sequential numbering."""
    errors: list[str] = []

    node_stages = re.findall(r'class="pipeline-node"\s+data-stage="(\w+)"', html)
    if not node_stages:
        errors.append("No pipeline nodes found (data-stage attributes)")
        return errors

    detail_stages = re.findall(r'<section\s+class="stage"\s+data-stage="(\w+)"', html)

    # Pipeline nodes that appear in the data-flow timeline but aren't
    # detail stage sections (they reuse stage colors for styling only).
    data_flow_only = {"display"}

    for stage in node_stages:
        if stage in data_flow_only:
            continue
        if stage not in detail_stages:
            errors.append(f"Pipeline node '{stage}' has no matching stage detail section")
    for stage in detail_stages:
        if stage not in node_stages:
            errors.append(f"Stage detail '{stage}' has no matching pipeline node")

    stage_numbers = [int(n) for n in re.findall(r'class="stage-number">(\d+)<', html)]
    for i, n in enumerate(stage_numbers):
        if n != i + 1:
            errors.append(f"Stage number {n} at position {i} should be {i + 1}")

    return errors


def check_bilingual_parity(html: str) -> list[str]:
    """lang-en and lang-ja element counts must match."""
    en_count = len(re.findall(r'class="[^"]*\blang-en\b', html))
    ja_count = len(re.findall(r'class="[^"]*\blang-ja\b', html))
    if en_count != ja_count:
        return [f"Bilingual mismatch: {en_count} lang-en vs {ja_count} lang-ja elements"]
    return []


def check_model_names(html: str) -> list[str]:
    """Model names in HTML must appear in recipes or compose."""
    errors: list[str] = []
    recipes = " ".join(_recipe_yamls().values())
    compose = _read(COMPOSE)
    reference = recipes + compose

    # Models explicitly named in the HTML
    expected_models = [
        "Qwen3-ASR-1.7B",
        "Qwen3.5-35B-A3B",  # partial match for the full INT4 name
        "speaker-diarization-community-1",
        "Qwen3-TTS-12Hz-0.6B",  # partial match for Base variant
    ]

    for model in expected_models:
        if model not in html:
            continue  # Not mentioned in HTML, skip
        if model not in reference:
            errors.append(f"Model '{model}' in HTML but not found in recipes or compose")

    return errors


def check_ports(html: str) -> list[str]:
    """Port numbers in HTML match recipes and config."""
    errors: list[str] = []
    recipes = _recipe_yamls()
    config = _read(CONFIG_PY)

    # Extract port → component from recipes
    recipe_ports: dict[int, str] = {}
    for name, content in recipes.items():
        port_m = re.search(r"^port:\s*(\d+)", content, re.MULTILINE)
        if port_m:
            recipe_ports[int(port_m.group(1))] = name

    # Extract all 4-digit port numbers (8xxx) from the HTML
    # From <span class="mv">NNNN</span> in the resource table
    table_ports = {int(p) for p in re.findall(r'class="mv">[^<]*?(8\d{3})', html)}
    # From the wire diagram: "port 8003" or "ports 8002, 8012"
    wire_ports = {int(p) for p in re.findall(r"ports?\s+[^<]*?(8\d{3})", html)}
    # Catch comma-separated ports like "8002, 8012"
    multi_ports = {int(p) for p in re.findall(r"\b(8\d{3})\b", html)}

    all_html_ports = table_ports | wire_ports | multi_ports

    # Check that key ports from recipes appear in the HTML
    for port, recipe_name in recipe_ports.items():
        # Skip profile-gated services (vllm-omni TTS on 8022, omni on 8032)
        if port in (8022, 8032):
            continue
        if port not in all_html_ports:
            errors.append(f"Port {port} (from {recipe_name}) not mentioned in how-it-works")

    # Check that TTS replica ports from config appear in the HTML
    tts_url_m = re.search(r'tts_vllm_url.*?=\s*"([^"]+)"', config)
    if tts_url_m:
        config_tts_ports = {int(p) for p in re.findall(r":(\d{4})", tts_url_m.group(1))}
        for port in config_tts_ports:
            if port not in all_html_ports:
                errors.append(f"TTS port {port} (from config default) not in how-it-works")

    return errors


def check_memory(html: str) -> list[str]:
    """Memory figures in the resource table should be within range of recipe estimates."""
    errors: list[str] = []
    table = _extract_memory_table(html)
    estimates = _recipe_memory_estimates()

    for component, estimated_gb in estimates.items():
        # Find matching table entry (fuzzy match on component name)
        table_value = None
        for label, mem_str in table.items():
            if component.lower() in label.lower():
                table_value = mem_str
                break

        if table_value is None:
            errors.append(f"Component '{component}' not found in resource table")
            continue

        # Parse the "~N GB" value
        m = re.match(r"~([\d.]+)", table_value)
        if not m:
            errors.append(f"Cannot parse memory value '{table_value}' for {component}")
            continue

        html_gb = float(m.group(1))

        # Allow 50% tolerance — these are estimates
        if abs(html_gb - estimated_gb) / max(estimated_gb, 0.1) > 0.5:
            errors.append(
                f"Memory mismatch for {component}: "
                f"HTML says {table_value}, recipe estimates ~{estimated_gb:.1f} GB"
            )

    return errors


def check_language_count(html: str) -> list[str]:
    """'10 supported languages' claim must match LANGUAGE_REGISTRY."""
    errors: list[str] = []
    lang_src = _read(LANGUAGES_PY)
    if not lang_src:
        errors.append("languages.py not found — cannot verify language count")
        return errors

    actual_count = _count_language_registry(lang_src)
    if actual_count is None:
        errors.append("Could not parse LANGUAGE_REGISTRY from languages.py")
        return errors

    # Find the claimed count in the HTML
    claimed = re.findall(r"(\d+)\s+(?:supported\s+)?(?:TTS-capable\s+)?languages", html)
    for c in claimed:
        if int(c) != actual_count:
            errors.append(f"HTML claims {c} languages but LANGUAGE_REGISTRY has {actual_count}")

    return errors


def check_tts_replicas(html: str) -> list[str]:
    """TTS replica count in HTML should match config default pool size."""
    errors: list[str] = []
    config_src = _read(CONFIG_PY)
    if not config_src:
        errors.append("config.py not found — cannot verify TTS replica count")
        return errors

    config_count = _tts_replica_count_from_config(config_src)
    if config_count is None:
        errors.append("Could not parse tts_vllm_url from config.py")
        return errors

    # Check if the HTML mentions the right replica count
    # Look for "×N replicas" or "N replicas"
    replica_claims = re.findall(r"[×x](\d+)\s*(?:replicas|レプリカ)", html)
    if not replica_claims:
        if config_count > 1:
            errors.append(
                f"Config has {config_count} TTS replicas but HTML doesn't mention replicas"
            )
    else:
        for claim in replica_claims:
            if int(claim) != config_count:
                errors.append(f"HTML says ×{claim} TTS replicas but config has {config_count}")

    return errors


def check_homepage(index_html: str) -> list[str]:
    """Verify homepage landing page claims against sources of truth."""
    errors: list[str] = []
    if not index_html:
        return errors

    lang_src = _read(LANGUAGES_PY)
    actual_count = _count_language_registry(lang_src) if lang_src else None

    # Should NOT say "Japanese and English" as the only pair
    ja_en_only = re.findall(
        r"(?:between|detects)\s+Japanese\s+and\s+English",
        index_html,
        re.IGNORECASE,
    )
    if ja_en_only:
        errors.append(
            f"Homepage still says 'Japanese and English' — "
            f"should reflect {actual_count or '10'} supported languages"
        )

    # Badge should say "multilingual" not "bilingual"
    badge_m = re.search(r'landing-hero-badge">(.*?)<', index_html)
    if badge_m and "bilingual" in badge_m.group(1).lower():
        errors.append("Homepage badge says 'bilingual' — should say 'multilingual'")

    return errors


# ── main ─────────────────────────────────────────────────────────────


def validate() -> list[str]:
    html = _read(HOW_IT_WORKS)
    if not html:
        return ["how-it-works.html not found"]

    index = _read(INDEX_HTML)

    errors: list[str] = []
    checks = [
        ("structure", lambda: check_structure(html)),
        ("bilingual parity", lambda: check_bilingual_parity(html)),
        ("model names", lambda: check_model_names(html)),
        ("ports", lambda: check_ports(html)),
        ("memory figures", lambda: check_memory(html)),
        ("language count", lambda: check_language_count(html)),
        ("TTS replicas", lambda: check_tts_replicas(html)),
        ("homepage", lambda: check_homepage(index)),
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
            f"how-it-works validation FAILED ({len(errors)} issue{'s' if len(errors) != 1 else ''}):"
        )
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("how-it-works validation: all checks passed ✓")
        print("  structure, bilingual parity, model names, ports,")
        print("  memory figures, language count, TTS replicas, homepage")


if __name__ == "__main__":
    main()
