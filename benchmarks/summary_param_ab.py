#!/usr/bin/env python3
"""A/B harness for the meeting-summary LLM call.

Runs ``meeting_scribe.summary.generate_summary`` against the same set
of real meeting transcripts under two parameter configs and reports
latency + quality side-by-side. Lives next to ``perf_baseline.py`` /
``reprocess_bench.py`` (this is the same kind of "compare runs" gate)
and writes its JSON output under ``benchmarks/results/`` so the
review pattern is identical: diff the JSON, decide whether to land
the change.

Usage:
    python benchmarks/summary_param_ab.py [--meetings N] [--vllm URL]

The two configs being compared are pinned in ``CONFIGS`` below;
adjust when running a new sweep.

Decision rule for landing a parameter change is printed at the bottom
of the markdown report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))  # share validate_meeting helpers

from validate_meeting import _best_finals_by_id

from meeting_scribe.summary import (
    _build_user_prompt,
    _call_vllm_summary,
    build_transcript_text,
)

logger = logging.getLogger("summary_param_ab")


# Pinned configs for this sweep — adjust + re-run when sweeping new
# dimensions. Defaults compare current production (thinking ON,
# 8 K cap) against a "fast" candidate (thinking OFF, 4 K cap).
CONFIGS: list[tuple[str, bool, int]] = [
    ("baseline_thinking_8192", True, 8192),
    ("fast_no_thinking_4096", False, 4096),
]

RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"


@dataclass
class _RunResult:
    config_name: str
    meeting_id: str
    elapsed_ms: float
    summary: dict[str, Any] | None
    error: str | None
    prompt_chars: int
    n_finals: int
    response_chars: int = 0
    num_topics: int = 0
    num_action_items: int = 0
    num_decisions: int = 0
    topic_names: list[str] = field(default_factory=list)
    hallucinated_speakers: list[str] = field(default_factory=list)


def _select_meetings(meetings_dir: Path, n: int) -> list[Path]:
    """Pick ``n`` meeting dirs that have a non-empty journal,
    stratified across short / medium / long buckets."""
    candidates: list[tuple[int, Path]] = []
    for d in sorted(meetings_dir.iterdir()):
        if not d.is_dir():
            continue
        journal = d / "journal.jsonl"
        if not journal.exists():
            continue
        finals = _best_finals_by_id(journal)
        if len(finals) < 5:
            continue
        candidates.append((len(finals), d))
    if not candidates:
        return []

    short = [c for c in candidates if c[0] < 100]
    medium = [c for c in candidates if 100 <= c[0] < 500]
    long_ = [c for c in candidates if c[0] >= 500]
    rng = random.Random(0xA5)  # deterministic across runs
    per_bucket = max(1, n // 3)
    picked: list[Path] = []
    for bucket in (short, medium, long_):
        if not bucket:
            continue
        rng.shuffle(bucket)
        picked.extend(p for _, p in bucket[:per_bucket])
    if len(picked) > n:
        picked = picked[:n]
    return picked


def _detected_speaker_names(meeting_dir: Path) -> set[str]:
    path = meeting_dir / "detected_speakers.json"
    if not path.exists():
        return set()
    try:
        speakers = json.loads(path.read_text())
    except json.JSONDecodeError:
        return set()
    names: set[str] = set()
    for s in speakers:
        for key in ("display_name", "name", "speaker_label"):
            v = s.get(key)
            if isinstance(v, str) and v.strip():
                names.add(v.strip())
    return names


def _collect_referenced_speakers(summary: dict[str, Any]) -> set[str]:
    """Pull speaker names referenced in summary's free-text fields.

    Heuristic — looks at action_items[*].owner, topics[*].speakers,
    decisions[*].owner."""
    referenced: set[str] = set()
    for item in summary.get("action_items", []) or []:
        if isinstance(item, dict):
            owner = item.get("owner") or item.get("assignee")
            if isinstance(owner, str) and owner.strip():
                referenced.add(owner.strip())
    for topic in summary.get("topics", []) or []:
        if isinstance(topic, dict):
            spks = topic.get("speakers") or []
            for s in spks if isinstance(spks, list) else []:
                if isinstance(s, str) and s.strip():
                    referenced.add(s.strip())
    for d in summary.get("decisions", []) or []:
        if isinstance(d, dict):
            owner = d.get("owner") or d.get("decider")
            if isinstance(owner, str) and owner.strip():
                referenced.add(owner.strip())
    return referenced


async def _run_one(
    vllm_url: str,
    meeting_dir: Path,
    config_name: str,
    enable_thinking: bool,
    max_tokens: int,
) -> _RunResult:
    events, transcript = build_transcript_text(meeting_dir, max_chars=100_000)
    if not events:
        return _RunResult(
            config_name=config_name,
            meeting_id=meeting_dir.name,
            elapsed_ms=0.0,
            summary=None,
            error="no events",
            prompt_chars=0,
            n_finals=0,
        )
    speakers_path = meeting_dir / "detected_speakers.json"
    speakers: list[dict] = []
    if speakers_path.exists():
        try:
            speakers = json.loads(speakers_path.read_text())
        except json.JSONDecodeError:
            speakers = []
    user_prompt = _build_user_prompt(events, transcript, speakers)
    t0 = time.monotonic()
    summary, error, _elapsed_ms, _model = await _call_vllm_summary(
        vllm_url,
        user_prompt,
        priority=-10,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )
    elapsed_ms = (time.monotonic() - t0) * 1000
    res = _RunResult(
        config_name=config_name,
        meeting_id=meeting_dir.name,
        elapsed_ms=elapsed_ms,
        summary=summary if isinstance(summary, dict) else None,
        error=error,
        prompt_chars=len(user_prompt),
        n_finals=len(events),
    )
    if isinstance(summary, dict):
        res.response_chars = len(json.dumps(summary, ensure_ascii=False))
        res.num_topics = len(summary.get("topics", []) or [])
        res.num_action_items = len(summary.get("action_items", []) or [])
        res.num_decisions = len(summary.get("decisions", []) or [])
        res.topic_names = [
            t.get("title") or t.get("name") or ""
            for t in (summary.get("topics", []) or [])
            if isinstance(t, dict)
        ]
        detected = _detected_speaker_names(meeting_dir)
        if detected:
            referenced = _collect_referenced_speakers(summary)
            res.hallucinated_speakers = sorted(referenced - detected)
    return res


def _jaccard(a: list[str], b: list[str]) -> float:
    sa = {x.strip().lower() for x in a if x.strip()}
    sb = {x.strip().lower() for x in b if x.strip()}
    if not (sa | sb):
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _format_report(
    results: dict[str, list[_RunResult]],
    meeting_ids: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# Summary LLM-param A/B")
    lines.append("")
    lines.append("Configs:")
    for name, thinking, mt in CONFIGS:
        lines.append(f"- **{name}** — enable_thinking={thinking}, max_tokens={mt}")
    lines.append("")

    name_a, name_b = CONFIGS[0][0], CONFIGS[1][0]
    runs_a = {r.meeting_id: r for r in results[name_a]}
    runs_b = {r.meeting_id: r for r in results[name_b]}

    lines.append("## Per-meeting comparison")
    lines.append("")
    lines.append(
        "| meeting | events | A latency (ms) | B latency (ms) | speedup | "
        "A topics | B topics | A actions | B actions | "
        "A resp chars | B resp chars | topic Jaccard |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    speedups: list[float] = []
    jaccards: list[float] = []
    for mid in meeting_ids:
        ra = runs_a.get(mid)
        rb = runs_b.get(mid)
        if not ra or not rb:
            continue
        speedup = ra.elapsed_ms / rb.elapsed_ms if rb.elapsed_ms > 0 else 0
        jac = _jaccard(ra.topic_names, rb.topic_names)
        speedups.append(speedup)
        jaccards.append(jac)
        lines.append(
            f"| `{mid[:8]}…` | {ra.n_finals} | {ra.elapsed_ms:.0f} | {rb.elapsed_ms:.0f} | "
            f"{speedup:.2f}x | {ra.num_topics} | {rb.num_topics} | "
            f"{ra.num_action_items} | {rb.num_action_items} | "
            f"{ra.response_chars} | {rb.response_chars} | {jac:.2f} |"
        )

    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        avg_jac = sum(jaccards) / len(jaccards)
        lines.append("")
        lines.append("## Aggregate")
        lines.append("")
        lines.append(f"- avg speedup A→B: **{avg_speedup:.2f}x**")
        lines.append(
            f"- avg topic Jaccard A vs B: **{avg_jac:.2f}** "
            "(1.0 = identical topic sets, 0.0 = disjoint)"
        )
        n_topic_diff = sum(
            1
            for mid in meeting_ids
            if mid in runs_a
            and mid in runs_b
            and abs(runs_a[mid].num_topics - runs_b[mid].num_topics) > 1
        )
        lines.append(f"- meetings where topic count differs by >1: **{n_topic_diff}**")
        n_actions_diff = sum(
            1
            for mid in meeting_ids
            if mid in runs_a
            and mid in runs_b
            and abs(runs_a[mid].num_action_items - runs_b[mid].num_action_items) > 1
        )
        lines.append(f"- meetings where action_item count differs by >1: **{n_actions_diff}**")
        n_halluc_a = sum(1 for r in results[name_a] if r.hallucinated_speakers)
        n_halluc_b = sum(1 for r in results[name_b] if r.hallucinated_speakers)
        lines.append(
            f"- meetings with hallucinated speakers — A: **{n_halluc_a}**, B: **{n_halluc_b}**"
        )

    lines.append("")
    lines.append("## Decision rule")
    lines.append("")
    lines.append(
        "Re-apply the change ONLY if avg speedup ≥ 2.0× AND avg topic "
        "Jaccard ≥ 0.65 AND no new hallucinated speakers in B. Otherwise "
        "keep the production config and document the tradeoff."
    )
    return "\n".join(lines) + "\n"


async def _amain(args: argparse.Namespace) -> int:
    meetings_dir = REPO_ROOT / "meetings"
    if not meetings_dir.exists():
        print(f"no meetings dir at {meetings_dir}", file=sys.stderr)
        return 2

    chosen = _select_meetings(meetings_dir, args.meetings)
    if not chosen:
        print("no eligible meetings found", file=sys.stderr)
        return 2

    print(f"Sweeping {len(chosen)} meetings against {len(CONFIGS)} configs...")
    results: dict[str, list[_RunResult]] = {name: [] for name, _, _ in CONFIGS}
    meeting_ids = [m.name for m in chosen]
    for cfg_name, thinking, mt in CONFIGS:
        print(f"\n── config: {cfg_name} (thinking={thinking}, max_tokens={mt}) ──")
        for m in chosen:
            print(f"  {m.name}...", end=" ", flush=True)
            r = await _run_one(args.vllm, m, cfg_name, thinking, mt)
            results[cfg_name].append(r)
            if r.error:
                print(f"ERROR ({r.error})")
            else:
                print(f"{r.elapsed_ms:.0f}ms · topics={r.num_topics} actions={r.num_action_items}")

    out_dir = (
        Path(args.out)
        if args.out
        else RESULTS_DIR / f"summary_param_ab_{time.strftime('%Y-%m-%dT%H-%M-%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for cfg_name, runs in results.items():
        for r in runs:
            (out_dir / f"{cfg_name}__{r.meeting_id}.json").write_text(
                json.dumps(asdict(r), indent=2, ensure_ascii=False) + "\n"
            )
    report = _format_report(results, meeting_ids)
    (out_dir / "summary.md").write_text(report)
    print(f"\nWrote per-meeting JSON + summary.md to {out_dir.relative_to(REPO_ROOT)}/")
    print()
    print(report)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meetings",
        type=int,
        default=6,
        help="Total meetings to sweep, stratified short/med/long.",
    )
    parser.add_argument("--vllm", default="http://localhost:8010", help="vLLM endpoint URL.")
    parser.add_argument(
        "--out",
        default="",
        help="Output dir; defaults to benchmarks/results/summary_param_ab_<ts>.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
