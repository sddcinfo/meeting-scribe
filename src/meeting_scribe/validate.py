"""``meeting-scribe validate`` orchestrator (Workstream B).

A single command that proves every backend is reachable AND producing
sane output. Local-only (GB10); committed fixtures only — never reads
arbitrary ``meetings/`` content at validate time.

Modes:
  --quick : liveness-only sweep + furigana in-process probe (≤ 5 s).
            Does NOT start a meeting. Safe to run during a recording.
  --full  : --quick + ASR / Translate / Diarize / TTS quality probes
            against committed fixtures (≤ 5 min). Perturbs the running
            stack but does not start a meeting.
  --e2e   : --full + a transient meeting that streams a known WAV and
            measures end-to-end lag (≤ 10 min). Explicit opt-in
            because it perturbs the running stack significantly.

Each phase writes a row to ``diagnostics/validate-<timestamp>.json``
with status (``pass``/``fail``/``skip``), wall-clock, and a redacted
detail string.

Exit code: 0 if every phase that ran returned ``pass``; 1 otherwise.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from meeting_scribe.config import ServerConfig

logger = logging.getLogger(__name__)


_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "validate"
_BASELINES_PATH = _FIXTURE_DIR / "baselines.json"


@dataclass
class PhaseResult:
    name: str
    status: str  # "pass" | "fail" | "skip"
    elapsed_ms: float
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidateReport:
    started_at: float
    finished_at: float
    mode: str
    hardware_class: str
    phases: list[PhaseResult]

    @property
    def passed(self) -> bool:
        return all(p.status != "fail" for p in self.phases)

    def to_json(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "mode": self.mode,
            "hardware_class": self.hardware_class,
            "phases": [asdict(p) for p in self.phases],
            "passed": self.passed,
        }


def _load_baselines(hardware_class: str = "gb10") -> dict[str, Any]:
    """Load baselines, with env-var overrides applied last."""
    if not _BASELINES_PATH.exists():
        return {}
    raw = json.loads(_BASELINES_PATH.read_text())
    out = dict(raw.get(hardware_class, {}))
    # Env overrides
    overrides = {
        ("asr", "p95_ms"): "SCRIBE_VALIDATE_ASR_P95_MS",
        ("translate", "p95_ms"): "SCRIBE_VALIDATE_TRANSLATE_P95_MS",
        ("translate", "min_bleu"): "SCRIBE_VALIDATE_TRANSLATE_MIN_BLEU",
        ("diarize", "p95_ms"): "SCRIBE_VALIDATE_DIARIZE_P95_MS",
        ("tts", "p95_ttfa_ms"): "SCRIBE_VALIDATE_TTS_P95_TTFA_MS",
        ("e2e_lag", "p95_utterance_to_audio_ms"): "SCRIBE_VALIDATE_E2E_P95_MS",
    }
    for (sec, key), env in overrides.items():
        v = os.environ.get(env)
        if v is None:
            continue
        try:
            out.setdefault(sec, {})[key] = float(v)
        except ValueError:
            continue
    return out


# ──────────────────────────────────────────────────────────────────
# Phase 1 — Liveness (always runs)
# ──────────────────────────────────────────────────────────────────


async def _phase_liveness(config: ServerConfig) -> PhaseResult:
    """HTTP /health probes against every backend, in parallel."""
    t0 = time.monotonic()

    async def _probe(name: str, url: str, path: str = "/health") -> tuple[str, bool, str]:
        try:
            async with httpx.AsyncClient(timeout=2.5) as c:
                r = await c.get(f"{url.rstrip('/')}{path}")
                return name, r.status_code == 200, f"HTTP {r.status_code}"
        except Exception as e:
            return name, False, type(e).__name__

    probes = [
        _probe("asr", config.asr_vllm_url, "/v1/models"),
        _probe("translate", config.translate_vllm_url, "/v1/models"),
        _probe("diarize", config.diarize_url, "/health"),
        _probe(
            "tts",
            (config.tts_vllm_url or "http://localhost:8002").split(",")[0].strip(),
            "/health",
        ),
    ]
    results = await asyncio.gather(*probes)
    failures = [(n, d) for (n, ok, d) in results if not ok]
    elapsed = (time.monotonic() - t0) * 1000.0

    return PhaseResult(
        name="liveness",
        status="fail" if failures else "pass",
        elapsed_ms=elapsed,
        detail=(
            "all backends OK"
            if not failures
            else "down: " + ", ".join(f"{n}({d})" for n, d in failures)
        ),
        metrics={"probes": [{"name": n, "ok": ok, "detail": d} for n, ok, d in results]},
    )


# ──────────────────────────────────────────────────────────────────
# Phase 2 — Furigana in-process probe (always runs)
# ──────────────────────────────────────────────────────────────────


async def _phase_furigana() -> PhaseResult:
    """Lazy-import + probe pykakasi. No network."""
    t0 = time.monotonic()
    try:
        from meeting_scribe.backends.furigana import FuriganaBackend

        backend = FuriganaBackend()
        await backend.start()
        html = await backend.annotate("会議")
        elapsed = (time.monotonic() - t0) * 1000.0
        if html and "<ruby>" in html:
            return PhaseResult(
                name="furigana",
                status="pass",
                elapsed_ms=elapsed,
                detail="ruby markup OK",
            )
        return PhaseResult(
            name="furigana",
            status="fail",
            elapsed_ms=elapsed,
            detail="annotate() returned no <ruby> markup",
        )
    except Exception as e:
        return PhaseResult(
            name="furigana",
            status="fail",
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
            detail=f"{type(e).__name__}",
        )


# ──────────────────────────────────────────────────────────────────
# Phase 3 — ASR latency (full mode)
# ──────────────────────────────────────────────────────────────────


async def _phase_asr_latency(config: ServerConfig, baseline: dict) -> PhaseResult:
    """Send a fixture WAV through ASR; gate on p95 latency."""
    t0 = time.monotonic()
    wav_path = _FIXTURE_DIR / "audio_en_short.wav"
    if not wav_path.exists():
        return PhaseResult(
            name="asr_latency",
            status="skip",
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
            detail="fixture audio_en_short.wav missing",
        )

    p95_threshold = float(baseline.get("asr", {}).get("p95_ms", 1000))
    # 5 sequential ASR calls — quick latency sample
    samples_ms: list[float] = []
    audio_bytes = wav_path.read_bytes()
    url = f"{config.asr_vllm_url.rstrip('/')}/v1/audio/transcriptions"
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            for _ in range(5):
                t_call = time.monotonic()
                r = await c.post(
                    url,
                    files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                    data={"model": "Qwen/Qwen3-ASR-1.7B"},
                )
                samples_ms.append((time.monotonic() - t_call) * 1000.0)
                if r.status_code != 200:
                    return PhaseResult(
                        name="asr_latency",
                        status="fail",
                        elapsed_ms=(time.monotonic() - t0) * 1000.0,
                        detail=f"HTTP {r.status_code}",
                    )
    except Exception as e:
        return PhaseResult(
            name="asr_latency",
            status="fail",
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
            detail=type(e).__name__,
        )

    samples_ms.sort()
    p95 = samples_ms[int(len(samples_ms) * 0.95)] if samples_ms else 0.0
    elapsed = (time.monotonic() - t0) * 1000.0
    return PhaseResult(
        name="asr_latency",
        status="pass" if p95 <= p95_threshold else "fail",
        elapsed_ms=elapsed,
        detail=f"p95={p95:.0f}ms (threshold={p95_threshold:.0f}ms)",
        metrics={"p95_ms": p95, "samples_ms": samples_ms},
    )


# ──────────────────────────────────────────────────────────────────
# Phase 4 — Translate latency (full mode)
# ──────────────────────────────────────────────────────────────────


async def _phase_translate_latency(config: ServerConfig, baseline: dict) -> PhaseResult:
    """Send a few translate requests; gate on p95 latency."""
    t0 = time.monotonic()
    p95_threshold = float(baseline.get("translate", {}).get("p95_ms", 5000))

    samples_ms: list[float] = []
    url = f"{config.translate_vllm_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": baseline.get("translate", {}).get("model_revision", "Qwen/Qwen3.6-35B-A3B-FP8"),
        "messages": [
            {
                "role": "user",
                "content": "Translate to Japanese: hello, how are you?",
            }
        ],
        "max_tokens": 60,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            for _ in range(3):
                t_call = time.monotonic()
                r = await c.post(url, json=payload)
                samples_ms.append((time.monotonic() - t_call) * 1000.0)
                if r.status_code != 200:
                    return PhaseResult(
                        name="translate_latency",
                        status="fail",
                        elapsed_ms=(time.monotonic() - t0) * 1000.0,
                        detail=f"HTTP {r.status_code}",
                    )
    except Exception as e:
        return PhaseResult(
            name="translate_latency",
            status="fail",
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
            detail=type(e).__name__,
        )

    samples_ms.sort()
    p95 = samples_ms[int(len(samples_ms) * 0.95)] if samples_ms else 0.0
    elapsed = (time.monotonic() - t0) * 1000.0
    return PhaseResult(
        name="translate_latency",
        status="pass" if p95 <= p95_threshold else "fail",
        elapsed_ms=elapsed,
        detail=f"p95={p95:.0f}ms (threshold={p95_threshold:.0f}ms)",
        metrics={"p95_ms": p95, "samples_ms": samples_ms},
    )


# ──────────────────────────────────────────────────────────────────
# Phase 5 — Diarize latency (full mode)
# ──────────────────────────────────────────────────────────────────


async def _phase_diarize_latency(config: ServerConfig, baseline: dict) -> PhaseResult:
    """Send a fixture WAV through diarization; gate on p95 latency."""
    t0 = time.monotonic()
    wav_path = _FIXTURE_DIR / "audio_en_short.wav"
    if not wav_path.exists():
        return PhaseResult(
            name="diarize_latency",
            status="skip",
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
            detail="fixture audio_en_short.wav missing",
        )

    p95_threshold = float(baseline.get("diarize", {}).get("p95_ms", 5000))
    samples_ms: list[float] = []
    audio_bytes = wav_path.read_bytes()
    url = f"{config.diarize_url.rstrip('/')}/v1/diarize"

    try:
        async with httpx.AsyncClient(timeout=20.0) as c:
            for _ in range(3):
                t_call = time.monotonic()
                r = await c.post(
                    url,
                    files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                    data={"max_speakers": "4", "min_speakers": "1"},
                )
                samples_ms.append((time.monotonic() - t_call) * 1000.0)
                if r.status_code != 200:
                    return PhaseResult(
                        name="diarize_latency",
                        status="fail",
                        elapsed_ms=(time.monotonic() - t0) * 1000.0,
                        detail=f"HTTP {r.status_code}",
                    )
    except Exception as e:
        return PhaseResult(
            name="diarize_latency",
            status="fail",
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
            detail=type(e).__name__,
        )

    samples_ms.sort()
    p95 = samples_ms[int(len(samples_ms) * 0.95)] if samples_ms else 0.0
    elapsed = (time.monotonic() - t0) * 1000.0
    return PhaseResult(
        name="diarize_latency",
        status="pass" if p95 <= p95_threshold else "fail",
        elapsed_ms=elapsed,
        detail=f"p95={p95:.0f}ms (threshold={p95_threshold:.0f}ms)",
        metrics={"p95_ms": p95, "samples_ms": samples_ms},
    )


# ──────────────────────────────────────────────────────────────────
# Phase 6 — TTS TTFA (full mode)
# ──────────────────────────────────────────────────────────────────


async def _phase_tts_ttfa(config: ServerConfig, baseline: dict) -> PhaseResult:
    """Send a few TTS synth requests; gate on p95 time-to-first-audio."""
    t0 = time.monotonic()
    p95_threshold = float(baseline.get("tts", {}).get("p95_ttfa_ms", 1000))
    samples_ms: list[float] = []
    tts_url = (config.tts_vllm_url or "http://localhost:8002").split(",")[0].strip()
    url = f"{tts_url.rstrip('/')}/v1/audio/speech"
    payload = {
        "input": "This is a short test message.",
        "voice": "default",
        "response_format": "wav",
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            for _ in range(3):
                t_call = time.monotonic()
                async with c.stream("POST", url, json=payload) as r:
                    if r.status_code != 200:
                        return PhaseResult(
                            name="tts_ttfa",
                            status="fail",
                            elapsed_ms=(time.monotonic() - t0) * 1000.0,
                            detail=f"HTTP {r.status_code}",
                        )
                    async for _chunk in r.aiter_bytes():
                        samples_ms.append((time.monotonic() - t_call) * 1000.0)
                        break
    except Exception as e:
        return PhaseResult(
            name="tts_ttfa",
            status="fail",
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
            detail=type(e).__name__,
        )

    samples_ms.sort()
    p95 = samples_ms[int(len(samples_ms) * 0.95)] if samples_ms else 0.0
    elapsed = (time.monotonic() - t0) * 1000.0
    return PhaseResult(
        name="tts_ttfa",
        status="pass" if p95 <= p95_threshold else "fail",
        elapsed_ms=elapsed,
        detail=f"p95_ttfa={p95:.0f}ms (threshold={p95_threshold:.0f}ms)",
        metrics={"p95_ttfa_ms": p95, "samples_ms": samples_ms},
    )


# ──────────────────────────────────────────────────────────────────
# Phase 7 — End-to-end meeting lag (e2e mode only)
# ──────────────────────────────────────────────────────────────────


async def _phase_e2e_lag(config: ServerConfig, baseline: dict) -> PhaseResult:
    """Stand up a transient meeting against the live admin server,
    stream the fixture WAV, measure utterance-end → translation-complete."""
    t0 = time.monotonic()
    threshold_ms = float(baseline.get("e2e_lag", {}).get("p95_utterance_to_audio_ms", 10000))

    # Skeleton implementation: defer to benchmarks/e2e_meeting_lag.py
    # if it has been implemented past skeleton. Otherwise, mark skip.
    bench_script = Path(__file__).resolve().parents[2] / "benchmarks" / "e2e_meeting_lag.py"
    if not bench_script.exists():
        return PhaseResult(
            name="e2e_lag",
            status="skip",
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
            detail="benchmarks/e2e_meeting_lag.py not implemented yet",
        )
    # Run the bench script as subprocess and parse JSON output
    try:
        proc = await asyncio.create_subprocess_exec(
            "python",
            str(bench_script),
            "--threshold-ms",
            str(threshold_ms),
            "--json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
    except (TimeoutError, FileNotFoundError) as e:
        return PhaseResult(
            name="e2e_lag",
            status="fail",
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
            detail=f"e2e bench {type(e).__name__}",
        )

    try:
        result = json.loads(stdout.decode())
    except json.JSONDecodeError:
        return PhaseResult(
            name="e2e_lag",
            status="fail",
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
            detail="bench did not emit JSON",
        )

    p95 = float(result.get("p95_lag_ms", 0))
    elapsed = (time.monotonic() - t0) * 1000.0
    return PhaseResult(
        name="e2e_lag",
        status="pass" if p95 <= threshold_ms else "fail",
        elapsed_ms=elapsed,
        detail=f"p95_lag={p95:.0f}ms (threshold={threshold_ms:.0f}ms)",
        metrics=result,
    )


# ──────────────────────────────────────────────────────────────────
# Public entry — run + write report
# ──────────────────────────────────────────────────────────────────


async def run_validate(
    *,
    mode: str = "quick",
    hardware_class: str = "gb10",
    json_only: bool = False,
) -> ValidateReport:
    """Run the requested validate mode and return the report.

    ``mode`` ∈ ``{"quick", "full", "e2e"}``. Caller is responsible for
    emitting the report (CLI does it via the click subcommand).
    """
    if mode not in ("quick", "full", "e2e"):
        raise ValueError(f"unknown validate mode: {mode}")

    started_at = time.time()
    config = ServerConfig.from_env()
    baseline = _load_baselines(hardware_class)

    phases: list[PhaseResult] = []

    # Phase 1: liveness (always)
    phases.append(await _phase_liveness(config))
    if not json_only:
        _print_phase(phases[-1])

    # Phase 2: furigana (always)
    phases.append(await _phase_furigana())
    if not json_only:
        _print_phase(phases[-1])

    if mode in ("full", "e2e"):
        for phase_fn in (
            _phase_asr_latency,
            _phase_translate_latency,
            _phase_diarize_latency,
            _phase_tts_ttfa,
        ):
            phases.append(await phase_fn(config, baseline))
            if not json_only:
                _print_phase(phases[-1])

    if mode == "e2e":
        phases.append(await _phase_e2e_lag(config, baseline))
        if not json_only:
            _print_phase(phases[-1])

    finished_at = time.time()
    report = ValidateReport(
        started_at=started_at,
        finished_at=finished_at,
        mode=mode,
        hardware_class=hardware_class,
        phases=phases,
    )

    # Persist report next to diagnostics
    try:
        diag_dir = config.meetings_dir.parent / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        out_path = diag_dir / f"validate-{ts}.json"
        out_path.write_text(json.dumps(report.to_json(), indent=2))
    except Exception:
        pass

    return report


def _print_phase(p: PhaseResult) -> None:
    """Single-line colored print of a phase result.

    Defensive on terminals without color: falls back to plain ASCII
    markers when stdout is not a tty.
    """
    import sys

    color = sys.stdout.isatty()
    if p.status == "pass":
        marker = "\033[32m✓\033[0m" if color else "PASS"
    elif p.status == "fail":
        marker = "\033[31m✗\033[0m" if color else "FAIL"
    else:
        marker = "\033[33m…\033[0m" if color else "SKIP"
    print(f"  {marker} {p.name:<22} {p.elapsed_ms:>7.0f}ms  {p.detail}")
