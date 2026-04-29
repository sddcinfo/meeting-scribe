# 2026-Q2 Model-Challenger Bench — Retrospective + Next Steps

*2026-04-28 — written immediately after the pyannote 4.0 / community-1 production swap landed.*

This doc consolidates everything we learned across Tracks A, B, C of the bench and ties it to the deep-embed of community-1 we just shipped.  It pairs with:

- `reports/2026-Q2-bench/asr_cohere/decision_gate.md` (Track A)
- `reports/2026-Q2-bench/tts_funcosyvoice/decision_gate.md` (Track B)
- `reports/2026-Q2-bench/diarize_community1/decision_gate.md` (Track C)
- `benchmarks/MODEL_SELECTION.md` (production state of record)

## TL;DR — where we ended up

| Component | Production today | Why |
|---|---|---|
| **Diarization** | **pyannote 4.0.4 + `pyannote/speaker-diarization-community-1`** (NEW — promoted 2026-04-28) | Track C cleared every gate; 100 % overlap-time resolution, +2.1 % wall-clock. Deep-embedded; 3.1 code paths deleted. |
| **ASR** | **Qwen3-ASR-1.7B** (unchanged) | Track A was DEFER (lean-promote): point estimate strongly favored Cohere (-3.64 pp WER, 3× faster), but the bootstrap CI on the small ceiling-bound corpus straddled zero, so the rigorous gate failed. No production change yet. |
| **Translation** | **Qwen3.6-35B-A3B-FP8** (unchanged from 2026-04-18) | Out of bench scope this round. |
| **TTS** | **Qwen3-TTS-12Hz-0.6B-Base** (unchanged) | Track B challenger (Fun-CosyVoice 3.5) does not exist as open weights — the 2026-03-02 announcement was Tongyi-platform-only. Pivot to 3.0 deferred: "no credible open-weights TTS challenger exists today" was the Track B finding. |

## Track-by-track learnings

### Track C — pyannote 4.0 / community-1 (PROMOTED, deep-embedded)

**The win.**  The exclusive-mode output (`DiarizeOutput.exclusive_speaker_diarization`) resolves 100 % of the previously-overlapping seconds to single-speaker assignment on a real 40-min 4-speaker production meeting.  That's exactly the input shape `speaker_attach._attach_speakers_to_events` needs for clean STT-timestamp reconciliation in the refinement worker — historically our #1 source of refinement-side mis-attribution.

**Five things that surprised us / cost time.**

1. **The HF attribute is `exclusive_speaker_diarization`, not `exclusive_diarization`.**  We coded against the bare informal name from the model card.  The first production-gated emission was a no-op because `getattr(...)` returned `None` on every request.  Five-minute fix once we did `dir(DiarizeOutput)` inside the live container — but the wrong name had a chilling effect on the 30 minutes preceding.  Lesson: when integrating a brand-new class, dump `dir()` once before assuming any attribute name.
2. **Library version vs pretrained-pipeline version is a real distinction.**  `pyannote.audio==4.0.4` had been on disk for weeks (it's the latest stable on PyPI, released 2026-02-07).  What was *old* was the pretrained pipeline name (`pyannote/speaker-diarization-3.1`), not the library.  Many "we're behind on pyannote" reflexes confused the two — including the original 2026-04-28 market survey, which framed Track C as "library upgrade" when it was really "pretrained-pipeline swap".  Our new `scripts/bench/check_stale_pins.py` workflow only catches library drift (PyPI), not pretrained-model drift.  See "Future tooling" below.
3. **Blackwell SM_121 patches generalised cleanly.**  The `one_hot` CPU fallback + `get_device_capability` Hopper spoof in `containers/pyannote/server.py` covered community-1 unchanged.  That was the biggest implementation risk going in, and it just worked.  Worth noting because the comment in `server.py:51-95` documents the wedge in detail; future maintainers can trust it.
4. **Exclusive output ships without embeddings; standard does.**  This shaped the design.  Cross-chunk cluster stitching (in `_merge_clusters_via_embeddings`) requires embeddings, so it operates on standard segments; the merge map is then projected onto exclusive so both arrays end up with the same global cluster ids.  Don't try to merge exclusive segments directly — it can't work.
5. **Cross-talk detection still requires the standard segmentation.**  The exclusive output, by design, collapses two-speaker windows to a single owner.  That's exactly what we want for primary assignment but NOT what we want for "did A and B speak at the same time?".  Both arrays in the response, both used in `speaker_attach`.

**The deep-embed change set (this one).**  We removed the 3.x compat path entirely:

- `containers/pyannote/server.py`: dropped the `from pyannote.core import Annotation` + `if isinstance(raw_output, Annotation)` branch.  4.x always returns `DiarizeOutput`.  The opt-in `X-Include-Exclusive: true` header is gone — `exclusive_segments` is part of every response.  `DIARIZE_PIPELINE_ID` defaults to community-1 (env-var override kept as a roll-back lever, not a config knob).
- `src/meeting_scribe/pipeline/diarize.py`: `_diarize_single_call` returns `(standard, exclusive)`.  `_merge_clusters_via_embeddings` projects the merge map onto exclusive so cluster ids line up.  `_diarize_full_audio` returns a `DiarizeResult(segments, exclusive_segments)` dataclass (breaking shape change vs the old list-of-segments return).
- `src/meeting_scribe/pipeline/speaker_attach.py`: primary speaker per ASR event is now picked from `exclusive_segments` (every frame has exactly one speaker, by construction).  Cross-talk detection still uses standard segments.  Minority-speaker rescue cross-references the union of both.
- `src/meeting_scribe/routes/meeting_lifecycle.py` (×2 callsites) + `src/meeting_scribe/reprocess.py`: updated to unpack `DiarizeResult` and pass both arrays into `_attach_speakers_to_events`.
- `src/meeting_scribe/recipes/sortformer-4spk.yaml`: `model_id` flipped to `pyannote/speaker-diarization-community-1`.
- `docker-compose.gb10.yml`: production `pyannote-diarize` explicitly pins community-1 via `DIARIZE_PIPELINE_ID`.  Bench-profile `pyannote-diarize-c1` and `pyannote-diarize-c1-scratch` services **removed** — production IS community-1 now, so the parallel bench sidecars are redundant.
- `scripts/validate_how_it_works.py`: validation list flipped from "speaker-diarization-3.1" to "speaker-diarization-community-1".

**Live verification (this session).**

- `docker compose build pyannote-diarize` rebuilt the image clean.
- Force-recreated `scribe-diarization`; container logs show `Loading pyannote/speaker-diarization-community-1 ...` + `Pipeline loaded on cuda` ✅.
- Hit `POST /v1/diarize` with a real 4-min meeting → 200 OK, response keys `[segments, exclusive_segments, num_speakers, audio_duration_s, processing_ms]`, 85 standard + 85 exclusive segments, 12.2 s processing, 2 detected speakers.
- `meeting-scribe restart` smoke test green: ASR probe 108 ms, translate probe 168 ms, diarize health 200, TTS 200.
- Full pytest sweep: 18 / 18 diarize + speaker_attach tests pass; 932 / 933 non-integration tests pass (the 1 failure is `test_meeting_integrity` baseline drift on two recent meetings unrelated to my changes — `git diff` confirms no integrity-test files were touched).
- Production GPU at 91.1 %; no meetings in flight throughout the swap.

### Track A — Cohere Transcribe (DEFER, lean-promote)

**The headline.**  On 36 JA utterances synthesized from the `en_ja_meeting_v2` corpus through Qwen3-TTS:

| Metric | Qwen3-ASR-1.7B | Cohere Transcribe (03-2026) | Δ |
|---|---:|---:|---:|
| Corpus-level char WER (JA) | 9.23 % | **5.59 %** | **+3.64 pp** (Cohere) |
| p50 total_ms | 494 ms | **151 ms** | **3.27× faster** |
| p95 total_ms | 1261 ms | 506 ms | 2.49× faster |
| Bootstrap 95 % CI on (Qwen − Cohere) | n/a | n/a | **[−2.22 pp, +10.10 pp]** — straddles zero |

The point estimate strongly favors Cohere on every dimension.  The CI straddles zero only because the corpus is small (36 samples after the JA filter) and ceiling-bound (most samples score 0 % WER for both backends — Qwen3-TTS produces phonetically-clean speech).  The plan's strict gate ("Δ ≥ 0.5 pp AND CI excludes zero") is conservatively-correct for the methodology, so the formal outcome is DEFER.

**Three real bugs caught along the way.**

1. **Cohere uses a custom `CohereAsrForConditionalGeneration`**, not `AutoModelForSpeechSeq2Seq`.  My first server.py loader was wrong; first 500 on bringup told us.
2. **`transformers>=5.4.0` required**, not `4.57+` as I'd pinned (the original pin was based on a stale forum post).  ImportError on bringup told us.
3. **`accelerate` required** when using `device_map="auto"`.  Second ImportError on bringup told us.
4. **The corpus has both directions.**  My first script scored every pair against `reference_text`, which is *English* for the `je_*` (JA→EN) pairs.  Cohere transcribed the JA audio cleanly; my scorer compared its JA output to an EN reference, getting 100 % WER on every `je_*` pair.  Fix: filter on `source_lang/target_lang == "ja"` to pick the JA utterance text regardless of direction.  This is a script-robustness lesson, not an ASR bug — but it's the kind of thing that makes a "real" bench result confidence-cratering until you find it.

**The methodology limitation that matters most.**  TTS-synthesized audio is too easy.  Qwen3-TTS produces phonetically-clean Japanese; both ASR backends transcribe most of it perfectly; the WER signal is dominated by a small tail of rare-token misses.  Real conversational JA (CommonVoice, Fleurs, internal recordings with consented redaction) would broaden the variance and probably narrow the CI substantially.  The recommended path to a clean PROMOTE is exactly that re-run.

**Pin Cohere has at run time** — for reproducibility:

- Model: `CohereLabs/cohere-transcribe-03-2026 @ 0a928bea9c35ac5fa6c03d732311e7ba75acd3be`
- Container: `cohere-transcribe:bench` (built 2026-04-28 from `containers/cohere-transcribe/`)
- License: Apache 2.0 (same as Qwen3-ASR-1.7B — no licensing differentiator).

### Track B — Fun-CosyVoice 3.5 (HARD PASS)

**The challenger does not exist as open weights.**  The 2026-04-28 market survey listed Fun-CosyVoice 3.5 as Apache 2.0 based on news coverage (gaga.art, pandaily).  The bench window upstream check turned up:

- `fun-cosyvoice` returns HTTP 404 on PyPI.
- The `FunAudioLLM/CosyVoice` GitHub repo only ships through 3.0 (December 2025; latest open weights `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`).
- GitHub issue #1840 is community asking when 3.5 will open-source.  As of 2026-04-28 the answer is unknown.

So 3.5 belongs in the same hard-pass bucket as MiMo-V2.5-TTS — Tongyi-platform-only.  This was a mis-classification in the original survey: a news headline said "Fun-CosyVoice 3.5 release", but the *open* release was 3.0; 3.5 was the platform launch.  Lesson for the next survey: *PyPI 404 / GitHub-tag absent / model card unreachable* are all hard signals that a "release" headline is platform-only, not open.

**No credible open-weights TTS challenger exists today** beyond the older models we already evaluated in our April baseline.  Production stays on Qwen3-TTS-12Hz-0.6B-Base.  Watch list:

- **Higgs-Audio V3** — Boson AI roadmap shows streaming TTS in Q2 2026.  Re-evaluate when V3 actually lands.
- **Fun-CosyVoice 3.5** — set a quarterly upstream check on the FunAudioLLM/CosyVoice GitHub releases page.

**Container scaffolding kept.**  `containers/funcosyvoice/{Dockerfile,requirements.txt}` and the bench-profile entry in `docker-compose.gb10.yml` were built before the upstream check turned up the gap; both are kept as ready-to-go infrastructure for whenever 3.5 actually open-sources.  No code paths reference them today.

## Cross-cutting bench-tooling learnings

These shaped the work but apply generally:

1. **Strict gates catch real ambiguity.**  Track A's "Δ ≥ 0.5 pp AND CI excludes zero" rule was rigorously correct for the corpus size, but the conclusion is "we need a better corpus", not "Cohere is bad".  The reducer (`scripts/bench/reduce_to_decision_gate.py`) emitted the right verdict; the writeup adds the lean-promote nuance.
2. **Compose `down` doesn't honor profile filtering for shared networks.**  We tore down `scribe-asr` + `scribe-tts` together with the bench sidecars in Track A — production was unavailable for ~30 s before we noticed and ran `up -d vllm-asr`.  Lesson: bring sidecars down individually (`docker stop <name>`) rather than `compose --profile bench down`.
3. **The 2026-Q2 leak-prevention rules held.**  `decision_gate.md` is the only artifact that lands in the repo; per-sample WAVs / JSON / hypotheses live offline at `/data/meeting-scribe-fixtures/bench-runs/2026-Q2/`.  `git status` after the bench showed only the expected file types.  The path validator in `benchmarks/_bench_paths.py` caught one operator typo during the run.
4. **DER-as-a-hard-gate is a trap when you don't have RTTMs.**  Track C deliberately omitted DER from its pass criteria.  A "DER number" computed against an unlabeled corpus would have been defensible-looking and meaningless.

## Future tooling we wished we had during this bench

These are *gaps surfaced by doing the work*, not new feature wishes.

- **Pretrained-model-version drift detector.**  `scripts/bench/check_stale_pins.py` catches `pyannote.audio==4.0.4` drift on PyPI but not "the *pipeline* `pyannote/speaker-diarization-3.1` has a successor".  The HF Hub doesn't have a direct "this model has a newer version" relation, but we could maintain a curated `models-watchlist.yaml` listing the model ids we depend on, then poll HF for *any* recent release in the same `org/<family>` namespace that matches a regex.  Catches "community-1 is out, you're still on 3.1" automatically.
- **Public-domain JA ASR test corpus, on disk.**  Track A's CI gate failed because of corpus size + ceiling.  A small (≥ 100 utterances) curated subset of CommonVoice JA / Fleurs JA / JSUT, with sha256 manifest, would be reusable for the next ASR challenger evaluation.  Today every ASR re-run starts from "but we don't have a real corpus".
- **A way to A/B diarization without standing up a parallel container.**  Track C's first attempt was to spin up `pyannote-diarize-c1` on a bench port, which conflicted with the autosre Anthropic proxy, then required image rebuilds for two attribute-name iterations.  In hindsight a thin in-process loader (the `scripts/bench/diarize_compare.py` we ended up writing) would have been faster from the start.  Generalize for the next pipeline-id swap.
- **A bench-window orchestrator.**  Today the bench window is "set env var, edit `/tmp/scribe-bench-window.txt`, start the SLO probe in a side terminal".  This is fine for one operator but error-prone (one missed probe-start is what almost cost us during Track A).  A `meeting-scribe bench start <reason> --duration 90m` command that owns the env var, the reason file, and the SLO probe lifecycle would be a small, useful CLI add.

---

# Articulated next steps (for user review before any detailed plan)

The pyannote 4.0 deep-embed is done.  Below are the candidate next steps in priority order — each one is a single-track effort, scoped to ~1–3 days of focused work.  None is started yet; this list is for **your review and re-prioritization** before we commit to a detailed plan for any of them.

## Tier 1 — Direct follow-ups to the work that just landed

**1. Real-corpus re-run for Cohere Transcribe (convert Track A DEFER → PROMOTE / REJECT).**
- Why now: the bench is fresh, the container is built and pinned (`cohere-transcribe:bench` @ HF revision `0a928bea`), and the methodology lessons are fresh enough to fix in one pass.
- What it takes: pull Fleurs JA + LibriSpeech `test-clean` EN as our reusable public-domain corpus (one-time, lives offline at `/data/meeting-scribe-fixtures/`); re-run the existing harness; expect the bootstrap CI to narrow substantially.
- Decision either way is real progress: if Cohere clears the gate, we start staging it on the refinement-side ASR path (async, less SLO-sensitive); if not, we close the track and stop the periodic re-evaluation.
- Risk: low — the only production touch is bringing the existing bench sidecar back up under the bench profile.

**2. Stage the refinement-side ASR path on Cohere (independent of #1's outcome).**
- Why now: the refinement worker calls ASR async (not on the SLO-critical live path), and Cohere's 3× latency win shows up most cleanly there.  Even if Cohere's WER is a wash with Qwen3, the latency win on the async path is real.
- What it takes: a small `meeting_scribe.refinement` change: a config flag + URL for the refinement ASR backend (separate from the live-path ASR URL); pilot behind an off-by-default flag for one week.
- Risk: low — refinement is async + per-event, no real-time SLO.

**3. Production-side write of `exclusive_segments` to the meeting timeline.**
- Why now: the new `speaker_attach` consumes `exclusive_segments` in memory but we don't persist them.  Saving them alongside `segments` in the meeting artifact (e.g., `timeline.json`) would let the UI show a clean single-speaker timeline (which is what users actually expect on a "who's speaking" timeline) instead of the standard segmentation's overlap blobs.
- What it takes: a one-file change in `_generate_speaker_data` (or wherever `timeline.json` is built) to write the exclusive array; UI consumers updated to read it.
- Risk: medium — touches the UI contract.  Would benefit from a schema-versioning bump on `timeline.json`.

## Tier 2 — Tooling debt the bench surfaced

**4. Pretrained-model-version drift watcher.**
- A YAML watchlist of HF model ids we depend on (Qwen3-ASR-1.7B, Qwen3.6-35B-A3B-FP8, Qwen3-TTS-12Hz-0.6B-Base, pyannote/speaker-diarization-community-1) plus a polling script that queries `huggingface_hub.list_repo_refs` for newer tagged revisions in the same `org/<family>` namespace.  Wired into the existing weekly `dependency-freshness` GitHub Action.
- Why now: the pyannote 3.1 → community-1 lag (Apr 22 release → Apr 28 promote) was caught by manual market survey, not automation.  Other model upgrades will land without us noticing.

**5. Public-domain JA ASR corpus on disk (one-time pull).**
- Pull a curated ≥ 100-utterance subset of Fleurs JA + LibriSpeech `test-clean` EN, with a manifest at `benchmarks/fixtures/meeting_consolidation/MANIFEST.yaml` cited by sha256 (not raw audio).
- Why now: every future ASR evaluation needs this; we ate the corpus-prep cost in Track A, no point re-paying it.

## Tier 3 — Watch-list reminders

**6. Quarterly upstream check on the closed-source TTS candidates.**
- Specifically: Higgs-Audio V3 (Q2 2026 expected) and FunAudioLLM/CosyVoice 3.5 open-weights status.  A simple cron job that checks the GitHub releases page + HF Hub for new tagged revisions in those orgs.
- Why now: we've shelved Track B until something open-sources; we don't want to re-discover availability via the next survey.

## Tier 4 — Out-of-bench-scope but worth flagging

**7. Bench-window orchestrator CLI.**
- A `meeting-scribe bench` subcommand that owns the lifecycle (env var, reason file, SLO probe, abort handler).  Replaces the current "edit shell + edit file + start probe in a separate terminal" ritual.
- Why now: the next bench will be cheaper if the harness handles the safety scaffolding.

**8. Translation-side Qwen3.6-Omni evaluation when/if it open-sources.**
- Out of scope for this bench (no open Omni weights exist in the 3.6 family today; the open Qwen-Omni is the older 30B-A3B from September 2025).  Re-evaluate when Alibaba ships an open Omni in the 3.6+ generation — that would unify ASR + translate + TTS in one model and meaningfully change the stack architecture.

---

# What I'm asking from you before any detailed planning

Pick a priority order from #1–#8 above (or add tasks I missed).  My recommendation for the next-up item is **#1 (real-corpus Cohere re-run)** because:

- It directly converts Track A's DEFER into a clear decision.
- The infra is already built — same container, same harness, just a different audio source.
- Shipping the result either way removes Track A from the watch list permanently.

If you'd rather invest in the tooling tier first (#4 + #5), that's a defensible choice — it makes every subsequent bench cheaper.  My counter-argument is that we have one concrete decision sitting at "needs more data", and finishing it is more useful right now than pre-paying for the next decision.

Tell me which item(s), in what order, and I'll come back with a focused plan for the top one.
