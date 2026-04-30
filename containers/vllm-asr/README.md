# scribe-vllm-asr image

Thin derivative of `vllm/vllm-openai` that adds the `vllm[audio]`
extras (soundfile, librosa, av) needed for Qwen3-ASR's `input_audio`
chat-completions messages. The stock upstream image returns
*"Please install vllm[audio] for audio support"* without them.

Translation runs on the stock upstream image directly (no audio path).
Only ASR needs this layer.

## Version-pin contract

The base image is pinned by **tag + sha256 digest** (not by a floating
`:latest`). Semver tags on Docker Hub are mutable across rebuilds and
architectures, so a digest is the only deterministic deployment
artifact.

| Field             | Value                                                                          |
|-------------------|--------------------------------------------------------------------------------|
| Tag               | `v0.20.0-aarch64-cu130-ubuntu2404`                                             |
| Digest            | `sha256:f81415d5682a03566d92dc46e883d73a6ac001e6ef1fc4293512269f1372b8b9`      |
| Date pinned       | 2026-04-30                                                                     |
| Validated against | `docker-compose.gb10.yml` `--gpu-memory-utilization 0.10` (production setting) |
| Validated on      | NVIDIA GB10 (DGX Spark, aarch64 Linux, CUDA 13)                                |
| ASR memory budget | model 3.87 GiB + encoder cache ~7 GiB → 0.10 × 124 GB = 12.4 GiB cap is the working minimum (1-shared-35B autosre layout, ~43 GB free at meeting start) |

## Why we pin

On 2026-04-30 the `:latest` tag silently rolled forward from a
prior version to `v0.20.0`, exposing an unrelated config drift
(compose `0.10` vs recipe-as-written `0.04`) that under cross-tenant
CUDA workspace contention with the autosre 35B vLLM produced a
`cudaErrorNotPermitted` ASR crash and cascaded into a
`CUDNN_STATUS_EXECUTION_FAILED_CUBLAS` failure in
`scribe-diarization` during a live meeting.

The recipe's `0.04` was always aspirational — the production setting
in compose has been `0.10` since the initial release, and the
encoder-cache budget alone (~7 GiB) exceeds the `0.04` cap on this
GPU, so `0.04` could never have booted. The recipe will be corrected
to match production reality (`0.10`) under separate perf-gate review.

A digest pin makes upstream rolls explicit instead of silent. The
specific failure-mode-of-the-day was triggered by an unobserved
`:latest` roll, not by `v0.20.0` itself.

## Bump procedure

1. Run `docker pull vllm/vllm-openai:<new-tag>` and capture the new
   digest:
   ```sh
   docker inspect --format '{{index .RepoDigests 0}}' vllm/vllm-openai:<new-tag>
   ```
2. Run ASR end-to-end smoke:
   - `docker compose -f docker-compose.gb10.yml build vllm-asr`
   - `docker compose -f docker-compose.gb10.yml up -d --force-recreate vllm-asr`
   - Verify the container boots (no `No available memory for the cache
     blocks`) with the recipe's current
     `gpu_memory_utilization`. If it doesn't, the recipe value must
     be re-validated against the new version, which requires the perf
     gate (see *Out of scope* in the reliability plan).
3. Run a 5-minute meeting end-to-end and confirm ASR finals stream
   consistently.
4. Update both the `FROM` line in the Dockerfile and this README in
   the same commit. The CI drift-guard test
   (`tests/test_recipes.py::TestComposeRecipeDriftGuard`) does not
   currently assert this README's metadata against the Dockerfile,
   but reviewers should — they are intended to move together.

## Future-proofing

The autosre `package-ecosystem: docker` block in
`.github/dependabot.yml` watches this Dockerfile and opens upgrade
PRs when new tags ship. Those PRs go through normal review and the
bump procedure above; they should never be merged silently.

A committed ASR perf baseline (currently absent — only translation
has one, via autosre's `gb10_qwen36_fp8_flashinfer`) would let
future bumps pass a real perf gate. That work is tracked as a
follow-up to this plan, not implemented here.
