# scribe-tts-vllm — vllm-omni TTS container

Replaces the two-replica `scribe-tts` / `scribe-tts-2` stack with a single
continuous-batched vllm-omni instance serving Qwen3-TTS-12Hz-1.7B-CustomVoice.
Migration plan: `~/.claude/plans/golden-dreaming-koala.md`.

## Pinned provenance

| Component     | Pin                                              |
|---------------|--------------------------------------------------|
| Base image    | `vllm/vllm-openai:v0.18.0` (arm64 manifest)      |
| vllm-omni SHA | `f55ea28005f889aa5f10d4ceb694573c564da127` (v0.18.0) |
| HF model      | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`           |
| HF revision   | `0c0e3051f131929182e2c023b9537f8b1c68adfe`       |
| FA2           | `flash-attn==2.7.4.post1` (built from source on aarch64/SM121 during image build) |

The HF model snapshot is prefetched into `/models/qwen3-tts-1.7b` at build
time; `HF_HUB_OFFLINE=1` at runtime. No hub access during serving.

## Phase 1a — Supply-chain + sandbox hardening

`--trust-remote-code` is still required (Qwen3-TTS modeling code ships with
the HF repo). Pinning the revision gives reproducibility; sandboxing
contains the code-execution risk.

- **Vendor model code** (pending): snapshot `modeling_*.py` from the pinned
  HF revision into `vendored_model_code/` and review for unexpected imports
  (`subprocess`/`socket`/`ctypes`/`urllib`/`requests`/`os.system`).
  A `scripts/audit_model_code.py` CI step enforces the scan. **Status:**
  skeleton only — run once the Phase 1 spike image validates and the
  baked-in code can be extracted.
- **Runtime sandbox** (enforced in `docker-compose.gb10.yml`):
  `read_only: true`, `tmpfs` for `/tmp` + `/root/.cache/vllm`,
  `cap_drop: [ALL]`, `security_opt: no-new-privileges:true`, no external
  egress (container uses host networking — HF offline flags prevent hub
  reach), zero secret env vars (no `HF_TOKEN`/`GH_TOKEN`/`CLOUDFLARE_API_TOKEN`
  inherited), minimal mounts (no host volumes — model is baked in).

## Risk acceptance

The `--trust-remote-code` attack surface is reduced to: a one-time build
against a pinned commit, with the executed code sandboxed as above, no
writable filesystem paths that persist, and no credentials reachable from
inside the container. Reviewer + date recorded on first landing PR.

## Phase 2 — Lockfile + wheelhouse (deferred until first spike success)

The current `Dockerfile` does a full dependency resolve at build time
(same as `Dockerfile.spike`). Once the spike image boots and passes the
smoke test, run:

```bash
docker run --rm scribe-tts-vllm:spike cat /workspace/requirements.lock > requirements.lock
docker run --rm scribe-tts-vllm:spike tar -C /workspace/wheelhouse -c . | zstd > wheelhouse.tar.zst
```

…and switch the `Dockerfile` to `pip install --no-index --find-links=wheelhouse --require-hashes -r requirements.lock`
for fully-frozen rebuilds.
