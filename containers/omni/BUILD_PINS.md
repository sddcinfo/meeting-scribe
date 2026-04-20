# Omni container — build pins

| Component     | Pin                                                    | Source                                          |
|---------------|--------------------------------------------------------|-------------------------------------------------|
| Base image    | `vllm/vllm-openai:v0.18.0` (arm64 manifest)            | Docker Hub                                      |
| vllm-omni     | `v0.18.0` / SHA `f55ea28005f889aa5f10d4ceb694573c564da127` | GitHub `vllm-project/vllm-omni`             |
| HF model      | `Qwen/Qwen3-Omni-30B-A3B-Instruct`                     | Hugging Face                                    |
| HF revision   | `26291f793822fb6be9555850f06dfe95f2d7e695`             | Resolved 2026-04-13 via `huggingface.co/api/models/...` |
| FA2           | `flash-attn==2.7.4.post1` (built from source on aarch64/SM121 if base image lacks it) | PyPI / source |
| Stage config  | `qwen3_omni_moe.yaml` (vendored from vllm-omni v0.18.0) | in-repo at `src/meeting_scribe/stage_configs/`  |

## Required build arg

```bash
docker build \
  --build-arg HF_MODEL_REVISION=26291f793822fb6be9555850f06dfe95f2d7e695 \
  -t scribe-omni:spike containers/omni/
```

`HF_MODEL_REVISION` has no default — the build fails loudly if you forget
it. Benchmarks (Phase A baseline, Phase C Omni run), spike smoke tests,
and cutover all consume the same SHA. Rotate the pin by bumping
`HF_MODEL_REVISION` across this file, any compose overrides, and the
relevant `benchmarks/results/*` filenames.

## Sandboxing (enforced in docker-compose.gb10.yml)

- `read_only: true` with `tmpfs` for `/tmp` + `/root/.cache/vllm`
- `cap_drop: [ALL]` + `security_opt: no-new-privileges:true`
- `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` (model baked in)
- No secret env (no `HF_TOKEN`/`GH_TOKEN`/`CLOUDFLARE_API_TOKEN` inherited)
- No host volume mounts (model is at `/models/qwen3-omni-30b-a3b` inside the image)
