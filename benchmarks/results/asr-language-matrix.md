# Qwen3-ASR-1.7B — per-language quality matrix

**Run:** 2026-04-29 on GB10 (NVIDIA DGX Spark)
**Image:** `scribe-vllm-asr:latest` (stock `vllm/vllm-openai:latest` + `vllm[audio]`)
**Corpus:** Google Fleurs test split (Apache-2.0), 30 samples per language (en + ja: 100 each)
**Scoring:** Levenshtein edit distance against fleurs reference, normalized (NFKC, lowercased, punctuation stripped). Word-level (WER) for space-segmented languages, character-level (CER) for `cmn` / `ja` / `th` (with whitespace stripped to defeat fleurs's per-character spacing artifact).
**Hint:** request includes `Transcribe the <Language> audio. Do not translate.` — same prompt the live path uses.

| Language | Code | n | Metric | p50 err | p95 err | p50 latency | p95 latency |
|----------|------|---|--------|---------|---------|-------------|-------------|
| German | de | 30 | WER | 0% | 20% | 991 ms | 1398 ms |
| Spanish | es | 30 | WER | 0% | 7% | 725 ms | 1759 ms |
| Indonesian | id | 30 | WER | 0% | 25% | 773 ms | 1824 ms |
| Italian | it | 30 | WER | 0% | 6% | 1032 ms | 1259 ms |
| Korean | ko | 30 | WER | 0% | 30% | 1049 ms | 1429 ms |
| Russian | ru | 30 | WER | 0% | 40% | 991 ms | 1749 ms |
| Chinese (Simplified) | zh / cmn | 30 | CER | 1% | 40% | 696 ms | 927 ms |
| Vietnamese | vi | 30 | WER | 1% | 16% | 881 ms | 1381 ms |
| Portuguese (BR) | pt | 30 | WER | 1% | 23% | 1252 ms | 2724 ms |
| Japanese | ja | 100 | CER | 3% | 20% | 839 ms | 1550 ms |
| English (US) | en | 100 | WER | 3% | 18% | 665 ms | 1020 ms |
| French | fr | 30 | WER | 4% | 15% | 1020 ms | 2012 ms |
| Thai | th | 30 | CER | 5% | 17% | 1508 ms | 3035 ms |
| Dutch | nl | 30 | WER | 5% | 28% | 1074 ms | 1709 ms |
| Hindi | hi | 30 | WER | 8% | 43% | 2113 ms | 4136 ms |
| Turkish | tr | 30 | WER | 10% | 52% | 972 ms | 1523 ms |
| Arabic (Egyptian) | ar | 30 | WER | 11% | 44% | 1129 ms | 1773 ms |
| Polish | pl | 30 | WER | 17% | 53% | 1231 ms | 1607 ms |
| Ukrainian | uk | 30 | WER | 18% | 84% | 1352 ms | 2332 ms |
| Malay | ms | — | — | not benchmarked | — | — | — |

## Notes

- **Malay (ms) is in our supported-languages list but was not benchmarked**: Fleurs has no `ms_my` split. Indonesian (`id`) is the closest available proxy; Malay shares ~80% lexical similarity with Indonesian, so real-world quality is expected to be in the same range as Indonesian (0% p50 WER), but this is unverified.
- **p95 tails are noisier than the medians suggest**: Fleurs includes hard samples (foreign loanwords, fast speech, accents). Real meeting audio with a consistent speaker pool will track closer to the p50.
- **Latency excludes the 16 kHz resample step** done in the live path (`torchaudio` Kaiser sinc, GPU-resident).
- **Reproduce**: `PYTHONPATH=. python benchmarks/asr_accuracy_latency.py --url http://localhost:8003 --model Qwen/Qwen3-ASR-1.7B --out <path>`. Pull fresh Fleurs fixtures with `python scripts/bench/pull_public_corpus.py --target-dir /data/meeting-scribe-fixtures --languages <split codes>`.
