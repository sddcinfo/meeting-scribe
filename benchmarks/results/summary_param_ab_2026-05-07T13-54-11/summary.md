# Summary LLM-param A/B

Configs:
- **baseline_thinking_8192** — enable_thinking=True, max_tokens=8192
- **fast_no_thinking_4096** — enable_thinking=False, max_tokens=4096

## Per-meeting comparison

| meeting | events | A latency (ms) | B latency (ms) | speedup | A topics | B topics | A actions | B actions | A resp chars | B resp chars | topic Jaccard |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `38d4efb0…` | 6 | 62524 | 9371 | 6.67x | 1 | 1 | 0 | 0 | 2884 | 1526 | 0.00 |
| `6934dd1b…` | 89 | 120212 | 33767 | 3.56x | 6 | 4 | 3 | 3 | 7166 | 6337 | 0.00 |
| `c4c913f8…` | 179 | 125277 | 38466 | 3.26x | 5 | 6 | 4 | 4 | 9979 | 7513 | 0.00 |
| `3ec3823a…` | 452 | 148611 | 44796 | 3.32x | 10 | 6 | 4 | 3 | 10666 | 7889 | 0.00 |
| `3db4286e…` | 782 | 180107 | 62032 | 2.90x | 0 | 6 | 0 | 4 | 0 | 10460 | 0.00 |
| `5029650e…` | 1119 | 146331 | 43824 | 3.34x | 8 | 6 | 4 | 4 | 10055 | 7256 | 0.00 |

## Aggregate

- avg speedup A→B: **3.84x**
- avg topic Jaccard A vs B: **0.00** (1.0 = identical topic sets, 0.0 = disjoint)
- meetings where topic count differs by >1: **4**
- meetings where action_item count differs by >1: **1**
- meetings with hallucinated speakers — A: **2**, B: **2**

## Decision rule

Re-apply the change ONLY if avg speedup ≥ 2.0× AND avg topic Jaccard ≥ 0.65 AND no new hallucinated speakers in B. Otherwise keep the production config and document the tradeoff.
