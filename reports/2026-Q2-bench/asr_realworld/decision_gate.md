# Real-world ASR bench — Cohere vs Qwen3-ASR on production meetings

Aggregate: **440** chunks across **4** meetings · **402** with speech, **38** near-silent (skipped from disagreement stats).
Errors: Qwen3-ASR **0**, Cohere **392**.

## Char-level disagreement between backends (per language)

| Language | n | p50 | p95 | mean |
|---|---:|---:|---:|---:|
| ? | 5 | 0.0% | 0.0% | 0.0% |
| en | 294 | 100.0% | 100.0% | 98.4% |
| ja | 103 | 100.0% | 100.0% | 95.2% |

_This is char-level Levenshtein between Qwen3-ASR's and Cohere's transcripts on the same chunk.  No ground truth needed.  Higher = backends disagree more on what was said._

## Each backend vs the meeting journal (production-ASR-at-record-time)

Each chunk's transcripts compared against the journal events that overlap the chunk window by ≥ 50 %.  This is _not_ ground truth — the journal IS Qwen3-ASR's output at record time — but it lets us see how much each backend would diverge from what the production pipeline produced.

### Qwen3-ASR vs journal (consistency check on its own past output)

| Language | n | p50 | p95 | mean |
|---|---:|---:|---:|---:|
| en | 3 | 100.0% | 100.0% | 100.0% |
| ja | 2 | 81.3% | 84.6% | 81.3% |

### Cohere vs journal (would-have-been-this-instead)

| Language | n | p50 | p95 | mean |
|---|---:|---:|---:|---:|
| en | 3 | 100.0% | 100.0% | 87.0% |
| ja | 2 | 100.0% | 100.0% | 100.0% |

## Latency (speech chunks only)

| Backend | p50 (ms) | p95 (ms) | n |
|---|---:|---:|---:|
| Qwen3-ASR-1.7B | 1352 | 1956 | 402 |
| Cohere Transcribe | 182 | 845 | 47 |

## Per-meeting summary

| Meeting | Chunks | Speech | Q errors | C errors | Disagreement p50 | Disagreement p95 |
|---|---:|---:|---:|---:|---:|---:|
| `fe77b412` | 56 | 55 | 0 | 8 | 93.2% | 100.0% |
| `73d3fbbd` | 68 | 60 | 0 | 68 | 100.0% | 100.0% |
| `4cee0e9b` | 134 | 106 | 0 | 134 | 100.0% | 100.0% |
| `d34ca2f0` | 182 | 181 | 0 | 182 | 100.0% | 100.0% |

## Top-30 highest-disagreement chunks (where the backends saw very different things)

| Meeting | Window | RMS | Lang | Disagreement | Qwen text (truncated) | Cohere text (truncated) |
|---|---|---:|---|---:|---|---|
| `fe77b412` | 0s–20s | 0.085 | ja | 100.0% | 啊，喂！哦，有红色的。啊，这、啊，那、那什么，有初生婴儿的，都给我记着。是。初生婴儿。这个、这个、这个。 | Sorry. Oh, you're not even there. I'm just gonna say, I'm ju |
| `fe77b412` | 342s–362s | 0.049 | en | 100.0% | बहुत अच्छी रेपेज मारे | Who does he depend? Marry. |
| `fe77b412` | 360s–380s | 0.049 | ja | 100.0% | 奶奶，这个蘑菇棒子鸡蛋，爸爸，爸爸，最爱吃的呀。哦，好厉害呀！你看，你看，会跳舞。小乖乖，西瓜，西瓜，西瓜，甜西瓜，真甜 | I'm not sure if I'm going to do that. I'm not sure if I'm go |
| `fe77b412` | 378s–398s | 0.064 | ja | 100.0% | 他妈的，你说你说关机，我不能删，怎么搞？嗯，我爸我爸我爸把关了。我爸，谁？我爸。え、アナも？アナもこれ我爸，なんかダビィ | About the bar, about the bar, about the bar. They go about i |
| `fe77b412` | 504s–524s | 0.257 | en | 100.0% | (empty) | I got cooking. I'll stop. I'll stop. I'll stop. I'll stop. I |
| `fe77b412` | 540s–560s | 0.018 | en | 100.0% | (empty) | Hey, Chacha, what's up, Solovy? |
| `fe77b412` | 558s–578s | 0.032 | en | 100.0% | (empty) | I don't know what I'm doing. I don't know what I'm doing. I  |
| `fe77b412` | 576s–596s | 0.040 | ja | 100.0% | 哎，哦哦哦，那是不是牛翅膀呢？哦。 | It's too easy to take with the water. I don't know if that's |
| `fe77b412` | 594s–614s | 0.072 | en | 100.0% | (empty) | I'm a young man. I'm a young man. I'm a young man. I'm a you |
| `fe77b412` | 612s–632s | 0.082 | ja | 100.0% | 哎嘿嘿。哎呀。来呀。啊，哪里买的？哎呀。还有吗？哎呦，给我发的。 | I think it's... I think it's... I think it's... No, I think  |
| `fe77b412` | 630s–650s | 0.064 | en | 100.0% | (empty) | I'm not sure if I'm going to be able to do that. I'm not sur |
| `fe77b412` | 666s–686s | 0.078 | ja | 100.0% | 啊？ | I'm not going to tell you what I'm going to tell you. I'm go |
| `fe77b412` | 684s–704s | 0.080 | en | 100.0% | (empty) | I'm going to go to the hospital. |
| `fe77b412` | 702s–722s | 0.048 | en | 100.0% | (empty) | I'm going to go to the hospital. I'm going to go to the hosp |
| `fe77b412` | 756s–776s | 0.034 | ja | 100.0% | 你是小白兔吗？啊，可以。啊，唔。 | I can't. |
| `fe77b412` | 810s–830s | 0.029 | en | 100.0% | (empty) | I'm going to go to the hospital. I'm going to go to the hosp |
| `fe77b412` | 846s–866s | 0.020 | ja | 100.0% | 没查到。 | (empty) |
| `fe77b412` | 882s–902s | 0.040 | en | 100.0% | Let's go, baby. Let's go, let's go, let's go, let's go, let' | (empty) |
| `fe77b412` | 900s–920s | 0.069 | ja | 100.0% | マースって今大学月から聞いて言ってんの？はい。あ、そうなんや。朝から岩田さん。はい。マースってやっぱりすごいよね。マッチ | (empty) |
| `fe77b412` | 918s–938s | 0.045 | ja | 100.0% | 何回生？うん。マーチャン何回生ぐらい？一回だったんじゃ。いや私三回までに全部単位と。よかれせはもう単位でいう状態にして。 | (empty) |
| `fe77b412` | 936s–956s | 0.094 | en | 100.0% | ममा ममा रासलियोग अंसाक से लायक गोलियों की घटना हो जाता है वो | (empty) |
| `fe77b412` | 954s–974s | 0.068 | en | 100.0% | Mommy, mommy. | (empty) |
| `fe77b412` | 972s–992s | 0.054 | ja | 100.0% | もうまあ生まれて僕帰る。僕明日ね。明日。明日。明日。明日。明日。明日。明日。明日。明日。明日。明日。明日。明日。明日。明 | (empty) |
| `fe77b412` | 990s–1006s | 0.066 | en | 100.0% | Okay, let's go. | (empty) |
| `73d3fbbd` | 0s–20s | 0.029 | en | 100.0% | Look, yes, testing one two three, okay, okay, good, love it, | (empty) |
| `73d3fbbd` | 18s–38s | 0.017 | en | 100.0% | Compact table table doesn't come on. Okay. Dun dun dun. | (empty) |
| `73d3fbbd` | 36s–56s | 0.031 | en | 100.0% | Den den den den den den den den den. | (empty) |
| `73d3fbbd` | 72s–92s | 0.044 | en | 100.0% | Stop! Stop! Test. Do testing. Do do testing. Orders. Hard li | (empty) |
| `73d3fbbd` | 90s–110s | 0.035 | en | 100.0% | We need to optimise the back end prioritisation. | (empty) |
| `73d3fbbd` | 108s–128s | 0.021 | en | 100.0% | Testing one two three. | (empty) |

---

_Char-level disagreement is a structural similarity metric — it does not tell you which transcript is correct.  It DOES tell you where the two backends would have written different transcripts on the same audio.  Use it to characterize the failure modes (long utterances, code-switching, rare vocabulary, near-silence) and decide whether one backend's strengths are worth the other's weaknesses on real production audio._
