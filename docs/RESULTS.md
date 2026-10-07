# Results (JevBench public items, October 2026)

All numbers are measured on JevBench's **231 public decisions** (original 72, easy 48, hard 111)
through this shim and JevBench's stock `typesafe` adapter, one request at a time. They are **not
official JevBench scores**: the official board also uses the judge tier and a sealed set, neither of
which is public. Intelligence, Calibration, Capability, Speed and the JevBench Score are
public-items estimates made with [`scripts/estimate.py`](../scripts/estimate.py), following JevBench's
v1.5 method where it is stated (see that script's docstring for the exact rules and approximations).

The Jev 1.13.0 reference row is taken from the public JevBench v1.5.5 board
(benchmarkheaven.com/jev-models/v1.5.5). Its Intelligence is shown as **I_open**, the board's own
open-items-only figure, which is the like-for-like comparison.

## Setup

| Model | Build | Server | Hardware |
|---|---|---|---|
| Qwen3.8-Flash-Next | community NVFP4 build (uncensored variant), MTP speculative decoding | vLLM, TP 4 + expert parallel | 4× RTX 5090 |
| Qwen3.8-27B | NVFP4 (ModelOpt) | vLLM | 1× RTX 5090 |
| Qwen3.8-27B | unsloth UD-Q6_K GGUF | llama.cpp (ROCm) | 1× AMD Radeon AI PRO R9700 |
| Qwen3.8-27B | amd/Qwen3.8-27B-Quark-AWQ-MXFP4 | vLLM (ROCm) | 1× AMD Radeon AI PRO R9700 |
| Qwen3.6-35B-A3B | RedHatAI NVFP4 | vLLM | 1× RTX 5090 |
| Qwen3.6-35B-A3B | noctrex/Qwen3.6-35B-A3B-MXFP4_MOE-GGUF | llama.cpp (ROCm) | 1× AMD Radeon AI PRO R9700 |
| Bedrock models | AWS-hosted | AWS Bedrock Converse, us-east-1 | — |

Settings: temperature 0, thinking off unless stated, 0.3–2 s pause between requests (latency is
measured per request and excludes the pause). Latency in the "scored" columns includes JevBench's
self-hosted adjustment (×2 + 0.15 s); Jev's hosted API is measured as is.

Cost uses the cheapest public per-token price found for each base model (OpenRouter providers or
DeepInfra's catalog), applied to each run's measured tokens. JevBench's own pricing rule may apply a
higher base-model reference price; see "Pricing" below.

## Best configuration per model, no thinking

Logprobs mode, JSON answer frame.

| System | Accuracy | Hard tier | Intelligence (open) | Calibration | Capability | Median latency (raw / scored) | $ per 1k | JevBench Score |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| **Jev 1.13.0** (board) | — | — | 71.5 | 88.0 | 80.0 | 0.62 s (hosted) | $0.032 | **72.1** |
| Qwen3.8-Flash-Next | 90.9% | 91/111 | **72.4** | **88.5** | **80.5** | 0.26 / 0.68 s | $0.089 | 45.7 |
| Qwen3.8-27B (NVFP4, 5090) | 84.0% | 76/111 | 62.0 | 83.0 | 72.5 | 0.17 / 0.49 s | $0.024 | 68.3 |
| Qwen3.6-35B-A3B | 80.5% | 68/111 | 55.5 | 77.8 | 66.7 | 0.14 / 0.43 s | $0.040 | 62.6 |

Flash-Next matches Jev on open-items Intelligence and Calibration; it loses on the composite only
because of price. (The Flash-Next row predates the final longest-match scoring; the 27B and 35B rows
use it.)

## Thinking budgets

`--think-budget N`: the model thinks under the whole-reply grammar for at most N tokens, then
answers under the answer grammar. JSON format.

| Model | Mode | Budget | Accuracy | Hard | Intelligence (open) | Calibration | Capability | Latency p50 / p95 (raw) | Output tok/decision |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| Qwen3.8-27B NVFP4 | logprobs | — | 84.0% | 76 | 62.0 | 83.0 | 72.5 | 0.17 / 0.51 s | 1 |
| | logprobs | 500 | 93.5% | 97 | 84.9 | 91.1 | 88.0 | 2.18 / 6.96 s | 230 |
| | logprobs | 1000 | 95.2% | 101 | 89.4 | 93.3 | 91.3 | 2.78 / 13.2 s | 355 |
| | logprobs | 1500 | 96.1% | 102 | 92.7 | **94.8** | **93.7** | 2.97 / 19.2 s | 435 |
| | grammar | — | 87.0% | 83 | 70.0 | 78.7 | 74.3 | 0.67 / 1.06 s | 37 |
| | grammar | 500 | 94.8% | 100 | 86.4 | 89.3 | 87.8 | 4.46 / 7.58 s | 345 |
| | grammar | 1000 | 97.0% | 105 | 90.7 | 89.0 | 89.8 | 5.88 / 13.8 s | 512 |
| | grammar | 1500 | **99.1%** | **109** | **93.8** | 87.8 | 90.8 | 6.32 / 19.7 s | 606 |
| Qwen3.6-35B-A3B | logprobs | — | 80.5% | 68 | 55.5 | 77.8 | 66.7 | 0.14 / 0.40 s | 1 |
| | logprobs | 500 | 89.6% | 87 | 79.1 | 86.1 | 82.6 | 1.33 / 2.90 s | 289 |
| | logprobs | 1000 | 90.9% | 90 | 85.3 | 87.7 | 86.5 | 1.90 / 5.34 s | 483 |
| | logprobs | 1500 | 94.4% | 98 | 89.4 | 90.0 | 89.7 | 2.10 / 7.75 s | 644 |
| | grammar | 500 | 87.9% | 85 | 73.0 | 79.6 | 76.3 | 2.66 / 2.98 s | 398 |
| | grammar | 1000 | 91.8% | 92 | 81.7 | 86.6 | 84.2 | 3.33 / 5.46 s | 631 |
| | grammar | 1500 | 95.7% | 103 | 87.5 | 90.4 | 89.0 | 3.55 / 7.89 s | 804 |

Thinking lifts every model well past Jev's Capability (80.0), but costs seconds of latency and
hundreds of output tokens per decision, which collapses the JevBench composite (Speed and Cost both
fall below 50 and trigger its quadratic gates). It is a quality ceiling, not a competitive
configuration against a sub-second decision model.

## AMD MXFP4 vs NVIDIA NVFP4, with thinking

Same prompt, full-reply grammar and scoring; logprobs mode, JSON format. "Cut" counts hard items whose
thought hit the budget.

| Model | Build | Budget | Accuracy | Hard | Cut | Intelligence (open) | Calibration | Capability | Latency p50 / p95 (raw) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3.8-27B | NVIDIA ModelOpt NVFP4, vLLM, RTX 5090 | 500 | 93.5% | 97 | 47 | 84.9 | 91.1 | 88.0 | 2.18 / 6.96 s |
| | AMD Quark AWQ MXFP4, vLLM, R9700 | 500 | 93.1% | 96 | 50 | 84.3 | 89.2 | 86.8 | 6.57 / 32.2 s |
| | NVIDIA ModelOpt NVFP4, vLLM, RTX 5090 | 1000 | 95.2% | 101 | 25 | 89.4 | 93.3 | 91.3 | 2.78 / 13.2 s |
| | AMD Quark AWQ MXFP4, vLLM, R9700 | 1000 | 96.1% | 103 | 22 | 87.0 | 93.7 | 90.4 | 7.83 / 60.5 s |
| Qwen3.6-35B-A3B | NVIDIA RedHatAI NVFP4, vLLM, RTX 5090 | 500 | 89.6% | 87 | 90 | 79.1 | 86.1 | 82.6 | 1.33 / 2.90 s |
| | AMD MXFP4_MOE GGUF, llama.cpp, R9700 | 500 | 90.5% | 90 | 79 | 77.0 | 85.5 | 81.3 | 5.37 / 11.7 s |
| | NVIDIA RedHatAI NVFP4, vLLM, RTX 5090 | 1000 | 90.9% | 90 | 65 | 85.3 | 87.7 | 86.5 | 1.90 / 5.34 s |
| | AMD MXFP4_MOE GGUF, llama.cpp, R9700 | 1000 | 93.1% | 96 | 60 | 84.6 | 89.6 | 87.1 | 9.09 / 21.2 s |

Accuracy is equal within noise for the 27B (±2 items) and slightly favours the AMD build for the 35B
(+2 and +5 items, which also changes backend). v1.5-style Intelligence is 1–2 points higher on the
NVIDIA builds at most budgets. The AMD builds are 3–5× slower: the startup probe measured ~31 tok/s
(27B) and ~42 tok/s (35B) single-stream on the R9700, against ~70–75 tok/s for the 27B on an RTX 5090,
and with thinking, generation speed dominates latency.

## AWS Bedrock (one-hot answers)

All 231 public items, one request at a time, from a home connection to us-east-1 (latency includes the
network round trip, as for Jev's hosted API). Bedrock exposes no logprobs, so every answer is one-hot
and Calibration is not reported. Cost uses Bedrock on-demand list prices and measured tokens.

| Model | Constraint | Accuracy | Original | Easy | Hard | Yes/no | Choice | Score | Intelligence (open) | Latency p50 / p95 | $ per 1k |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Claude Haiku 4.5 | forced tool | **87.9%** | 71/72 | 48/48 | **84/111** | 65/74 | 123/139 | 15/18 | **74.5** | 0.64 / 0.79 s | $1.620 |
| Claude Haiku 4.5 | native JSON schema | 86.1% | 71/72 | 48/48 | 80/111 | 62/74 | 123/139 | 14/18 | 69.6 | 1.29 / 1.90 s | $1.005 |
| Qwen3 Next 80B A3B | native JSON schema | 83.1% | 69/72 | 48/48 | 75/111 | 61/74 | 117/139 | 14/18 | 65.7 | 0.37 / 0.82 s | $0.115 |
| DeepSeek V3.2 | native JSON schema | 82.3% | 68/72 | 48/48 | 74/111 | 58/74 | 118/139 | 14/18 | 64.3 | 0.49 / 1.99 s | $0.451 |
| Qwen3 32B | native JSON schema | 81.4% | 69/72 | 48/48 | 71/111 | 61/74 | 114/139 | 13/18 | 58.9 | 0.28 / 0.56 s | $0.117 |
| Qwen3 Next 80B A3B | tool choice any | 81.0% | 68/72 | 48/48 | 71/111 | 57/74 | 116/139 | 14/18 | 62.4 | 0.55 / 1.26 s | $0.153 |
| Ministral 3 14B | forced tool | 80.1% | 67/72 | 48/48 | 70/111 | 58/74 | 114/139 | 13/18 | 52.3 | 0.26 / 0.76 s | $0.168 |
| Qwen3 32B | tool choice any | 79.2% | 70/72 | 48/48 | 65/111 | 60/74 | 109/139 | 14/18 | 57.7 | 0.39 / 0.68 s | $0.152 |
| Ministral 3 14B | native JSON schema | 79.2% | 68/72 | 48/48 | 67/111 | 56/74 | 114/139 | 13/18 | 50.7 | 0.24 / 0.93 s | $0.151 |
| Nova Micro | forced tool (no native support) | 70.6% | 66/72 | 48/48 | 49/111 | 51/74 | 99/139 | 13/18 | 45.7 | 0.30 / 0.39 s | $0.043 |

- Native JSON-schema output helped the Qwen models (about +2 points, and faster); Claude Haiku 4.5 was
  better and twice as fast with a forced tool call.
- DeepSeek V3.2 accepts a forced tool choice but answers in plain text instead of calling the tool; the
  shim refuses those answers. It works with native JSON-schema output.
- Llama 4 Scout supports neither native structured output nor forced/any tool choice, so the shim
  refuses to run it.
- One-hot answers never fall in v1.5's 0.2–0.8 yes/no abstention band, which flatters Intelligence
  slightly relative to logit-based systems.
- Claude Haiku 4.5's price ($1 / $5 per M) is its published Bedrock list price; the others come from the
  AWS price list.

## Quantization / backend: Qwen3.8-27B NVFP4 (vLLM, RTX 5090) vs UD-Q6_K (llama.cpp, R9700)

No thinking. At the time, the llama.cpp path read a raw top-200 instead of grammar-masked
probabilities (masking via `post_sampling_probs` came later). Accuracy is within a few items; the 5090
build is 5–9× faster.

| Configuration | UD-Q6_K, llama.cpp | NVFP4, vLLM | Same answer | Latency p50 / p95 |
|---|---|---|---:|---|
| JSON logprobs | 193/231 | 194/231 | 216/231 | 1.18 / 4.95 s → 0.17 / 0.51 s |
| JSON grammar | 201/231 | 201/231 | 222/231 | 3.54 / 7.65 s → 0.67 / 1.06 s |
| label logprobs | 191/231 | 186/231 | 211/231 | 1.25 / 4.89 s → 0.18 / 0.51 s |
| label grammar | 204/231 | 197/231 | — | 2.97 / 6.96 s → 0.58 / 0.95 s |

## Findings

- **Logprobs beats model-written probabilities** without thinking: higher accuracy, better
  calibration (Brier 0.134 vs 0.223 on Flash-Next), ~1 output token instead of ~40. Model-written
  numbers cluster at round values.
- **Keep a JSON answer frame in logprobs mode.** Bare-label answers lost 4–6 points of accuracy on
  every model; the frame commits the model to answering.
- **Probe, then hold.** A startup probe of every setting caught a vLLM quirk (masked-out tokens are
  reported at a −9999 sentinel rather than omitted) that would otherwise have silently pushed the shim
  into slow grammar mode, and caught DeepSeek V3.2 on Bedrock ignoring a forced tool choice.
- **Send the grammar to the server at every scoring step.** Reading raw top-k logprobs and
  renormalizing afterwards is mathematically equivalent, but on hard items the top 20 fill up with
  "Let", "To", … and the options fall out, forcing slow fallbacks. Server-side masking removed them.
- **Throw out partial matches.** Under a grammar mask a model spreads some mass over partial
  spellings of a label (e.g. `cr`, `c` for `credit`); keeping only each option's longest matching
  token drops ~0.1–2 % of mass and keeps the scoring well-defined.
- **Thinking budgets must be enforced by token cap, not grammar counts.** Expressing the budget as
  `( word ws ){0,N}` in GBNF slowed xgrammar from ~180 to ~9 tokens/s; a free-text thought rule plus
  an exact token cap costs nothing.
- **JevBench v1.5 counts yes/no answers with 0.2 < P(yes) < 0.8 as wrong.** Honest logprob
  probabilities land in that band more often than model-written round numbers, which is why grammar
  mode can out-score logprobs mode on Intelligence while being less accurate.

## Pricing

JevBench v1.5 prices a self-hosted system at its base model's public reference price
(`docs/METHOD-v1.5-ADDENDUM-PRICING.md` in the JevBench repo); on the v1.5.5 board that is the
OpenRouter headline price, e.g. Qwen3.8-27B at $0.425 / $2.55 per million tokens. At those prices
none of these models is within the Jev-class cost cap (2× Jev). The tables above use the cheapest
public provider prices found instead:

| Base model | Cheapest public price found (in / out per M) | Source |
|---|---|---|
| Qwen3.8-Flash(-Next) | $0.113 / $0.382 | DeepInfra catalog |
| Qwen3.8-27B | $0.024 / $4.35 (input-heavy runs), $0.05 / $2.20 (output-heavy) | OpenRouter providers |
| Qwen3.6-35B-A3B | $0.05 / $0.70 | OpenRouter provider |

## Not measured

- Flash-Next with the final longest-match scoring (JSON logprobs) and with thinking budgets.
- OpenAI-compatible hosted providers (the API-logprobs tier was tested against vLLM's OpenAI-compatible
  API only).
- The judge tier and sealed items (not public).
- Concurrency/throughput: JevBench measures one request at a time.
