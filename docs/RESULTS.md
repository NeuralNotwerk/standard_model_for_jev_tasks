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
| Qwen3.6-35B-A3B | RedHatAI NVFP4 | vLLM | 1× RTX 5090 |

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

## Quantization / backend: Qwen3.8-27B NVFP4 (vLLM, RTX 5090) vs UD-Q6_K (llama.cpp, R9700)

Same scoring method on both. Accuracy is within a few items; the 5090 build is 5–9× faster.

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
- The judge tier and sealed items (not public).
- Concurrency/throughput: JevBench measures one request at a time.
