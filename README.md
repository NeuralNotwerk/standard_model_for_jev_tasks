# standard_model_for_jev_tasks

A shim that serves an ordinary LLM as a **Jev-class decision model**.

Jev-style models take a piece of state plus a bounded question and return a typed answer with a
probability for every option: yes/no (`noul`), `choice` or `score`. This shim exposes the same
interface as TypeSafe's API (`POST /v1/systemone`), so any Jev client, including
[JevBench](https://github.com/fstandhartinger/jevbench)'s stock `typesafe` adapter, can drive:

| Backend | Output constraint | Probabilities |
|---|---|---|
| **vLLM** | GBNF grammar | from the model's logits (grammar-masked) |
| **llama.cpp** | GBNF grammar | from the model's logits (grammar-masked, via `post_sampling_probs`) |
| **OpenAI-compatible APIs** (OpenAI, OpenRouter, DeepInfra, …) | strict JSON schema (or JSON mode) | from the API's top logprobs when offered, otherwise one-hot |
| **AWS Bedrock** | native JSON-schema output, or a forced tool call | one-hot (Bedrock exposes no logprobs) |

The core is stdlib only; Bedrock needs `boto3` (optional extra).

## How it works

For each question the shim builds a prompt containing the state, the question, the options and **the
exact grammar or JSON schema the reply must match**, and the backend enforces it.

**`logprobs` mode: probabilities from logits.** Nothing is generated. The shim walks generation from
the prompt. At each point where the options still differ, it sends one request with a grammar of the
text each option has left (`root ::= "credit_10_percent" | "credit_20_percent" | …`). The server
applies the grammar mask and returns the masked next-token distribution. For each option the shim
keeps only its **longest matching token** (partial spellings such as `cr` for `credit` are dropped),
renormalizes, and follows the kept tokens. An option's probability is the product of its steps: the
distribution grammar-constrained decoding would sample from. Most questions need a single request.

**`grammar` mode: model-written probabilities.** The model writes the answer, probabilities included,
under the grammar; the shim renormalizes the numbers to sum to 1.

**`pick` mode: a JSON decision.** For APIs without logits the model returns `{"choice": "…"}` (or
`answer` / `score`) under a JSON schema. With top logprobs available (some OpenAI-compatible
providers), probabilities come from them with the same longest-match rule. Without them, the answer is
**one-hot**: the chosen option gets 1 and every other option 0.

Further options for local backends:

- `--format json|label`: the model writes the Jev JSON object, or just the decision, and the shim
  builds the JSON. JSON is better in logprobs mode, because the frame commits the model to answering.
- `--think-budget N`: let the model think first, for at most N tokens, under a whole-reply grammar
  (`root ::= "<think>\n" thought "\n</think>\n\n" answer`) that is also shown in the prompt. If the
  budget runs out, the thought is cut, `...I just need to make a decision` is appended, and the answer
  step runs as usual.

Supported request shapes: `noul` (criteria `{true, false}` or none), `choice` (criteria map
option → description, or a list), `score` (criteria list of levels, or a `{"0": …, "1": …}` map); the
state can be a string or a JSON object; several questions per request are answered in parallel.

## Probe once, then hold the settings

At startup the shim **probes the server once for every setting it could use**, prints the results,
resolves the settings for the run and **holds them for the whole run**. Nothing is switched at
runtime.

| Backend | What the probe tests |
|---|---|
| vLLM, llama.cpp | server identity, ChatML prompts, grammar enforcement (asks for a poem while the grammar allows only yes/no), grammar-masked logprobs, exact logprob lookup, thinking, single-stream generation speed |
| OpenAI-compatible | strict JSON schema and JSON mode, each with and without top logprobs; on OpenRouter also the model's published `supported_parameters` |
| Bedrock | native JSON-schema output (`outputConfig`), forced tool call, tool choice `any` |

Probe prompts invite prose ("think it over and explain"), so a backend that accepts a setting but
ignores it fails the probe.

- Options left on `auto` get the strongest supported value.
- **Options given explicitly are requirements.** `--backend-type`, `--mode`, `--format`, `--constraint`
  and `--think-budget` must be supported, or the shim refuses to start and says what was asked for, what
  the server supports, and which checks failed. A model whose output cannot be constrained at all is
  refused.
- **Warnings.** When `auto` lands on a degraded setting, the shim warns about it: `SPEED` when grammar
  mode is chosen because masked logprobs are missing; `ZERO PROBABILITIES` when answers are one-hot;
  `SPEED/ACCURACY` when Bedrock falls back from native JSON schema to a tool call; JSON mode without a
  schema; no exact logprob lookup. A setting you chose explicitly never warns about itself, with one
  exception: a **thinking budget always warns** (`EXTREME PERFORMANCE IMPACT`), with an estimate from
  the measured generation speed.

The probe results and the settings are printed at startup, returned by `GET /health`, and stamped on
every response (`runtime`).

## Install

```bash
pip install -e .            # provides the `jev-shim` command
pip install -e '.[aws]'     # plus boto3 for AWS Bedrock
pip install -e '.[test]'    # plus pytest
```

## Run

```bash
# vLLM (recommended)
vllm serve <model> --port 8000 --enable-prefix-caching \
  --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}'
jev-shim --backend http://127.0.0.1:8000 --port 8250          # backend type and model auto-detected

# llama.cpp
llama-server -m model.gguf --port 8000 -fa on
jev-shim --backend http://127.0.0.1:8000 --port 8250

# OpenAI-compatible API
OPENAI_API_KEY=… jev-shim --backend https://api.openai.com/v1 --model gpt-4o-mini --port 8250
OPENROUTER_API_KEY=… jev-shim --backend https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY \
  --model qwen/qwen3.8-27b --port 8250

# AWS Bedrock (standard AWS credential chain; region from the URL or AWS_REGION)
jev-shim --backend bedrock://us-east-1 --model qwen.qwen3-next-80b-a3b --port 8250
```

The backend type is detected from the URL (`bedrock://…`, `*.amazonaws.com`) and from the server's
endpoints (`/v1/models` `owned_by: vllm` or `llamacpp`, `/props`, `/version`); any other server with
`/v1/models` is treated as OpenAI-compatible. `--backend-type` overrides detection but is checked
against the server; any OpenAI-style server can also be driven as `openai`.

Ask it something:

```bash
curl -s localhost:8250/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Ticket: customer was double charged $40. Policy: duplicate charges are refunded automatically.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Should the agent issue the refund?"},
    "route":  {"type": "choice", "instructions": "Which queue?",
               "criteria": {"billing": "payments issues", "tech": "bugs", "sales": "new purchases"}}
  }}'
```

Notes on serving:

- Don't run a reasoning parser on a local backend. The shim writes raw prompts with thinking handled
  explicitly, and some reasoning parsers stop the grammar from being applied without any error.
- On hybrid (Mamba/linear-attention) models with prefix caching, vLLM may need
  `--max-num-batched-tokens 8192` (its cache block can be larger than the default batch limit).
- Local prompts use Qwen's ChatML tokens (`<|im_start|>`, `<think>`); the probe checks for them.
  Other model families need `build_prompt` adapted. Remote APIs use plain chat messages.

## Limitations

- **One-hot probabilities on APIs without logprobs** (all of Bedrock, many OpenAI-compatible
  providers). Accuracy is measured normally; calibration is meaningless, and the composite
  scores that use it are not comparable with logit-based systems. Responses say
  `"probabilities_source": "one_hot_no_logprobs"`.
- **API top logprobs cover only the generated path.** Options that branch off it and then split
  further share their probability equally (flagged in `scoring.unresolved_splits`). This tier was
  tested against vLLM's OpenAI-compatible API, not yet against a hosted provider.
- **JSON mode** (`json_object`) does not hold the reply to the schema; it is only used when strict
  JSON schema is unavailable, and invalid replies fail.
- **Thinking** is only available on local backends, and costs seconds per decision (see
  [docs/RESULTS.md](docs/RESULTS.md)).
- Arbitrary GBNF grammars are not offered by the standard OpenAI-compatible or Bedrock APIs, so remote
  backends are limited to JSON-schema constraints.

## Benchmark with JevBench

```bash
git clone https://github.com/fstandhartinger/jevbench ../jevbench
export JEVBENCH_DIR=../jevbench

# one shim per mode, run one after another; every invocation writes a new runs/<timestamp>-… folder
BACKEND_URL=http://127.0.0.1:8000 TAG=my-model scripts/run_matrix.sh
FORMAT=json MODES=logprobs THINK=500 BACKEND_URL=… TAG=my-model scripts/run_matrix.sh
BACKEND_URL=bedrock://us-east-1 MODEL=qwen.qwen3-32b-v1:0 MODES=pick TAG=qwen3-32b scripts/run_matrix.sh

scripts/watch_runs.sh                     # live progress
scripts/estimate.py runs/*/logprobs/results.jsonl --price-in 0.05 --price-out 2.2
```

`scripts/estimate.py` estimates JevBench v1.5's axes (Intelligence, Calibration, Speed, Cost), the
Capability score and the composite from the **public** items. It is not an official score; see its
docstring for the exact rules.

## Results

Measured on JevBench's 231 public decisions; full tables, setup and caveats are in
[docs/RESULTS.md](docs/RESULTS.md). Without thinking, logprobs mode:

| System | Intelligence (open) | Calibration | Capability | Median latency (scored) |
|---|---:|---:|---:|---:|
| Jev 1.13.0 (public board) | 71.5 | 88.0 | 80.0 | 0.62 s |
| Qwen3.8-Flash-Next, 4× RTX 5090 | **72.4** | **88.5** | **80.5** | 0.68 s |
| Qwen3.8-27B NVFP4, 1× RTX 5090 | 62.0 | 83.0 | 72.5 | 0.49 s |
| Qwen3.6-35B-A3B NVFP4, 1× RTX 5090 | 55.5 | 77.8 | 66.7 | 0.43 s |

With a thinking budget, Qwen3.8-27B reaches Capability 91.3 (1,000 tokens) to 93.7 (1,500 tokens),
at 2–3 s median latency. AMD MXFP4 builds of the same models match the NVIDIA NVFP4 builds on
accuracy but run 3–5× slower on an R9700. On AWS Bedrock (one-hot answers), Claude Haiku 4.5 scored
87.9% and Qwen3 Next 80B A3B 83.1%.

## Tests

```bash
pytest                                                    # pure-Python tests, no backend needed
GBNF_VALIDATOR=/path/to/llama.cpp/build/bin/test-gbnf-validator \
JEVBENCH_DIR=../jevbench pytest                           # also validate every generated grammar
```

The validator tests check every grammar the shim can build (including the label sets of all public
JevBench items) against inputs it should and shouldn't accept. The resolver tests check that
unsupported explicit options are refused and that warnings appear only for unrequested degradations.

## Not affiliated

This project is not affiliated with or endorsed by TypeSafe AI (Jev) or JevBench/Benchmark Heaven.
"Jev" is used only to describe interface compatibility.

## License

Apache-2.0, see [LICENSE](LICENSE).
