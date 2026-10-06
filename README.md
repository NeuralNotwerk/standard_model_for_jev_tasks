# standard_model_for_jev_tasks

A shim that serves an ordinary instruct LLM as a **Jev-class decision model**.

Jev-style models take a piece of state plus a bounded question and return a typed answer with a
probability for every option: yes/no (`noul`), `choice` or `score`. This shim exposes the same
interface as TypeSafe's API (`POST /v1/systemone`), so any Jev client, including
[JevBench](https://github.com/fstandhartinger/jevbench)'s stock `typesafe` adapter, can drive a model
you serve yourself with **vLLM** or **llama.cpp**. Every answer is constrained by a GBNF grammar,
and probabilities come from the model's own logits.

Stdlib only, no dependencies.

## How it works

For each question the shim builds a ChatML prompt containing the state, the question, the options
and **the exact GBNF grammar the reply must match**, then uses one of two modes:

**`logprobs` (default): probabilities from logits.** Nothing is generated. The shim walks generation
from the prompt. At each point where the options still differ, it sends one request with a grammar
of the text each option has left (`root ::= "credit_10_percent" | "credit_20_percent" | …`). The
server applies the grammar mask and returns the masked next-token distribution. For each option the
shim keeps only its **longest matching token** (partial spellings such as `cr` for `credit` are
dropped), renormalizes, and follows the kept tokens. An option's probability is the product of its
steps: the distribution grammar-constrained decoding would sample from. Most questions need a single
request.

**`grammar`: model-written probabilities.** The model writes the answer, probabilities included,
under the grammar; the shim renormalizes the numbers to sum to 1.

Options:

- `--format json|label`: the model writes the Jev JSON object, or just the decision, and the shim
  builds the JSON. JSON is better in logprobs mode, because the frame commits the model to answering.
- `--think-budget N`: let the model think first, for at most N tokens, under a whole-reply grammar
  (`root ::= "<think>\n" thought "\n</think>\n\n" answer`) that is also shown in the prompt. If the
  budget runs out, the thought is cut, `...I just need to make a decision` is appended, and the answer
  step runs as usual.

Supported request shapes: `noul` (criteria `{true, false}` or none), `choice` (criteria map
option → description, or a list), `score` (criteria list of levels, or a `{"0": …, "1": …}` map); the
state can be a string or a JSON object; several questions per request are answered in parallel.

## Install

```bash
pip install -e .            # provides the `jev-shim` command
pip install -e '.[test]'    # plus pytest
```

## Run

Start a backend, then the shim:

```bash
# vLLM (recommended): grammar masking is applied before logprobs, so scoring is exact and fast
vllm serve <model> --port 8000 --enable-prefix-caching \
  --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}'
jev-shim --backend http://127.0.0.1:8000 --port 8250            # model name auto-detected

# llama.cpp: works, but reports pre-grammar probabilities, so the shim reads a raw top-200
llama-server -m model.gguf --port 8000 -fa on
jev-shim --backend http://127.0.0.1:8000 --backend-type llamacpp --port 8250
```

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

- Don't run a reasoning parser on the backend. The shim writes raw prompts with thinking handled
  explicitly, and some reasoning parsers stop the grammar from being applied without any error.
- On hybrid (Mamba/linear-attention) models with prefix caching, vLLM may need
  `--max-num-batched-tokens 8192` (its cache block can be larger than the default batch limit).
- Prompts use Qwen's ChatML tokens (`<|im_start|>`, `<think>`). Other model families need
  `build_prompt` adapted.

## Benchmark with JevBench

```bash
git clone https://github.com/fstandhartinger/jevbench ../jevbench
export JEVBENCH_DIR=../jevbench

# one shim per mode, run one after another; every invocation writes a new runs/<timestamp>-… folder
BACKEND_URL=http://127.0.0.1:8000 TAG=my-model scripts/run_matrix.sh
FORMAT=json MODES=logprobs THINK=500 BACKEND_URL=… TAG=my-model scripts/run_matrix.sh

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

With a 1,500-token thinking budget, Qwen3.8-27B reaches Capability 93.7 and Qwen3.6-35B-A3B 89.7,
at 2–3 s median latency.

## Tests

```bash
pytest                                                    # pure-Python tests, no backend needed
GBNF_VALIDATOR=/path/to/llama.cpp/build/bin/test-gbnf-validator \
JEVBENCH_DIR=../jevbench pytest                           # also validate every generated grammar
```

The validator tests check every grammar the shim can build (including the label sets of all public
JevBench items) against inputs it should and shouldn't accept.

## Not affiliated

This project is not affiliated with or endorsed by TypeSafe AI (Jev) or JevBench/Benchmark Heaven.
"Jev" is used only to describe interface compatibility.

## License

Apache-2.0, see [LICENSE](LICENSE).
