#!/usr/bin/env bash
# Run JevBench's public tiers (original + easy + hard, 231 decisions) against a running shim.
# Usage: scripts/run_jevbench.sh <out_dir> [extra `jevbench run` args, e.g. --limit 20]
#
# Environment:
#   JEVBENCH_DIR  JevBench checkout (default: ../jevbench next to this repo)
#   SHIM_URL      shim endpoint (default http://127.0.0.1:8250)
#   MODEL         label recorded in the run (default: jev-shim)
#   PRICE_IN / PRICE_OUT  USD per million input/output tokens for the Cost axis (default 0)
#   DELAY_S       pause between requests, seconds (default 0.5)
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:?usage: run_jevbench.sh <out_dir> [extra args]}; shift
JEVBENCH_DIR=${JEVBENCH_DIR:-$HERE/../jevbench}
D=$JEVBENCH_DIR/datasets/public
TASKS=$D/original.jsonl,$D/easy.jsonl,$D/hard.jsonl
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
cd "$JEVBENCH_DIR"
python3 -m jevbench.cli run --tasks "$TASKS" --adapter typesafe \
  --endpoint "${SHIM_URL:-http://127.0.0.1:8250}" --model "${MODEL:-jev-shim}" --key-env "" \
  --price-in-per-m "${PRICE_IN:-0}" --price-out-per-m "${PRICE_OUT:-0}" \
  --cost-basis "${COST_BASIS:-self_hosted}" --run-label "${RUN_LABEL:-jev-shim}" --delay-s "${DELAY_S:-0.5}" \
  --results "$OUT/results.jsonl" --ledger "$OUT/ledger.jsonl" --raw-dir "$OUT/raw" \
  --manifest "$OUT/manifest.json" "$@"
python3 -m jevbench.cli summarize --tasks "$TASKS" --results "$OUT/results.jsonl" \
  --ledger "$OUT/ledger.jsonl" > "$OUT/summary.json"
python3 -c "import json,sys;s=json.load(open(sys.argv[1]));print({k:s[k] for k in ('accuracy','macro_accuracy','brier_mean','coverage') if k in s})" "$OUT/summary.json"
echo "-> $OUT"
