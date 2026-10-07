#!/usr/bin/env bash
# Start a shim per mode and run JevBench through it, one mode after another (never
# sharing the GPU). Every invocation writes a NEW timestamped folder; nothing is deleted.
#
# Usage: scripts/run_matrix.sh [runs_dir]
# Environment (plus everything run_jevbench.sh reads):
#   BACKEND_URL   inference server or API (default http://127.0.0.1:8000); bedrock://<region> for AWS
#   BACKEND_TYPE  auto | vllm | llamacpp | openai | bedrock (default auto)
#   MODEL         model name / Bedrock model id (default: auto-detected where the backend lists models)
#   TAG           folder/run label (default: shim)
#   MODES         "logprobs grammar" (default) or a subset; "pick" for JSON decisions (remote APIs)
#   CONSTRAINT    auto | gbnf | json_schema | json_object | forced_tool | any_tool (default auto)
#   FORMAT        json | label (default json)
#   THINK         thinking budget in tokens, 0 = none (default 0)
#   PORT_BASE     first shim port (default 8250; grammar mode uses +1)
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
BACKEND_URL=${BACKEND_URL:-http://127.0.0.1:8000}
BACKEND_TYPE=${BACKEND_TYPE:-auto}
CONSTRAINT=${CONSTRAINT:-auto}
MODEL=${MODEL:-}
TAG=${TAG:-shim}
MODES=${MODES:-logprobs grammar}
FORMAT=${FORMAT:-json}
THINK=${THINK:-0}
PORT_BASE=${PORT_BASE:-8250}
R=${1:-${RUNS_DIR:-$HERE/runs}}/$(date +%Y%m%d-%H%M%S)-$TAG-$FORMAT$([ "$THINK" -gt 0 ] && echo "-think$THINK")
export PYTHONPATH=$HERE/src${PYTHONPATH:+:$PYTHONPATH}

run_mode() {  # mode port
  local mode=$1 port=$2 out=$R/$1
  if [ -e "$out" ]; then echo "refusing to overwrite $out"; exit 1; fi
  python3 -m jev_shim --port "$port" --mode "$mode" --format "$FORMAT" --think-budget "$THINK" \
    --constraint "$CONSTRAINT" --quiet \
    --backend "$BACKEND_URL" --backend-type "$BACKEND_TYPE" ${MODEL:+--model "$MODEL"} & local pid=$!
  trap "kill $pid 2>/dev/null" EXIT
  until curl -sf "http://127.0.0.1:$port/health" >/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "shim failed to start"; exit 1; }; sleep 0.5
  done
  echo "=== $mode mode, $FORMAT format, think $THINK -> $out"
  SHIM_URL=http://127.0.0.1:$port MODEL=${MODEL:-$TAG} RUN_LABEL=$TAG-$mode-$FORMAT-think$THINK \
    "$HERE/scripts/run_jevbench.sh" "$out"
  kill $pid; wait $pid 2>/dev/null; trap - EXIT
}
for m in $MODES; do
  case $m in
    logprobs) run_mode logprobs "$PORT_BASE" ;;
    grammar)  run_mode grammar "$((PORT_BASE + 1))" ;;
    pick)     run_mode pick "$((PORT_BASE + 2))" ;;
    *) echo "unknown mode $m"; exit 1 ;;
  esac
done
echo "=== ALL DONE -> $R"
