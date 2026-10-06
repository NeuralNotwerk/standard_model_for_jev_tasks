#!/usr/bin/env bash
# Live progress of every run under a runs directory: scripts/watch_runs.sh [runs_dir]
R=${1:-$(cd "$(dirname "$0")/.." && pwd)/runs}
watch -n 5 "python3 - '$R' <<'PY'
import json, glob, os, sys
for f in sorted(glob.glob(os.path.join(sys.argv[1], '**', 'results.jsonl'), recursive=True)):
    R = [json.loads(l) for l in open(f) if l.strip()]
    if not R: continue
    lat = sorted(r['latency_s'] for r in R)
    ok = sum(bool(r['correct']) for r in R)
    print(f\"{os.path.relpath(os.path.dirname(f), sys.argv[1]):<48} {len(R):>4} done  acc {ok/len(R):.3f}  \"
          f\"p50 {lat[len(lat)//2]:.2f}s  p95 {lat[int(len(lat)*.95)]:.2f}s  failed {sum(r['status']=='failed' for r in R)}\")
PY"
