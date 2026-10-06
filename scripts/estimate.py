#!/usr/bin/env python3
"""Public-items estimate of JevBench v1.5 axes and score for one or more runs.

    JEVBENCH_DIR=../jevbench scripts/estimate.py runs/<run>/logprobs/results.jsonl \
        [--price-in 0.05 --price-out 2.2] [--hosted]

Follows docs/METHOD-v1.5.md of the JevBench repo where the formula is stated:
  Intelligence  per type (choice / noul / score) and tier, chance-corrected;
                tiers easy .10 / standard .20 / judge .30 / hard .40 (judge is not
                public, so its weight is redistributed); types .50 / .25 / .25.
                Noul: P(yes) <= .2 is No, >= .8 is Yes, in between counts wrong.
  Calibration   typed; mapped to 0-100 with the v1.3 rule 100*(1 - ECE/0.5).
                Choice TVD is skipped (no public gold distributions).
  Capability    (Intelligence + Calibration) / 2
  Speed         mean of score(p50), score(p95) on the standard tier,
                score(s) = 100 - 20 log10(s / 0.1 s); self-hosted latency is
                adjusted x2 + 0.15 s unless --hosted
  Cost          100 - 30 log10($ per 1,000 decisions / $0.001), from measured tokens
  JevBench      weighted harmonic mean, headline weights I/C/S/$ = 40/20/20/20;
                Intelligence, Speed and Cost below 50 multiply it by (axis/50)^2.

This is NOT an official score: the official one blends in sealed items.
"""

import argparse
import glob
import json
import math
import os
import sys

TIER = {"original": "standard", "easy": "easy", "hard": "hard"}
TW = {"easy": .10, "standard": .20, "judge": .30, "hard": .40}
TYW = {"choice": .50, "noul": .25, "score": .25}


def load_tasks(jevbench_dir):
    tasks = {}
    for path in glob.glob(os.path.join(jevbench_dir, "datasets", "public", "*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    tasks[d["id"]] = d
    if not tasks:
        sys.exit(f"no public datasets under {jevbench_dir}; set JEVBENCH_DIR")
    return tasks


def ece(pairs, bins=10):
    tot, e = len(pairs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, y) for c, y in pairs if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if sel:
            e += len(sel) / tot * abs(sum(c for c, _ in sel) / len(sel) - sum(y for _, y in sel) / len(sel))
    return e


def cal_score(e):
    return max(0.0, 100 * (1 - e / 0.5))


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def clamp(x):
    return max(0.0, min(100.0, x))


def estimate(path, tasks, price_in=0.0, price_out=0.0, hosted=False):
    R = [json.loads(line) for line in open(path) if line.strip()]
    cc, calib = {}, {"choice": [], "noul": [], "score_rps": [], "score_top": []}
    for r in R:
        t = tasks[r["task_id"]]
        qt, tier, p = t["question"]["type"], TIER[r["task_id"].split("-")[0]], r.get("probs") or {}
        if qt == "noul":
            py = p.get("yes", 0.0) if r["valid"] else None
            label = None if py is None else ("yes" if py >= .8 else "no" if py <= .2 else None)
            cc.setdefault(("noul", tier), []).append(1.0 if label == t["expected"] else 0.0)
            if py is not None:
                calib["noul"].append((py, 1.0 if t["expected"] == "yes" else 0.0))
        elif qt == "choice":
            ok = bool(r["correct"])
            cc.setdefault(("choice", tier), []).append((1.0 if ok else 0.0, 1 / len(t["labels"])))
            if p:
                calib["choice"].append((max(p.values()), 1.0 if ok else 0.0))
        else:
            K, gold = len(t["labels"]), int(t["expected"])
            pred = sum(int(k) * v for k, v in p.items()) if p else 0.0
            chance = sum(abs(lv - gold) for lv in range(K)) / K / (K - 1)
            cc.setdefault(("score", tier), []).append((abs(pred - gold) / (K - 1), chance))
            if p:
                cum, rps = 0.0, 0.0
                for lv in range(K - 1):
                    cum += p.get(str(lv), 0.0)
                    rps += (cum - (1.0 if gold <= lv else 0.0)) ** 2
                calib["score_rps"].append(rps / (K - 1))
                top = max(p, key=p.get)
                calib["score_top"].append((p[top], 1.0 if int(top) == gold else 0.0))
    per_type = {}
    for qt in TYW:
        num = den = 0.0
        for tier, w in TW.items():
            v = cc.get((qt, tier))
            if not v:
                continue
            if qt == "noul":
                val = 100 * (sum(v) / len(v) - .5) / .5
            elif qt == "choice":
                acc, ch = sum(a for a, _ in v) / len(v), sum(c for _, c in v) / len(v)
                val = 100 * (acc - ch) / (1 - ch)
            else:
                val = 100 * (1 - (sum(a for a, _ in v) / len(v)) / (sum(c for _, c in v) / len(v)))
            num, den = num + w * val, den + w
        if den:
            per_type[qt] = num / den
    I = sum(TYW[k] * v for k, v in per_type.items()) / sum(TYW[k] for k in per_type)
    C = (TYW["choice"] * cal_score(ece(calib["choice"])) + TYW["noul"] * cal_score(ece(calib["noul"]))
         + TYW["score"] * (100 * (1 - sum(calib["score_rps"]) / max(len(calib["score_rps"]), 1))
                           + cal_score(ece(calib["score_top"]))) / 2)
    std = [r["latency_s"] if hosted else r["latency_s"] * 2 + .15 for r in R if r["task_id"].startswith("original")]
    S = (clamp(100 - 20 * math.log10(pct(std, .5) / .1)) + clamp(100 - 20 * math.log10(pct(std, .95) / .1))) / 2
    tin = sum(r["usage"].get("input_tokens") or 0 for r in R) / len(R)
    tout = sum(r["usage"].get("output_tokens") or 0 for r in R) / len(R)
    usd = (tin * price_in + tout * price_out) / 1e3
    K = clamp(100 - 30 * math.log10(usd / 0.001)) if usd > 0 else 100.0
    axes = [(I, .4), (C, .2), (S, .2), (K, .2)]
    score = sum(w for _, w in axes) / sum(w / max(v, 1e-9) for v, w in axes)
    for v in (I, S, K):
        if v < 50:
            score *= (max(v, 0) / 50) ** 2
    acc = sum(bool(r["correct"]) for r in R)
    return dict(n=len(R), accuracy=acc / len(R), intelligence=I, calibration=C, capability=(I + C) / 2,
                speed=S, usd_per_1000=usd, cost=K, jevbench_score=score, per_type=per_type,
                p50_s=pct([r["latency_s"] for r in R], .5), p95_s=pct([r["latency_s"] for r in R], .95),
                input_tokens=tin, output_tokens=tout)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", nargs="+", help="results.jsonl files from scripts/run_jevbench.sh")
    ap.add_argument("--price-in", type=float, default=0.0, help="USD per million input tokens")
    ap.add_argument("--price-out", type=float, default=0.0, help="USD per million output tokens")
    ap.add_argument("--hosted", action="store_true", help="no self-host latency adjustment")
    ap.add_argument("--jevbench-dir", default=os.environ.get("JEVBENCH_DIR", "../jevbench"))
    a = ap.parse_args()
    tasks = load_tasks(a.jevbench_dir)
    for path in a.results:
        e = estimate(path, tasks, a.price_in, a.price_out, a.hosted)
        print(f"{path}\n  n={e['n']} acc={e['accuracy']:.3f}  I={e['intelligence']:.1f}  C={e['calibration']:.1f}  "
              f"Cap={e['capability']:.1f}  Speed={e['speed']:.1f}  Cost={e['cost']:.1f} (${e['usd_per_1000']:.4f}/1k)  "
              f"JevBench={e['jevbench_score']:.1f}  p50/p95={e['p50_s']:.2f}/{e['p95_s']:.2f}s")


if __name__ == "__main__":
    main()
