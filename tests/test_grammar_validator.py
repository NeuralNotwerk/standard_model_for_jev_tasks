"""Validate every grammar the shim builds with llama.cpp's test-gbnf-validator.

Skipped unless GBNF_VALIDATOR points at the validator binary
(build llama.cpp with -DLLAMA_BUILD_TESTS=ON, target test-gbnf-validator).
If JEVBENCH_DIR points at a JevBench checkout, every label set in its public
datasets is checked too; otherwise only the synthetic cases run.

The validator exits 0 whether or not the input matched, so the verdict is read
from its output, never from the exit code.
"""

import glob
import json
import os
import subprocess

import pytest

from jev_shim import server as s

VALIDATOR = os.environ.get("GBNF_VALIDATOR", "")
pytestmark = pytest.mark.skipif(not (VALIDATOR and os.path.exists(VALIDATOR)),
                                reason="set GBNF_VALIDATOR to llama.cpp's test-gbnf-validator")

SYNTHETIC = [
    {"type": "choice", "instructions": "x", "criteria": {'quo"te': 1, "back\\slash": 2, "uni-é✓": 3, "new\nline": 4}},
    {"type": "choice", "instructions": "x", "criteria": {"license": "a", "license_required": "b"}},
    {"type": "score", "instructions": "x", "criteria": [f"L{i}" for i in range(11)]},
    {"type": "noul", "instructions": "x"},
]


def _questions():
    qs = list(SYNTHETIC)
    root = os.environ.get("JEVBENCH_DIR", "")
    for path in sorted(glob.glob(os.path.join(root, "datasets", "public", "*.jsonl"))) if root else []:
        with open(path, encoding="utf-8") as fh:
            qs += [json.loads(line)["question"] for line in fh if line.strip()]
    seen, out = set(), []
    for q in qs:
        labels, _ = s.labels_and_options(q)
        key = (q["type"], tuple(labels))
        if key not in seen:
            seen.add(key)
            out.append((q["type"], labels))
    return out


def classify(grammar, text, tmp_path):
    g, i = tmp_path / "g.gbnf", tmp_path / "in.txt"
    g.write_text(grammar)
    i.write_text(text)
    p = subprocess.run([VALIDATOR, str(g), str(i)], capture_output=True, text=True)
    if "error parsing grammar" in p.stderr.lower() or "Failed to initialize llama_grammar" in p.stdout:
        return "BROKEN"
    if "is valid according to the grammar" in p.stdout:
        return "VALID"
    if "is invalid according to the grammar" in p.stdout:
        return "INVALID"
    return f"UNKNOWN {p.returncode} {p.stdout!r} {p.stderr!r}"


def compact(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def cases(qtype, labels):
    """(grammar, text, should_match) for all four grammar builders."""
    key = s.pick_key(qtype)
    out = []
    # logprobs mode, JSON frame
    g = s.build_pick_grammar(qtype, labels)
    out += [(g, s.pick_answer(qtype, lab), True) for lab in labels]
    out += [(g, s.pick_answer(qtype, "__nope__"), False), (g, s.pick_answer(qtype, labels[0]) + " ", False)]
    # logprobs mode, bare labels
    g = s.build_label_grammar(labels)
    out += [(g, lab, True) for lab in labels] + [(g, "__nope__", False), (g, " " + labels[0], False)]
    # grammar mode, JSON
    g = s.build_grammar(qtype, labels)
    if qtype == "noul":
        out += [(g, '{"type":"noul","noul":0.83}', True), (g, '{"type":"noul","noul":1.5}', False)]
    else:
        probs = {lab: round(1 / len(labels), 3) for lab in labels}
        out += [(g, compact({"type": qtype, "choice" if qtype == "choice" else "score": labels[-1],
                             "probabilities": probs}), True),
                (g, compact({"type": qtype, "choice" if qtype == "choice" else "score": "__nope__",
                             "probabilities": probs}), False)]
    # grammar mode, label lines (and the parser must read back what the grammar accepts)
    g = s.build_lines_grammar(qtype, labels)
    if qtype == "noul":
        out += [(g, "yes=0.83", True), (g, "yes=1.5", False)]
    else:
        body = "".join(f"\n{lab}={round(1 / len(labels), 3)}" for lab in labels)
        text = labels[-1] + body
        assert s.parse_lines(qtype, labels, text)[key] == labels[-1]
        out += [(g, text, True), (g, "__nope__" + body, False)]
    # thinking: whole-reply grammar shown to the model, and the server-side version
    ans_g = s.build_pick_grammar(qtype, labels)
    a = s.pick_answer(qtype, labels[0])
    shown = "\n".join(line for line in s.full_reply_grammar(ans_g, 500).split("\n") if not line.startswith("#")) + "\n"
    out += [(shown, "<think>\nx < 5, so pick it.\n</think>\n\n" + a, True),
            (shown, "<think>\nsee </b>\n</think>\n\n" + a, False),
            (s.thinking_grammar(ans_g, 500), "long thought" + s.THINK_CUT_TEXT + "\n</think>\n\n" + a, True)]
    return out


@pytest.mark.parametrize("qtype,labels", _questions(), ids=lambda v: str(v)[:40])
def test_every_grammar(qtype, labels, tmp_path):
    for grammar, text, want in cases(qtype, labels):
        got = classify(grammar, text, tmp_path)
        assert got == ("VALID" if want else "INVALID"), f"{got} for {text!r}\n{grammar}"
