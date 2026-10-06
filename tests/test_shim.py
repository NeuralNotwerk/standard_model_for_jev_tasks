"""Pure-Python tests: no GPU, no backend, no network."""

import json
import math

import pytest

from jev_shim import server as s


# ─── question shapes ─────────────────────────────────────────────────────

@pytest.mark.parametrize("q,labels", [
    ({"type": "noul", "instructions": "x", "criteria": {"true": "y", "false": "n"}}, ["yes", "no"]),
    ({"type": "noul", "instructions": "x"}, ["yes", "no"]),
    ({"type": "score", "instructions": "x", "criteria": ["lo", "mid", "hi"]}, ["0", "1", "2"]),
    ({"type": "score", "instructions": "x", "criteria": {"0": "lo", "1": "hi"}}, ["0", "1"]),
    ({"type": "choice", "instructions": "x", "criteria": {"b": "B", "a": "A"}}, ["b", "a"]),
    ({"type": "choice", "instructions": "x", "criteria": ["alpha", "beta"]}, ["alpha", "beta"]),
])
def test_labels_and_options(q, labels):
    got, options = s.labels_and_options(q)
    assert got == labels
    assert [lab for lab, _ in options] == labels


@pytest.mark.parametrize("q", [
    {"type": "choice", "instructions": "x"},
    {"type": "score", "instructions": "x", "criteria": {"low": "a", "high": "b"}},
    {"type": "rank", "instructions": "x"},
])
def test_unanswerable_questions_are_rejected(q):
    with pytest.raises(s.ShimError) as e:
        s.labels_and_options(q)
    assert e.value.status == 400


# ─── grammars and parsing ────────────────────────────────────────────────

def test_gbnf_literal_escapes():
    assert s.gbnf_lit('a"b\\c\n') == '"a\\"b\\\\c\\n"'


def test_pick_grammar_lists_every_label_once():
    g = s.build_pick_grammar("choice", ["license", "license_required", 'q"x'])
    assert g.startswith("root ::= ")
    assert g.count("\\\"license\\\"") == 1 and "license_required" in g and 'q\\\\\\"x' in g


@pytest.mark.parametrize("qtype,labels,text,expected", [
    ("noul", ["yes", "no"], "yes=0.83", {"type": "noul", "noul": 0.83}),
    ("choice", ["billing", "billing_disputes"], "billing_disputes\nbilling=0.1\nbilling_disputes=0.9",
     {"type": "choice", "choice": "billing_disputes", "probabilities": {"billing": 0.1, "billing_disputes": 0.9}}),
    ("score", ["0", "1", "10"], "10\n0=0\n1=0.2\n10=0.8",
     {"type": "score", "score": "10", "probabilities": {"0": 0.0, "1": 0.2, "10": 0.8}}),
])
def test_parse_lines(qtype, labels, text, expected):
    assert s.parse_lines(qtype, labels, text) == expected


def test_full_reply_grammar_wraps_answer():
    g = s.full_reply_grammar(s.build_pick_grammar("choice", ["a", "b"]), 500)
    assert g.startswith('root ::= "<think>\\n" thought "\\n</think>\\n\\n" answer')
    assert "answer ::= " in g and "root ::= \"{" not in g
    assert s.THINK_CUT_TEXT in g


def test_prompt_shows_grammar_and_ends_in_assistant_turn():
    q = {"type": "choice", "instructions": "Which queue?", "criteria": {"billing": "pay", "tech": "bugs"}}
    labels, options = s.labels_and_options(q)
    g = s.build_pick_grammar("choice", labels)
    p = s.build_prompt("ticket text", q, options, g, s._HOW_PICK["choice"])
    assert g in p and p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    p2 = s.build_prompt("ticket text", q, options, g, s._HOW_PICK["choice"], think_budget=500)
    assert "Your whole reply, thinking and answer" in p2 and "answer ::= " in p2


# ─── fake backend for scoring / thinking logic ───────────────────────────

class FakeBackend(s.Backend):
    """Character-level 'tokenizer' plus a scripted next-token distribution.

    Token ids are code points; multi-character tokens are registered in `vocab`.
    `dist(ids)` returns raw logprobs over candidate token ids after `ids`.
    """

    END = 1_000_000

    def __init__(self, vocab, dist):
        super().__init__("http://fake", "fake", 1, 16)
        self.vocab = vocab                       # id -> text
        self.dist = dist
        self.text_to_id = {t: i for i, t in vocab.items()}

    def tokenize(self, text):
        if text == s.END_OF_TURN:
            return [self.END]
        if text == "<|endoftext|>":
            return [self.END + 1]
        return [ord(c) for c in text]

    def detokenize(self, ids):
        return "".join(self.vocab.get(i, "" if i >= self.END else chr(i)) for i in ids)

    def node_top(self, ids, grammar):
        # emulate a grammar mask: keep tokens that are a prefix of some allowed tail
        alts = [json.loads(lit) for lit in _literals(grammar)]
        optional = ")?" in grammar
        raw = self.dist(ids)
        kept = {}
        for tid, lp in raw.items():
            txt = self.token_text(tid)
            if tid >= self.END:
                if optional:
                    kept[tid] = lp
            elif any(a.startswith(txt) for a in alts) and txt:
                kept[tid] = lp
        z = math.log(sum(math.exp(v) for v in kept.values()))
        return {k: v - z for k, v in kept.items()}, True

    def think(self, prompt, budget, grammar):
        return "step one. step two", budget, True


def _literals(grammar):
    import re
    return ['"' + m + '"' for m in re.findall(r'"((?:[^"\\]|\\.)*)"', grammar.split("::=", 1)[1])]


def test_longest_match_drops_partial_tokens():
    # "credit" (id 500) and its partial spelling "cr" (id 501) both lead to option 0;
    # "no" (id 502) leads to option 1. Only the longest match per option is kept.
    vocab = {500: "credit", 501: "cr", 502: "no"}
    raw = {500: math.log(0.6), 501: math.log(0.3), 502: math.log(0.1)}
    b = FakeBackend(vocab, lambda ids: raw)
    logp, usage = b.score_options("P", ["credit", "no"])
    p = [math.exp(x) for x in logp]
    assert p[0] == pytest.approx(0.6 / 0.7) and p[1] == pytest.approx(0.1 / 0.7)
    assert usage["_stats"]["masked_nodes"] == 1 and usage["_stats"]["fallbacks"] == 0


def test_prefix_labels_branch_on_end_of_turn():
    # "lic" vs "lic_x": the shared "lic" is context, so the only decision is
    # end-of-turn (option 0) vs "_x" (option 1).
    vocab = {601: "_x"}
    b = FakeBackend(vocab, lambda ids: {FakeBackend.END: math.log(0.25), 601: math.log(0.75)})
    logp, _ = b.score_options("P", ["lic", "lic_x"])
    assert [round(math.exp(x), 6) for x in logp] == [0.25, 0.75]


def test_thinking_is_spliced_into_the_answer_prompt():
    b = FakeBackend({}, lambda ids: {})
    prompt = "head<|im_start|>assistant\n<think>\n" + s.NO_THINK_SUFFIX
    new, info = s.with_thinking(b, prompt, 500, s.build_pick_grammar("choice", ["a", "b"]))
    assert info["hit_budget"] and info["text"].endswith(s.THINK_CUT_TEXT)
    assert new == "head<|im_start|>assistant\n<think>\nstep one. step two" + s.THINK_CUT_TEXT + s.NO_THINK_SUFFIX
