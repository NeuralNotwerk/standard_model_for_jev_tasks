#!/usr/bin/env python3
"""Jev-compatible decision-model shim for standard LLMs served by vLLM or llama.cpp.

Exposes TypeSafe's native decision interface (POST /v1/systemone), so JevBench's
stock `typesafe` adapter, or any Jev client, can drive an ordinary instruct model:

    POST /v1/systemone
      {"state": <str|obj>, "model": "...", "questions": {"<key>": {
          "type": "noul"|"choice"|"score",
          "instructions": "...",
          "criteria": {...} | [...]}}}
    ->
      {"model": "...", "answers": {"<key>": <answer>}, "usage": {...}}

Answer shapes returned to the caller:
    noul    {"type":"noul","noul":0.83}
    choice  {"type":"choice","choice":"b","probabilities":{"a":0.1,"b":0.9}}
    score   {"type":"score","score":"2","probabilities":{"0":0.0,...}}

Every prompt includes the exact GBNF grammar the reply must match, and the
backend enforces that grammar.

Modes (--mode):

  logprobs (default)  Probabilities from the model's own logits; nothing is
      generated. Generation is walked from the prompt: at each point where the
      options still differ, one request carries a grammar of the text each
      option has left, the server returns the grammar-masked next-token
      distribution, only each option's LONGEST matching token is kept (partial
      spellings such as "cr" for "credit" are dropped), the kept tokens are
      renormalized and followed. An option's probability is the product of its
      steps, i.e. the distribution grammar-constrained decoding would sample.

  grammar             The model writes the answer, probabilities included,
      under the grammar. The numbers are renormalized to sum to 1 unless
      --no-renormalize is given (raw values are returned as raw_probabilities).

Output format (--format): `json` (the model writes the Jev answer object) or
`label` (the model writes only the decision; the shim builds the JSON).

Thinking (--think-budget N): the model first thinks under a whole-reply grammar
(think block + answer) for at most N tokens. If it runs out, the thought is cut
and "...I just need to make a decision" is appended; the answer step then runs
exactly as without thinking.

Stdlib only.  Run:  jev-shim --backend http://127.0.0.1:8000 --port 8250
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# A probability with at most three decimals, nothing above 1.
PROB_RULE = 'prob ::= "0" ( "." [0-9] [0-9]? [0-9]? )? | "1" ( "." "0" "0"? "0"? )?'


class ShimError(Exception):
    """A question the shim refuses to answer; surfaced as HTTP 4xx/5xx."""

    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status


# ─── question → labels / option text ──────────────────────────────────

def labels_and_options(q: dict) -> tuple[list[str], list[tuple[str, str | None]]]:
    """Return (labels the answer is keyed by, [(label, description)])."""
    qtype, crit = q.get("type"), q.get("criteria")
    if qtype == "noul":
        crit = crit if isinstance(crit, dict) else {}
        return ["yes", "no"], [("yes", crit.get("true")), ("no", crit.get("false"))]
    if qtype == "score":
        if isinstance(crit, dict) and sorted(map(str, crit)) == sorted(str(i) for i in range(len(crit))):
            crit = [crit.get(str(i), crit.get(i)) for i in range(len(crit))]  # {"0": ..., "1": ...} form
        if not isinstance(crit, list) or len(crit) < 2:
            raise ShimError(400, "score question needs criteria as a list (or 0..K-1 map) of >= 2 levels")
        labels = [str(i) for i in range(len(crit))]
        return labels, [(lab, _text(c)) for lab, c in zip(labels, crit)]
    if qtype == "choice":
        if isinstance(crit, dict) and crit:
            labels = [str(k) for k in crit]
            return labels, [(lab, _text(crit[lab])) for lab in labels]
        if isinstance(crit, list) and crit:
            labels = [str(k) for k in crit]
            return labels, [(lab, None) for lab in labels]
        raise ShimError(400, "choice question needs criteria (dict of option -> description, or list)")
    raise ShimError(400, f"unsupported question type: {qtype!r}")


def _text(v) -> str | None:
    if v is None:
        return None
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


# ─── grammar ──────────────────────────────────────────────────────────

def gbnf_lit(s: str) -> str:
    """A GBNF string literal that matches exactly the characters of `s`."""
    out = s.replace("\\", "\\\\").replace('"', '\\"')
    out = out.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return f'"{out}"'


def build_grammar(qtype: str, labels: list[str]) -> str:
    """GBNF forcing the exact compact JSON answer object for this question."""
    if qtype == "noul":
        return "\n".join([
            "root ::= " + gbnf_lit('{"type":"noul","noul":') + " prob " + gbnf_lit("}"),
            PROB_RULE, ""])
    pick_key = "choice" if qtype == "choice" else "score"
    alts = " | ".join(gbnf_lit(json.dumps(lab, ensure_ascii=False)) for lab in labels)
    parts = []
    for i, lab in enumerate(labels):
        sep = "," if i else ""
        parts.append(gbnf_lit(sep + json.dumps(lab, ensure_ascii=False) + ":") + " prob")
    head = '{"type":' + json.dumps(qtype) + ',"' + pick_key + '":'
    return "\n".join([
        "root ::= " + gbnf_lit(head) + " pick " + gbnf_lit(',"probabilities":{')
        + " " + " ".join(parts) + " " + gbnf_lit("}}"),
        "pick ::= " + alts,
        PROB_RULE, ""])


def pick_key(qtype: str) -> str:
    return {"noul": "answer", "choice": "choice", "score": "score"}[qtype]


def pick_answer(qtype: str, label: str) -> str:
    """The complete answer string for one option in logprob mode."""
    return '{"type":' + json.dumps(qtype) + ',"' + pick_key(qtype) + '":' + json.dumps(label, ensure_ascii=False) + "}"


def build_pick_grammar(qtype: str, labels: list[str]) -> str:
    """GBNF whose language is exactly {pick_answer(qtype, l) for l in labels}."""
    head = '{"type":' + json.dumps(qtype) + ',"' + pick_key(qtype) + '":'
    alts = " | ".join(gbnf_lit(json.dumps(lab, ensure_ascii=False)) for lab in labels)
    return "\n".join(["root ::= " + gbnf_lit(head) + " pick " + gbnf_lit("}"), "pick ::= " + alts, ""])


# Label format: the model writes only the decision; the shim builds the Jev JSON
# (post-processing, as Jev itself does). An answer ends at the end-of-turn token.
END_OF_TURN = "<|im_end|>"


def build_label_grammar(labels: list[str]) -> str:
    """GBNF whose language is exactly the option labels."""
    return "root ::= " + " | ".join(gbnf_lit(lab) for lab in labels) + "\n"


def build_lines_grammar(qtype: str, labels: list[str]) -> str:
    """Grammar-mode label format: `yes=p` for noul; otherwise the best label on
    the first line, then `label=p` for every option in the given order."""
    if qtype == "noul":
        return "\n".join(["root ::= " + gbnf_lit("yes=") + " prob", PROB_RULE, ""])
    lines = " ".join(gbnf_lit("\n" + lab + "=") + " prob" for lab in labels)
    return "\n".join(["root ::= pick " + lines,
                      "pick ::= " + " | ".join(gbnf_lit(lab) for lab in labels),
                      PROB_RULE, ""])


def parse_lines(qtype: str, labels: list[str], text: str) -> dict:
    """Inverse of build_lines_grammar -> the Jev answer object."""
    if qtype == "noul":
        return {"type": "noul", "noul": float(text[len("yes="):])}
    rest = text
    pick = next(lab for lab in sorted(labels, key=len, reverse=True) if rest.startswith(lab + "\n"))
    rest = rest[len(pick):]
    probs = {}
    for lab in labels:
        head = "\n" + lab + "="
        if not rest.startswith(head):
            raise ShimError(502, f"grammar not enforced? unexpected output {text[:200]!r}")
        rest = rest[len(head):]
        end = rest.find("\n")
        num, rest = (rest, "") if end < 0 else (rest[:end], rest[end:])
        probs[lab] = float(num)
    return {"type": qtype, pick_key(qtype): pick, "probabilities": probs}


# Whole-reply grammar for thinking runs: the think block, then the answer.
# The thought is free text (anything but "</" until the closing tag). Its budget is
# enforced by the server's exact token cap, not by grammar repetition counts:
# xgrammar slows from ~180 to ~9 tok/s on word-structured or counted thoughts.
# When the cap is hit the shim appends THINK_CUT_TEXT and closes the block.
THINK_CUT_TEXT = "...I just need to make a decision"
THOUGHT_RULE = 'thought ::= ( [^<] | "<" [^/] )*'


def think_words(budget: int) -> int:
    return max(25, round(budget * 0.4 / 25) * 25)   # measured ~0.3 (hard, ID-heavy) to ~0.6 (easy) words/token


def _as_answer_rule(answer_grammar: str) -> str:
    assert answer_grammar.startswith("root ::= ")
    return "answer ::= " + answer_grammar[len("root ::= "):].rstrip("\n")


def full_reply_grammar(answer_grammar: str, budget: int) -> str:
    """The grammar shown to the model: its entire reply, thinking included."""
    return "\n".join(['root ::= "<think>\\n" thought "\\n</think>\\n\\n" answer',
                      THOUGHT_RULE,
                      f"# thought: at most {budget} tokens (about {think_words(budget)} words). At the limit the thought",
                      f"# is cut off, \"{THINK_CUT_TEXT}\" is appended, and the block is closed for you.",
                      _as_answer_rule(answer_grammar), ""])


def thinking_grammar(answer_grammar: str, budget: int) -> str:
    """Server-side grammar for the thinking step (generation starts after "<think>\\n")."""
    return "\n".join(['root ::= thought "\\n</think>\\n\\n" answer', THOUGHT_RULE,
                      _as_answer_rule(answer_grammar), ""])


# ─── prompt ───────────────────────────────────────────────────────────

SYSTEM = (
    "You are a decision model. You read a state and one bounded question, then "
    "output a typed JSON answer and nothing else. Your output is constrained by "
    "the GBNF grammar given with the question: it is exactly one string that "
    "grammar accepts, compact JSON with no whitespace."
)

SYSTEM_LABEL = (
    "You are a decision model. You read a state and one bounded question, then "
    "answer in the exact format given by the GBNF grammar with the question: "
    "your output is exactly one string that grammar accepts, and nothing else."
)

_KIND = {"noul": "yes/no", "score": "score on an ordinal scale", "choice": "choose one option"}
_HEADING = {"noul": "Meaning of the answers", "score": "Levels", "choice": "Options"}


def build_prompt(state, q: dict, options: list[tuple[str, str | None]], grammar: str, how: str,
                 system: str = SYSTEM, think_budget: int = 0) -> str:
    """ChatML prompt that shows the model the grammar its answer must match.

    Ends right after the assistant header with thinking suppressed, so the
    first generated token is the first character of the answer.
    """
    qtype = q["type"]
    state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
    opt_lines = "\n".join(f"- {lab}: {d}" if d else f"- {lab}" for lab, d in options)
    user = (f"State:\n{state_text}\n\n"
            f"Question type: {_KIND[qtype]}.\n{q['instructions']}\n\n"
            f"{_HEADING[qtype]}:\n{opt_lines}\n\n"
            f"{how}\n\n")
    if think_budget > 0:
        user += (f"Think first, inside the think block: you have at most {think_budget} tokens "
                 f"(about {think_words(think_budget)} words). At the limit your thought is cut off with "
                 f"\"{THINK_CUT_TEXT}\" and you must answer immediately.\n\n"
                 f"Your whole reply, thinking and answer, must be a string accepted by this GBNF grammar:\n"
                 f"```gbnf\n{full_reply_grammar(grammar, think_budget)}```")
    else:
        user += f"Your answer must be a string accepted by this GBNF grammar:\n```gbnf\n{grammar}```"
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


_HOW_GRAMMAR = {
    "noul": 'Set "noul" to your probability that the answer is yes.',
    "score": 'Set "score" to the single best level and "probabilities" to your probability for every level.',
    "choice": 'Set "choice" to the single best option and "probabilities" to your probability for every option.',
}
_HOW_PICK = {
    "noul": 'Set "answer" to "yes" or "no".',
    "score": 'Set "score" to the single best level.',
    "choice": 'Set "choice" to the single best option.',
}


_HOW_LABEL_PICK = {
    "noul": "Answer yes or no.",
    "score": "Answer with the single best level.",
    "choice": "Answer with the single best option.",
}
_HOW_LABEL_LINES = {
    "noul": 'Write "yes=" followed by your probability that the answer is yes.',
    "score": ("First line: the single best level. Then one line per level, in the order shown: "
              "level=your probability for it."),
    "choice": ("First line: the single best option. Then one line per option, in the order shown: "
               "option=your probability for it."),
}


# ─── backend calls + answer shaping ───────────────────────────────────────

class Backend:
    def __init__(self, base_url: str, model: str, timeout_s: float, max_tokens: int):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise ShimError(502, f"backend HTTP {e.code}: {e.read()[:300]!r}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise ShimError(502, f"backend unreachable: {e}") from e

    def complete(self, prompt: str, grammar: str) -> tuple[str, dict]:
        parsed = self._post("/v1/completions", {
            "model": self.model,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "structured_outputs": {"grammar": grammar, "disable_any_whitespace": True},
        })
        choice = (parsed.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "length":
            raise ShimError(502, "backend hit max_tokens before the grammar closed")
        return choice.get("text", ""), parsed.get("usage") or {}

    def tokenize(self, text: str) -> list[int]:
        out = self._post("/tokenize", {"model": self.model, "prompt": text, "add_special_tokens": False})
        return out["tokens"]

    def detokenize(self, ids: list[int]) -> str:
        return self._post("/detokenize", {"model": self.model, "tokens": ids})["prompt"]

    def token_text(self, tid: int) -> str:
        cache = self.__dict__.setdefault("_tok_text", {})
        if tid not in cache:
            cache[tid] = self.detokenize([tid])
        return cache[tid]

    def think(self, prompt: str, budget: int, grammar: str) -> tuple[str, int, bool]:
        """Thinking after `prompt` under the whole-reply `grammar`, stopping at </think>
        or `budget` tokens. Returns (text, tokens generated, hit_budget)."""
        out = self._post("/v1/completions", {
            "model": self.model, "prompt": prompt, "max_tokens": budget, "temperature": 0,
            "stop": [THINK_CLOSE], "structured_outputs": {"grammar": grammar}})
        c = (out.get("choices") or [{}])[0]
        return c.get("text", ""), (out.get("usage") or {}).get("completion_tokens", 0), c.get("finish_reason") == "length"

    def node_top(self, ids: list[int], grammar: str) -> tuple[dict[int, float], bool]:
        """Top-20 next-token logprobs after `ids` with `grammar` applied server-side.

        On the vLLM builds tested, the structured-output bitmask is applied before
        logprobs are computed, so every returned token is grammar-allowed and the
        values are renormalized over what the grammar permits. Returns (top, masked=True).
        """
        out = self._post("/v1/completions", {
            "model": self.model, "prompt": ids, "max_tokens": 1, "temperature": 0,
            "logprobs": 20, "return_tokens_as_token_ids": True,
            "structured_outputs": {"grammar": grammar}})
        top = ((out.get("choices") or [{}])[0].get("logprobs") or {}).get("top_logprobs") or [{}]
        return {int(k.split(":", 1)[1]): v for k, v in top[0].items()}, True

    def raw_logprobs_of(self, ids: list[int], toks: list[int]) -> list[float]:
        """Exact, unmasked logprob of each candidate token after `ids` (slow path)."""
        out = self._post("/v1/completions", {
            "model": self.model, "prompt": [ids + [t] for t in toks], "max_tokens": 1,
            "temperature": 0, "prompt_logprobs": 0})
        choices = sorted(out["choices"], key=lambda c: c.get("index", 0))
        return [c["prompt_logprobs"][-1][str(t)]["logprob"] for c, t in zip(choices, toks)]

    def score_options(self, prompt: str, answers: list[str]) -> tuple[list[float], dict]:
        """Log-probability of each answer under grammar-constrained decoding.

        Walks generation from the prompt as the server would see it. At every
        point where the options still differ, one request with a grammar of the
        text each option has left returns the grammar-masked next-token
        distribution. For each option only its LONGEST matching token is kept
        (shorter partial spellings such as "cr" for "credit" are thrown out),
        the kept tokens are renormalized, and each one is followed for the
        options it leads to. An option's probability is the product of its
        renormalized steps. Text all options share at the start (e.g. a JSON
        frame) is context, not a decision, and is appended to the prompt.
        If an option has no matching token in the top 20, that step is redone
        with exact unmasked logprobs for every kept token (scoring.fallbacks).
        """
        shared = os.path.commonprefix(answers)
        base = self.tokenize(prompt + shared)
        rest = [a[len(shared):] for a in answers]
        end_tok = self.tokenize(END_OF_TURN)[0]
        end_ids = {end_tok, *self.tokenize("<|endoftext|>")}  # either ends the turn
        stats = {"branch_nodes": 0, "fallbacks": 0, "masked_nodes": 0, "node_prompt_tokens": 0,
                 "valid_mass": None}
        logp = [0.0] * len(answers)

        def grammar_for(group, remaining):
            tails = [remaining[i] for i in group]
            body = " | ".join(gbnf_lit(t) for t in sorted({t for t in tails if t}))
            return "root ::= " + (f"( {body} )?" if "" in tails else body) + "\n"

        def pick(top, rem):
            """Longest returned token matching `rem` (end of turn when rem is empty)."""
            best = None
            for tid, lp in top.items():
                if rem == "":
                    ok, length = tid in end_ids, 0
                else:
                    txt = self.token_text(tid)
                    ok, length = (tid not in end_ids and txt != "" and rem.startswith(txt)), len(txt)
                if ok and (best is None or (length, lp) > (best[1], best[2])):
                    best = (tid, length, lp)
            return best

        def step(node):
            group, path, remaining = node
            return self.node_top(base + path, grammar_for(group, remaining))

        level = [(list(range(len(answers))), [], rest)] if len(answers) > 1 else []
        with ThreadPoolExecutor(max_workers=8) as pool:
            while level:
                results = list(pool.map(step, level))
                nxt = []
                for (group, path, remaining), (top, masked) in zip(level, results):
                    stats["branch_nodes"] += 1
                    stats["masked_nodes"] += masked
                    stats["node_prompt_tokens"] += len(base) + len(path)
                    chosen = {i: pick(top, remaining[i]) for i in group}
                    if any(c is None for c in chosen.values()):
                        # Some option has no matching token here: redo this step exactly,
                        # unmasked, for every option's token so they stay comparable.
                        stats["fallbacks"] += 1
                        for i, c in chosen.items():
                            if c is None:
                                rem = remaining[i]
                                tid = end_tok if rem == "" else self.tokenize(rem)[0]
                                chosen[i] = (tid, len(self.token_text(tid)) if rem else 0, None)
                        toks = sorted({c[0] for c in chosen.values()})
                        exact = dict(zip(toks, self.raw_logprobs_of(base + path, toks)))
                        chosen = {i: (c[0], c[1], exact[c[0]]) for i, c in chosen.items()}
                    by_tok = {}
                    for i, (tid, length, lp) in chosen.items():
                        by_tok.setdefault(tid, (length, lp, []))[2].append(i)
                    m = max(lp for _, lp, _ in by_tok.values())
                    z = sum(math.exp(lp - m) for _, lp, _ in by_tok.values())
                    if stats["valid_mass"] is None:  # share kept after dropping partial matches
                        stats["valid_mass"] = math.exp(m) * z
                    for tid, (length, lp, members) in by_tok.items():
                        for i in members:
                            logp[i] += lp - m - math.log(z)
                        if len(members) > 1:
                            nrem = list(remaining)
                            for i in members:
                                nrem[i] = remaining[i][length:]
                            nxt.append((members, path + [tid], nrem))
                level = nxt
        usage = {"prompt_tokens": len(base) + max(0, stats["node_prompt_tokens"] - stats["branch_nodes"] * len(base)),
                 "completion_tokens": stats["branch_nodes"],
                 "backend_prompt_tokens": stats["node_prompt_tokens"]}
        return logp, {**usage, "_stats": stats}


class LlamaCppBackend(Backend):
    """Same calls against llama.cpp's llama-server (/tokenize, /completion).

    `n_probs` returns raw (pre-sampling) top-k probabilities. llama.cpp has no
    prompt_logprobs, so the out-of-top-20 fallback re-asks the same branch point
    for the top 1000; a token still missing gets the 1000th logprob, which is an
    upper bound on its true value (counted in scoring.fallbacks either way).
    """

    def complete(self, prompt: str, grammar: str) -> tuple[str, dict]:
        out = self._post("/completion", {
            "prompt": prompt, "n_predict": self.max_tokens, "temperature": 0,
            "grammar": grammar, "cache_prompt": True})
        if out.get("stop_type") == "limit" or out.get("truncated"):
            raise ShimError(502, "backend hit n_predict before the grammar closed")
        return out.get("content", ""), {"prompt_tokens": out.get("tokens_evaluated", 0),
                                        "completion_tokens": out.get("tokens_predicted", 0)}

    def tokenize(self, text: str) -> list[int]:
        return self._post("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]

    def _top(self, ids: list[int], k: int) -> dict[int, float]:
        out = self._post("/completion", {
            "prompt": ids, "n_predict": 1, "temperature": 0, "n_probs": k,
            "post_sampling_probs": False, "cache_prompt": True})
        probs = (out.get("completion_probabilities") or [{}])[0]
        cache = self.__dict__.setdefault("_tok_text", {})
        for t in probs.get("top_logprobs") or []:
            cache.setdefault(t["id"], t["token"])
        return {t["id"]: t["logprob"] for t in probs.get("top_logprobs") or []}

    def detokenize(self, ids: list[int]) -> str:
        return self._post("/detokenize", {"tokens": ids})["content"]

    def think(self, prompt: str, budget: int, grammar: str) -> tuple[str, int, bool]:
        out = self._post("/completion", {
            "prompt": prompt, "n_predict": budget, "temperature": 0, "stop": [THINK_CLOSE],
            "grammar": grammar, "cache_prompt": True})
        return out.get("content", ""), out.get("tokens_predicted", 0), out.get("stop_type") == "limit"

    def node_top(self, ids: list[int], grammar: str) -> tuple[dict[int, float], bool]:
        # llama.cpp reports pre-grammar probabilities, so read a wide raw top-k
        # and let the longest-match selection play the grammar's role.
        return self._top(ids, 200), False

    def raw_logprobs_of(self, ids: list[int], toks: list[int]) -> list[float]:
        top = self._top(ids, 1000)
        if not top:
            raise ShimError(502, "backend returned no probabilities")
        floor = min(top.values())  # a token outside the top 1000 gets this upper bound
        return [top.get(t, floor) for t in toks]


THINK_CLOSE = "</think>"
NO_THINK_SUFFIX = "\n</think>\n\n"           # build_prompt ends with "<think>\n" + this


def with_thinking(backend: Backend, prompt: str, budget: int,
                  answer_grammar: str) -> tuple[str, dict | None]:
    """Replace the empty think block with up to `budget` tokens of real thinking.

    The model has been shown the whole-reply grammar (think block + answer) and
    thinks under it. If it closes </think> itself, its thought is used as is; if
    it hits the budget, the thought is cut there and THINK_CUT_TEXT is appended.
    Either way the block is closed and the answer is produced under the rest of
    the grammar exactly as without thinking.
    """
    if budget <= 0:
        return prompt, None
    assert prompt.endswith("<think>\n" + NO_THINK_SUFFIX)
    head = prompt[: -len(NO_THINK_SUFFIX)]          # ends with "<think>\n"
    text, n, token_capped = backend.think(head, budget, thinking_grammar(answer_grammar, budget))
    thought = text.rstrip()
    if token_capped:
        thought += THINK_CUT_TEXT
    return head + thought + NO_THINK_SUFFIX, {
        "tokens": n, "words": len(thought.split()), "hit_budget": thought.endswith(THINK_CUT_TEXT),
        "token_capped": token_capped, "text": thought}


def _bill_thinking(usage: dict, info: dict | None) -> dict:
    """Thinking tokens are output; don't also bill them as input to the answer step."""
    if info:
        usage["prompt_tokens"] = max(0, usage.get("prompt_tokens", 0) - info["tokens"])
        usage["completion_tokens"] = usage.get("completion_tokens", 0) + info["tokens"]
    return usage


def answer_logprobs(backend: Backend, state, q: dict, fmt: str = "json",
                    think: int = 0) -> tuple[dict, dict]:
    qtype = q["type"]
    labels, options = labels_and_options(q)
    if fmt == "label":
        grammar = build_label_grammar(labels)
        prompt = build_prompt(state, q, options, grammar, _HOW_LABEL_PICK[qtype], SYSTEM_LABEL, think)
        answers = list(labels)
    else:
        grammar = build_pick_grammar(qtype, labels)
        prompt = build_prompt(state, q, options, grammar, _HOW_PICK[qtype], think_budget=think)
        answers = [pick_answer(qtype, lab) for lab in labels]
    prompt, think_info = with_thinking(backend, prompt, think, grammar)
    logp, usage = backend.score_options(prompt, answers)
    stats = usage.pop("_stats")
    _bill_thinking(usage, think_info)
    probs = {lab: math.exp(x) for lab, x in zip(labels, logp)}
    if qtype == "noul":
        ans = {"type": "noul", "noul": probs["yes"]}
    else:
        ans = {"type": qtype, pick_key(qtype): max(labels, key=probs.__getitem__), "probabilities": probs}
    ans["label_logprobs"] = dict(zip(labels, logp))
    # Share of the first step's probability kept after throwing out partial
    # matches (masked: of the grammar-allowed mass; llama.cpp: of the raw mass).
    ans["label_mass"] = stats["valid_mass"]
    ans["scoring"] = {k: stats[k] for k in ("branch_nodes", "masked_nodes", "fallbacks")}
    if think_info:
        ans["thinking"] = think_info
    return ans, usage


def answer_question(backend: Backend, state, q: dict, renormalize: bool, mode: str,
                    fmt: str = "json", think: int = 0) -> tuple[dict, dict]:
    if not isinstance(q, dict) or "instructions" not in q:
        raise ShimError(400, "each question needs 'type' and 'instructions'")
    if mode == "logprobs":
        return answer_logprobs(backend, state, q, fmt, think)
    qtype = q["type"]
    labels, options = labels_and_options(q)
    if fmt == "label":
        grammar = build_lines_grammar(qtype, labels)
        prompt, think_info = with_thinking(
            backend, build_prompt(state, q, options, grammar, _HOW_LABEL_LINES[qtype], SYSTEM_LABEL, think),
            think, grammar)
        text, usage = backend.complete(prompt, grammar)
        try:
            ans = parse_lines(qtype, labels, text)
        except (StopIteration, ValueError) as e:
            raise ShimError(502, f"grammar not enforced? unparseable output {text[:200]!r}") from e
    else:
        grammar = build_grammar(qtype, labels)
        prompt, think_info = with_thinking(
            backend, build_prompt(state, q, options, grammar, _HOW_GRAMMAR[qtype], think_budget=think),
            think, grammar)
        text, usage = backend.complete(prompt, grammar)
        try:
            ans = json.loads(text)
        except json.JSONDecodeError as e:
            # The grammar should make this impossible; if it happens the
            # constraint was not applied and the answer must not be trusted.
            raise ShimError(502, f"grammar not enforced? unparseable output {text[:200]!r}") from e

    _bill_thinking(usage, think_info)
    if think_info:
        ans["thinking"] = think_info
    if q["type"] == "noul":
        return ans, usage

    raw = ans["probabilities"]
    total = sum(raw.values())
    if renormalize:
        if total <= 0:
            raise ShimError(502, "model put zero mass on every option")
        ans["probabilities"] = {k: v / total for k, v in raw.items()}
        ans["raw_probabilities"] = raw
    return ans, usage


# ─── HTTP ─────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    backend: Backend
    renormalize: bool
    mode: str
    fmt: str
    think: int
    pool: ThreadPoolExecutor
    quiet: bool

    def log_message(self, fmt, *args):
        if not self.quiet:
            sys.stderr.write("[shim] " + (fmt % args) + "\n")

    def _send(self, status: int, obj: dict):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/health", "/v1/health"):
            self._send(200, {"ok": True, "backend": self.backend.base_url, "model": self.backend.model})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/systemone":
            return self._send(404, {"error": "not found"})
        t0 = time.perf_counter()
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            questions = req.get("questions")
            if not isinstance(questions, dict) or not questions:
                raise ShimError(400, "'questions' must be a non-empty object")
            if "state" not in req:
                raise ShimError(400, "'state' is required")
            futs = {k: self.pool.submit(answer_question, self.backend, req["state"], q, self.renormalize, self.mode, self.fmt, self.think)
                    for k, q in questions.items()}
            answers, usage = {}, {"prompt_tokens": 0, "completion_tokens": 0, "backend_prompt_tokens": 0}
            for k, f in futs.items():
                ans, u = f.result()
                answers[k] = ans
                usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                usage["completion_tokens"] += u.get("completion_tokens", 0)
                usage["backend_prompt_tokens"] += u.get("backend_prompt_tokens", u.get("prompt_tokens", 0))
            usage["input_tokens"] = usage["prompt_tokens"]
            usage["output_tokens"] = usage["completion_tokens"]
            self._send(200, {"model": self.backend.model, "answers": answers, "usage": usage,
                             "latency_s": round(time.perf_counter() - t0, 4),
                             "runtime": {"shim": "jev_shim", "mode": self.mode, "format": self.fmt,
                                         "think_budget": self.think,
                                         "renormalize": self.renormalize if self.mode == "grammar" else None,
                                         "thinking": self.think > 0}})
        except ShimError as e:
            self._send(e.status, {"error": str(e)})
        except json.JSONDecodeError:
            self._send(400, {"error": "request body is not JSON"})


def detect_model(base_url: str) -> str:
    """First model id served by an OpenAI-compatible backend."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=10) as r:
            return json.loads(r.read())["data"][0]["id"]
    except Exception as e:  # noqa: BLE001 - surface any failure as a clear message
        sys.exit(f"[shim] could not list models at {base_url}/v1/models ({e}); pass --model")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8250)
    ap.add_argument("--backend", default="http://127.0.0.1:8000", help="inference server base URL")
    ap.add_argument("--backend-type", choices=["vllm", "llamacpp"], default="vllm")
    ap.add_argument("--model", default="",
                    help="served model name (vLLM); default: the first model the backend lists")
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8, help="concurrent backend calls")
    ap.add_argument("--mode", choices=["logprobs", "grammar"], default="logprobs",
                    help="logprobs: native label probabilities; grammar: model-written JSON under GBNF")
    ap.add_argument("--think-budget", dest="think", type=int, default=0,
                    help="tokens of free thinking before the grammar-constrained answer (0 = none); "
                         f"if the budget runs out the thought is cut and '{THINK_CUT_TEXT}' appended")
    ap.add_argument("--format", dest="fmt", choices=["json", "label"], default="json",
                    help="what the model writes: Jev JSON, or just the decision (shim builds the JSON)")
    ap.add_argument("--no-renormalize", dest="renormalize", action="store_false")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    model = a.model or (detect_model(a.backend) if a.backend_type == "vllm" else "llama.cpp")
    Handler.backend = (LlamaCppBackend if a.backend_type == "llamacpp" else Backend)(
        a.backend, model, a.timeout_s, a.max_tokens)
    Handler.renormalize = a.renormalize
    Handler.mode = a.mode
    Handler.fmt = a.fmt
    Handler.think = a.think
    Handler.pool = ThreadPoolExecutor(max_workers=a.workers)
    Handler.quiet = a.quiet
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"[shim] http://{a.host}:{a.port}/v1/systemone -> {a.backend} ({model}), "
          f"mode={a.mode}, format={a.fmt}, think={a.think}" + (f", renormalize={a.renormalize}" if a.mode == "grammar" else ""), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
