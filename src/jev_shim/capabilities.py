"""Probe a model server once at startup, then fix the settings for the whole run.

probe() tests every setting the shim could use on this backend and records what
works. resolve() turns the user's request into concrete settings: an option
left on "auto" gets the best supported value; an option given explicitly must
be supported, otherwise the shim refuses to start with a message saying what
was asked for and what the server supports. Nothing is switched at runtime.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from .server import (Backend, ShimError, build_pick_grammar, thinking_grammar)

CHATML_PROBE = ("<|im_start|>user\n{q}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")


@dataclass
class Capabilities:
    backend: str
    model: str
    checks: dict = field(default_factory=dict)      # check name -> "ok" | reason it failed
    modes: list = field(default_factory=list)       # logprobs | grammar | pick
    formats: list = field(default_factory=list)     # json | label
    constraints: list = field(default_factory=list)  # gbnf | json_schema | json_object | forced_tool | any_tool
    logprobs_with: list = field(default_factory=list)  # remote: constraints that also returned logprobs
    thinking: bool = False
    decode_tok_s: float | None = None                  # single-stream generation speed, measured

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Settings:
    backend: str
    model: str
    mode: str            # logprobs | grammar | pick
    fmt: str             # json | label
    think: int
    constraint: str      # gbnf | json_schema | json_object | forced_tool | any_tool
    probabilities: str   # logits (grammar-masked) | model_written | api_top_logprobs | one_hot

    def as_dict(self) -> dict:
        return asdict(self)


def _check(checks: dict, name: str, fn) -> bool:
    try:
        ok, detail = fn()
    except (ShimError, OSError, KeyError, ValueError, TypeError) as e:
        ok, detail = False, f"{type(e).__name__}: {str(e)[:160]}"
    checks[name] = "ok" if ok else detail
    return ok


# ─── local servers (vLLM, llama.cpp): raw prompts, GBNF, logits ─────────

def probe_local(kind: str, backend: Backend) -> Capabilities:
    caps = Capabilities(kind, backend.model, constraints=["gbnf"])
    ck = caps.checks

    def chatml():
        n = len(backend.tokenize("<|im_start|>"))
        return n == 1, "ok" if n == 1 else f"<|im_start|> is {n} tokens; prompts need a ChatML (Qwen-style) model"

    def grammar_enforced():
        # Ask for prose while the grammar only allows yes/no: only an enforced grammar complies.
        text, _ = backend.complete(CHATML_PROBE.format(q="Write a long poem about the sea."),
                                   'root ::= "yes" | "no"\n')
        return text in ("yes", "no"), f"grammar ignored: model wrote {text[:60]!r}"

    def masked_logprobs():
        ids = backend.tokenize(CHATML_PROBE.format(q="Is the sky blue? Answer yes or no."))
        top, masked = backend.node_top(ids, 'root ::= "no"\n')
        texts = [backend.token_text(t) for t in top]
        bad = [t for t in texts if not t or not "no".startswith(t)]
        return (masked and bool(top) and not bad,
                f"probabilities are not grammar-masked (returned {texts[:6]})")

    def exact_lookup():
        ids = backend.tokenize(CHATML_PROBE.format(q="Is the sky blue? Answer yes or no."))
        tok = backend.tokenize("yes")[0]
        lp = backend.raw_logprobs_of(ids, [tok])[0]
        return lp is not None and lp <= 0, f"unexpected logprob {lp!r}"

    def thinking():
        # Also times single-stream generation, used to warn about thinking's latency cost.
        head = CHATML_PROBE.format(q="Explain step by step why the sky is blue, then answer yes or no.")
        head = head[: -len("\n</think>\n\n")]
        t0 = time.perf_counter()
        text, n, _ = backend.think(head, 64, thinking_grammar(build_pick_grammar("noul", ["yes", "no"]), 64))
        dt = time.perf_counter() - t0
        if n > 0 and dt > 0:
            caps.decode_tok_s = n / dt
        return n > 0, "no thinking tokens generated"

    has_chatml = _check(ck, "chatml_prompts", chatml)
    if _check(ck, "grammar_enforced", grammar_enforced):
        caps.modes.append("grammar")
    if _check(ck, "grammar_masked_logprobs", masked_logprobs):
        caps.modes.insert(0, "logprobs")
    _check(ck, "exact_logprob_lookup", exact_lookup)
    if has_chatml:
        caps.formats = ["json", "label"]
        caps.thinking = _check(ck, "thinking", thinking)
    return caps


# ─── OpenAI-compatible chat APIs ─────────────────────────────────────────

def probe_openai(backend) -> Capabilities:
    caps = Capabilities("openai", backend.model, formats=["json"])
    meta = backend.openrouter_metadata()
    if meta is not None:
        sp = set(meta.get("supported_parameters", []))
        caps.checks["openrouter_listing"] = ("ok" if meta.get("listed") else "model not listed on OpenRouter")
        caps.checks["openrouter_supported_parameters"] = ", ".join(sorted(
            sp & {"structured_outputs", "response_format", "logprobs", "top_logprobs"})) or "none relevant"
    for fmt in ("json_schema", "json_object"):
        if _check(caps.checks, fmt, lambda f=fmt: backend.try_setting(f, False)):
            caps.constraints.append(fmt)
            if _check(caps.checks, f"{fmt}+logprobs", lambda f=fmt: backend.try_setting(f, True)):
                caps.logprobs_with.append(fmt)
    if caps.constraints:
        caps.modes.append("pick")
    if caps.logprobs_with:
        caps.modes.insert(0, "logprobs")
    return caps


# ─── AWS Bedrock ─────────────────────────────────────────────────────────

def probe_bedrock(backend) -> Capabilities:
    caps = Capabilities("bedrock", backend.model, formats=["json"])
    for method in ("json_schema", "forced_tool", "any_tool"):
        if _check(caps.checks, method, lambda m=method: backend.try_method(m)):
            caps.constraints.append(method)
    if caps.constraints:
        caps.modes.append("pick")
    return caps


def probe(kind: str, backend) -> Capabilities:
    return {"vllm": probe_local, "llamacpp": probe_local}.get(kind, None)(kind, backend) \
        if kind in ("vllm", "llamacpp") else (probe_openai(backend) if kind == "openai" else probe_bedrock(backend))


# ─── resolve the request against what the server supports ───────────────

class Refused(SystemExit):
    pass


def _refuse(what: str, asked, supported, caps: Capabilities):
    failed = {k: v for k, v in caps.checks.items() if v != "ok"}
    detail = "".join(f"\n    {k}: {v}" for k, v in failed.items())
    raise Refused(f"[shim] refusing to start: {what}={asked!r} is not supported by "
                  f"{caps.backend} ({caps.model}); supported: {supported or 'none'}"
                  + (f"\n  failed checks:{detail}" if detail else ""))


def resolve(caps: Capabilities, mode: str = "auto", fmt: str = "auto", think: int = 0,
            constraint: str = "auto") -> tuple[Settings, list[str]]:
    """Concrete settings for this run, plus warnings.

    Explicit options must be supported or the shim refuses to start. Options left on
    "auto" get the best supported value; when that value is a degradation (slower,
    or zero-valued probabilities) a warning says so. An explicitly chosen option
    never triggers a warning about itself.
    """
    local = caps.backend in ("vllm", "llamacpp")
    warnings = []
    if not caps.modes:
        _refuse("output constraint", "any", [], caps)
    # mode
    explicit_mode = mode != "auto"
    if not explicit_mode:
        mode = caps.modes[0]
    elif mode not in caps.modes:
        _refuse("--mode", mode, caps.modes, caps)
    # format
    if fmt == "auto":
        fmt = "json"
    if fmt not in caps.formats:
        _refuse("--format", fmt, caps.formats, caps)
    # thinking
    if think and not caps.thinking:
        _refuse("--think-budget", think, "thinking needs a local ChatML model (vLLM / llama.cpp)", caps)
    # constraint
    allowed = caps.constraints if (local or mode != "logprobs") else caps.logprobs_with
    explicit_constraint = constraint != "auto"
    if not explicit_constraint:
        constraint = allowed[0]
    elif constraint not in allowed:
        _refuse("--constraint", constraint, allowed, caps)

    # Thinking is warned about even when explicit: its cost is large and easy to underestimate.
    if think:
        rate = caps.decode_tok_s
        est = (f"up to ~{think / rate:.1f} s extra per question at this server's measured "
               f"{rate:.0f} tok/s single-stream" if rate else "up to the full budget's generation time per question")
        warnings.append(
            f"EXTREME PERFORMANCE IMPACT: --think-budget {think} generates up to {think} extra tokens per "
            f"question ({est}), plus prefill of the thought. In our JevBench runs, 500-1,500-token budgets "
            "raised median latency from ~0.2 s to 1.3-3 s and p95 to 3-20 s, added 230-800 output tokens "
            "per decision, and collapsed the JevBench Speed and Cost scores, while raising accuracy.")

    # warnings for degradations the user did not ask for
    if not explicit_mode and local and mode == "grammar":
        warnings.append("SPEED: grammar-masked logprobs are unavailable "
                        f"({caps.checks.get('grammar_masked_logprobs')}); using grammar mode, which generates "
                        "~30-80 tokens per decision instead of ~1, with model-written probabilities.")
    if not explicit_mode and mode == "pick":
        warnings.append("ZERO PROBABILITIES: this backend returns no logprobs; answers are one-hot "
                        "(chosen option 1, every other option 0), so calibration is meaningless.")
    if local and mode == "logprobs" and caps.checks.get("exact_logprob_lookup") != "ok":
        warnings.append("exact logprob lookup is unavailable: a decision whose option falls outside the "
                        "grammar-masked top 20 will fail instead of being looked up.")
    if not explicit_constraint and caps.backend == "bedrock" and constraint != "json_schema":
        warnings.append(f"SPEED/ACCURACY: native structured output is unsupported "
                        f"({caps.checks.get('json_schema')}); using {constraint}, which was slower and less "
                        "accurate than native JSON schema on most models tested.")
    if not explicit_constraint and caps.backend == "openai" and constraint == "json_object":
        warnings.append("strict JSON schema is unsupported; using JSON mode, which the server does not hold to "
                        "the schema (replies are validated, and invalid ones fail).")

    probabilities = ({"logprobs": "logits (grammar-masked)", "grammar": "model_written"}[mode] if local
                     else "api_top_logprobs" if mode == "logprobs" else "one_hot")
    return Settings(caps.backend, caps.model, mode, fmt, think, constraint, probabilities), warnings
