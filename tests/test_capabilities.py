"""Resolution rules: auto picks the best supported setting (with warnings for
degradations); explicit options must be supported or the shim refuses."""

import pytest

from jev_shim.capabilities import Capabilities, Refused, resolve

LOCAL_FULL = Capabilities("vllm", "m", checks={"grammar_masked_logprobs": "ok", "exact_logprob_lookup": "ok"},
                          modes=["logprobs", "grammar"], formats=["json", "label"], constraints=["gbnf"],
                          thinking=True)
LOCAL_NO_LOGITS = Capabilities("llamacpp", "m", checks={"grammar_masked_logprobs": "not masked",
                                                        "exact_logprob_lookup": "ok"},
                               modes=["grammar"], formats=["json", "label"], constraints=["gbnf"], thinking=True)
OPENAI_LP = Capabilities("openai", "m", checks={"json_schema": "ok"}, modes=["logprobs", "pick"],
                         formats=["json"], constraints=["json_schema", "json_object"], logprobs_with=["json_schema"])
BEDROCK_TOOL = Capabilities("bedrock", "m", checks={"json_schema": "ValidationException"}, modes=["pick"],
                            formats=["json"], constraints=["forced_tool", "any_tool"])
NOTHING = Capabilities("bedrock", "m", checks={"json_schema": "no", "forced_tool": "no", "any_tool": "no"})


def test_auto_picks_best_and_is_silent_when_nothing_degrades():
    s, w = resolve(LOCAL_FULL)
    assert (s.mode, s.fmt, s.constraint, s.probabilities) == ("logprobs", "json", "gbnf", "logits (grammar-masked)")
    assert w == []


def test_auto_degradation_warns_about_speed():
    s, w = resolve(LOCAL_NO_LOGITS)
    assert s.mode == "grammar" and any(x.startswith("SPEED") for x in w)


def test_explicit_choice_suppresses_its_own_warning():
    s, w = resolve(LOCAL_NO_LOGITS, mode="grammar")
    assert s.mode == "grammar" and not any(x.startswith("SPEED") for x in w)


def test_one_hot_warns_unless_asked_for():
    _, w = resolve(BEDROCK_TOOL)
    assert any(x.startswith("ZERO PROBABILITIES") for x in w) and any(x.startswith("SPEED/ACCURACY") for x in w)
    _, w = resolve(BEDROCK_TOOL, mode="pick", constraint="forced_tool")
    assert w == []


def test_remote_logprobs_use_a_constraint_that_returned_them():
    s, _ = resolve(OPENAI_LP)
    assert (s.mode, s.constraint, s.probabilities) == ("logprobs", "json_schema", "api_top_logprobs")
    with pytest.raises(Refused):
        resolve(OPENAI_LP, mode="logprobs", constraint="json_object")


@pytest.mark.parametrize("caps,kwargs", [
    (LOCAL_NO_LOGITS, {"mode": "logprobs"}),
    (OPENAI_LP, {"fmt": "label"}),
    (BEDROCK_TOOL, {"think": 500}),
    (BEDROCK_TOOL, {"constraint": "json_schema"}),
    (BEDROCK_TOOL, {"mode": "logprobs"}),
    (NOTHING, {}),
])
def test_unsupported_requests_refuse(caps, kwargs):
    with pytest.raises(Refused):
        resolve(caps, **kwargs)


def test_thinking_always_warns_about_performance_even_when_explicit():
    caps = Capabilities(**{**LOCAL_FULL.as_dict(), "decode_tok_s": 100.0})
    _, w = resolve(caps, mode="logprobs", think=1000)
    assert any(x.startswith("EXTREME PERFORMANCE IMPACT") and "~10.0 s" in x for x in w)
    _, w = resolve(caps, mode="logprobs")
    assert not any("EXTREME" in x for x in w)
