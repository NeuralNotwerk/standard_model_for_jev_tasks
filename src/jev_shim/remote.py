"""Remote APIs that return JSON but no logits: OpenAI-compatible chat APIs and AWS Bedrock.

The model's reply is constrained to a JSON object naming exactly one option:

    choice  {"choice": "<one of the option labels>"}
    noul    {"answer": "yes" | "no"}
    score   {"score":  "<one of the level indices>"}

OpenAI-compatible APIs get it as `response_format: {"type": "json_schema", strict}`;
Bedrock gets it as a forced tool call whose input schema is the decision (the
Converse API's way of constraining output). The schema is also shown in the prompt.

Probabilities depend on what the API offers; capabilities.probe tests every
setting once at startup and the chosen settings are held for the whole run:
  - top logprobs (some OpenAI-compatible providers): probabilities come from the
    API's top-20 logprobs at the decision tokens, with the same longest-match rule
    as local scoring; "probabilities_source": "api_top_logprobs". Only the path the
    model generated is visible, so options that branch off it and then split
    further share their mass equally (flagged in "scoring.unresolved_splits").
  - no logprobs (Bedrock, many APIs): the answer is ONE-HOT, chosen option 1 and
    every other 0; "probabilities_source": "one_hot_no_logprobs". Accuracy is
    measured normally; calibration is meaningless.
  - no structured output at all: the shim refuses to start.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .server import ShimError, _HEADING, _KIND, labels_and_options

PROBABILITIES_SOURCE = "one_hot_no_logprobs"

SYSTEM = (
    "You are a decision model. You read a state and one bounded question, then reply "
    "with a single JSON object that matches the given JSON Schema, naming exactly one "
    "option, and nothing else."
)


def PROBE_USER(schema: dict) -> str:  # noqa: N802 - constant-like helper
    """A probe prompt that invites prose, so a backend that ignores the constraint fails."""
    return ("Is the sky blue on a clear day? Think it over and explain your reasoning, then answer.\n\n"
            f"Reply with a JSON object that matches this JSON Schema:\n{json.dumps(schema)}")


def decision_schema(qtype: str, labels: list[str]) -> tuple[str, dict]:
    """(key, JSON Schema) for the decision object."""
    key = {"noul": "answer", "choice": "choice", "score": "score"}[qtype]
    return key, {
        "type": "object",
        "properties": {key: {"type": "string", "enum": list(labels)}},
        "required": [key],
        "additionalProperties": False,
    }


def build_messages(state, q: dict, options, schema: dict) -> tuple[str, str]:
    """(system, user) chat messages; the user message shows the schema to match."""
    qtype = q["type"]
    state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
    opt_lines = "\n".join(f"- {lab}: {d}" if d else f"- {lab}" for lab, d in options)
    how = {"noul": 'Set "answer" to "yes" or "no".',
           "score": 'Set "score" to the single best level.',
           "choice": 'Set "choice" to the single best option.'}[qtype]
    user = (f"State:\n{state_text}\n\n"
            f"Question type: {_KIND[qtype]}.\n{q['instructions']}\n\n"
            f"{_HEADING[qtype]}:\n{opt_lines}\n\n{how}\n\n"
            f"Reply with a JSON object that matches this JSON Schema:\n"
            f"```json\n{json.dumps(schema, ensure_ascii=False)}\n```")
    return SYSTEM, user


def extract_json(text: str) -> dict:
    """The decision object from a reply; tolerates code fences or stray prose."""
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ShimError(502, f"no JSON object in model reply: {text[:200]!r}")


def one_hot_answer(qtype: str, labels: list[str], key: str, obj: dict) -> dict:
    pick = obj.get(key)
    if not isinstance(pick, str) or pick not in labels:
        raise ShimError(502, f"model reply {obj!r} does not name one of {labels}")
    if qtype == "noul":
        return {"type": "noul", "noul": 1.0 if pick == "yes" else 0.0,
                "probabilities_source": PROBABILITIES_SOURCE}
    return {"type": qtype, key: pick, "probabilities": {lab: (1.0 if lab == pick else 0.0) for lab in labels},
            "probabilities_source": PROBABILITIES_SOURCE}


class JsonApiBackend:
    """A backend that can only return a constrained JSON reply."""

    kind = "json-api"

    def decide(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        raise NotImplementedError


def answer_pick(backend: JsonApiBackend, state, q: dict) -> tuple[dict, dict]:
    labels, options = labels_and_options(q)
    key, schema = decision_schema(q["type"], labels)
    system, user = build_messages(state, q, options, schema)
    obj, usage, *rest = backend.decide(system, user, schema)
    ans = one_hot_answer(q["type"], labels, key, obj)        # validates the reply
    lp = rest[0] if rest else None
    if lp:
        dist, info = logprob_distribution(lp["text"], lp["tokens"], key, labels)
        if dist:
            if q["type"] == "noul":
                ans["noul"] = dist["yes"]
            else:
                ans["probabilities"] = dist
            ans["probabilities_source"] = "api_top_logprobs"
            ans["scoring"] = info
    return ans, usage


def logprob_distribution(text: str, tokens: list[dict], key: str, labels: list[str]):
    """Option probabilities from an API's per-token top logprobs.

    Walks the generated tokens from where the decision value starts. At each step
    the candidates (the generated token plus its top alternatives) are matched to
    the options still in play; each option keeps its LONGEST matching candidate,
    the kept candidates are renormalized, and the walk follows the generated one.
    Branches the model did not take cannot be split further; options that share
    such a branch share its mass equally. Returns (None, info) if the token text
    cannot be aligned with the reply.
    """
    if not "".join(t.get("token", "") for t in tokens).startswith(text):  # trailing specials are fine
        return None, {"error": "logprob tokens do not reproduce the reply"}
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*"', text)
    if not m:
        return None, {"error": "decision value not found in reply"}
    start, logp, group, consumed = m.end(), {lab: 0.0 for lab in labels}, set(labels), ""
    dead, unresolved, steps, pos = set(), 0, 0, 0
    for tok in tokens:
        a, b = pos, pos + len(tok["token"])
        pos = b
        if a >= len(text):
            break                                              # past the reply (end-of-turn etc.)
        if b <= start or len(group) <= 1:
            continue
        pre = text[a:start] if a < start else ""
        cands = {c["token"]: c["logprob"] for c in tok.get("top_logprobs") or []}
        cands.setdefault(tok["token"], tok.get("logprob", 0.0))
        chosen = {}
        for cs, lp in cands.items():
            if not cs.startswith(pre) or lp is None:
                continue
            v = consumed + cs[len(pre):]
            if v == consumed:
                continue
            for lab in group:
                lq = lab + '"'
                if lq.startswith(v) or v.startswith(lq):
                    score = (min(len(v), len(lq)), lp)
                    if lab not in chosen or score > chosen[lab][0]:
                        chosen[lab] = (score, cs)
        if not chosen:
            break
        steps += 1
        for lab in group - set(chosen):
            dead.add(lab)                                  # no candidate in the top-k
        by_cand = {}
        for lab, (_, cs) in chosen.items():
            by_cand.setdefault(cs, []).append(lab)
        mx = max(cands[cs] for cs in by_cand)
        z = sum(math.exp(cands[cs] - mx) for cs in by_cand)
        for cs, labs in by_cand.items():
            share = cands[cs] - mx - math.log(z)
            for lab in labs:
                logp[lab] += share
            if cs != tok["token"] and len(labs) > 1:     # untaken branch, can't split further
                unresolved += 1
                for lab in labs:
                    logp[lab] -= math.log(len(labs))
        group = set(by_cand.get(tok["token"], []))
        consumed = consumed + tok["token"][len(pre):]
    probs = {lab: (0.0 if lab in dead else math.exp(v)) for lab, v in logp.items()}
    total = sum(probs.values())
    if total <= 0:
        return None, {"error": "no option matched the top logprobs"}
    return ({lab: p / total for lab, p in probs.items()},
            {"steps": steps, "unresolved_splits": unresolved, "options_outside_top_k": sorted(dead)})


# ─── OpenAI-compatible chat APIs ─────────────────────────────────────────

def api_url(base_url: str, path: str) -> str:
    """Join an OpenAI-style base (with or without a trailing /v1) and an endpoint path."""
    base = base_url.rstrip("/")
    return f"{base}/{path}" if re.search(r"/v\d+$", base) else f"{base}/v1/{path}"


def api_root(base_url: str, api_key: str = "") -> str:
    """The prefix under which /models and /chat/completions live.

    Bases ending in /vN are used as is. Otherwise <base>/v1 and <base> are probed
    (e.g. DeepInfra's https://api.deepinfra.com/v1/openai has no /v1 suffix).
    """
    base = base_url.rstrip("/")
    if re.search(r"/v\d+$", base):
        return base
    for root in (base + "/v1", base):
        if isinstance(_get_json(root + "/models", api_key), dict):
            return root
    return base + "/v1"


class OpenAICompatBackend(JsonApiBackend):
    """POST {root}/chat/completions with the decision constrained by response_format.

    Settings (json_schema | json_object, with or without top logprobs) are probed
    once at startup by capabilities.probe and fixed with configure(); nothing is
    switched at runtime. On OpenRouter, requests ask to be routed only to
    providers that support every parameter sent (provider.require_parameters).
    """

    kind = "openai"

    def __init__(self, base_url: str, model: str, timeout_s: float, max_tokens: int,
                 api_key_env: str = "OPENAI_API_KEY", request_options: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.api_key = os.environ.get(api_key_env, "")
        self.root = api_root(self.base_url, self.api_key)
        self.request_options = request_options or {}
        self.openrouter = "openrouter.ai" in self.root
        self.response_format = None
        self.use_logprobs = False
        self.constraint = None

    def _post(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.root + "/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise ShimError(400 if e.code == 400 else 502, f"backend HTTP {e.code}: {e.read()[:300]!r}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise ShimError(502, f"backend unreachable: {e}") from e

    def _body(self, system: str, user: str, schema: dict, fmt: str, logprobs: bool) -> dict:
        body = {"model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        body["response_format"] = ({"type": "json_schema", "json_schema": {"name": "decision", "schema": schema,
                                                                           "strict": True}}
                                   if fmt == "json_schema" else {"type": "json_object"})
        if logprobs:
            body["logprobs"], body["top_logprobs"] = True, 20
        if self.openrouter:
            body["provider"] = {"require_parameters": True}
        return {**body, **self.request_options}

    @staticmethod
    def _parse(out: dict):
        choice = (out.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        lp = ((choice.get("logprobs") or {}).get("content")) or None
        usage = out.get("usage") or {}
        return extract_json(content), content, lp, {"prompt_tokens": usage.get("prompt_tokens", 0),
                                                    "completion_tokens": usage.get("completion_tokens", 0)}

    def openrouter_metadata(self) -> dict | None:
        """The model's published supported_parameters on OpenRouter (None elsewhere)."""
        if not self.openrouter:
            return None
        meta = _get_json(self.root + "/models", self.api_key, timeout=15) or {}
        m = next((x for x in meta.get("data", []) if x.get("id") == self.model), None)
        return {"listed": m is not None, "supported_parameters": sorted(m.get("supported_parameters") or [])} if m else {"listed": False}

    def try_setting(self, fmt: str, logprobs: bool) -> tuple[bool, str]:
        """One probe request with a prose-inviting prompt; True only if the reply obeys the schema
        (and, when asked for, logprobs come back)."""
        key, schema = decision_schema("noul", ["yes", "no"])
        try:
            obj, _, lp, _ = self._parse(self._post(self._body(SYSTEM, PROBE_USER(schema), schema, fmt, logprobs)))
        except ShimError as e:
            return False, str(e)[:160]
        if set(obj) != {key} or obj[key] not in ("yes", "no"):
            return False, f"reply did not match the schema: {obj!r}"[:160]
        if logprobs and not lp:
            return False, "accepted but returned no logprobs"
        return True, "ok"

    def configure(self, fmt: str, logprobs: bool):
        self.response_format, self.use_logprobs = fmt, logprobs

    def decide(self, system: str, user: str, schema: dict):
        if self.response_format is None:
            raise ShimError(500, "backend not configured (startup probe did not run)")
        obj, content, lp, usage = self._parse(self._post(
            self._body(system, user, schema, self.response_format, self.use_logprobs)))
        if self.use_logprobs and not lp:
            raise ShimError(502, "logprobs were configured but the API returned none")
        return obj, usage, ({"text": content, "tokens": lp} if lp else None)


# ─── AWS Bedrock (Converse API) ──────────────────────────────────────────

def bedrock_region(base_url: str) -> str | None:
    """Region from bedrock://<region> or https://bedrock-runtime.<region>.amazonaws.com."""
    u = urlparse(base_url)
    if u.scheme == "bedrock":
        return u.netloc or None
    m = re.match(r"bedrock(?:-runtime)?(?:-fips)?\.([a-z0-9-]+)\.amazonaws\.com", u.netloc)
    return m.group(1) if m else None


class BedrockBackend(JsonApiBackend):
    """AWS Bedrock (Converse API) with the decision constrained to a JSON schema.

    Credentials and region come from the standard AWS chain (environment, shared
    config/credentials files, SSO, instance roles). Needs boto3 (`pip install jev-shim[aws]`).

    Constraint methods (probed once at startup by capabilities.probe, fixed with
    configure(); never switched at runtime):
      json_schema  native structured output, outputConfig.textFormat (preferred)
      forced_tool  a forced tool call whose input schema is the decision
      any_tool     toolChoice "any" with that single tool
    Bedrock returns no logprobs, so answers are one-hot.
    """

    kind = "bedrock"
    METHODS = ("json_schema", "forced_tool", "any_tool")

    def __init__(self, base_url: str, model: str, timeout_s: float, max_tokens: int):
        try:
            import boto3  # noqa: PLC0415 - optional dependency
            from botocore.config import Config  # noqa: PLC0415
        except ImportError as e:
            raise SystemExit("the Bedrock backend needs boto3: pip install 'jev-shim[aws]'") from e
        if not model:
            raise SystemExit("the Bedrock backend needs --model (a Bedrock model id or inference profile)")
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        region = bedrock_region(base_url) or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        endpoint = base_url if urlparse(base_url).scheme == "https" else None
        # Short socket timeouts with adaptive retries: a stalled connection is retried
        # well inside a client's request timeout instead of hanging it.
        self.client = boto3.client("bedrock-runtime", region_name=region, endpoint_url=endpoint,
                                   config=Config(connect_timeout=10, read_timeout=min(timeout_s, 60),
                                                 retries={"max_attempts": 4, "mode": "adaptive"}))
        self.method = None
        self.constraint = None

    def _converse(self, system: str, user: str, schema: dict, method: str) -> dict:
        req = {"modelId": self.model, "system": [{"text": system}],
               "messages": [{"role": "user", "content": [{"text": user}]}],
               "inferenceConfig": {"temperature": 0, "maxTokens": self.max_tokens}}
        if method == "json_schema":
            req["outputConfig"] = {"textFormat": {"type": "json_schema", "structure": {
                "jsonSchema": {"schema": json.dumps(schema), "name": "decision"}}}}
        else:
            choice = {"tool": {"name": "decide"}} if method == "forced_tool" else {"any": {}}
            req["toolConfig"] = {"tools": [{"toolSpec": {"name": "decide", "description": "Record the decision.",
                                                         "inputSchema": {"json": schema}}}],
                                 "toolChoice": choice}
        return self.client.converse(**req)

    @staticmethod
    def _decision(out: dict, method: str) -> dict:
        blocks = out.get("output", {}).get("message", {}).get("content") or []
        if method == "json_schema":
            return extract_json("".join(b.get("text", "") for b in blocks))
        for b in blocks:
            if "toolUse" in b:
                return b["toolUse"].get("input") or {}
        raise ShimError(502, "model did not call the decision tool")

    def try_method(self, method: str) -> tuple[bool, str]:
        """One probe request with a prose-inviting prompt; True only if the reply obeys the schema."""
        from botocore.exceptions import BotoCoreError, ClientError  # noqa: PLC0415
        key, schema = decision_schema("noul", ["yes", "no"])
        try:
            obj = self._decision(self._converse(SYSTEM, PROBE_USER(schema), schema, method), method)
        except ClientError as e:
            err = e.response.get("Error", {})
            return False, f"{err.get('Code')}: {err.get('Message', '')[:140]}"
        except (BotoCoreError, ShimError) as e:
            return False, f"{type(e).__name__}: {str(e)[:140]}"
        if set(obj) == {key} and obj[key] in ("yes", "no"):
            return True, "ok"
        return False, f"reply did not match the schema: {obj!r}"[:160]

    def configure(self, method: str):
        self.method = method

    def decide(self, system: str, user: str, schema: dict) -> tuple[dict, dict]:
        from botocore.exceptions import BotoCoreError, ClientError  # noqa: PLC0415
        if self.method is None:
            raise ShimError(500, "backend not configured (startup probe did not run)")
        try:
            out = self._converse(system, user, schema, self.method)
        except ClientError as e:
            raise ShimError(502, f"bedrock {e.response.get('Error', {}).get('Code', '')}: {e}") from e
        except BotoCoreError as e:          # timeouts, connection resets, ...
            raise ShimError(502, f"bedrock transport error: {type(e).__name__}: {e}") from e
        usage = out.get("usage") or {}
        return self._decision(out, self.method), {"prompt_tokens": usage.get("inputTokens", 0),
                                                  "completion_tokens": usage.get("outputTokens", 0)}


# ─── backend detection ───────────────────────────────────────────────────

def _get_json(url: str, api_key: str = "", timeout: float = 5.0):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001 - any failure just means "not this backend"
        return None


def detect_backend_type(base_url: str, api_key: str = "") -> str:
    """Guess vllm | llamacpp | openai | bedrock from the URL and the server's endpoints."""
    u = urlparse(base_url)
    if u.scheme == "bedrock" or u.netloc.endswith("amazonaws.com"):
        return "bedrock"
    models = _get_json(api_root(base_url, api_key) + "/models", api_key)
    owners = {str(m.get("owned_by", "")).lower() for m in (models or {}).get("data", []) if isinstance(m, dict)}
    if "vllm" in owners:
        return "vllm"
    if "llamacpp" in owners or owners & {"llama.cpp", "llama-cpp"}:
        return "llamacpp"
    root = re.sub(r"/v\d+$", "", base_url.rstrip("/"))
    props = _get_json(root + "/props")
    if isinstance(props, dict) and ("default_generation_settings" in props or "build_info" in props):
        return "llamacpp"
    version = _get_json(root + "/version")
    if isinstance(version, dict) and "version" in version and models is not None:
        return "vllm"
    if models is not None:
        return "openai"
    raise SystemExit(f"could not detect the backend at {base_url}; pass --backend-type")


def first_model(base_url: str, api_key: str = "") -> str:
    url = api_root(base_url, api_key) + "/models"
    models = _get_json(url, api_key, timeout=10)
    try:
        return models["data"][0]["id"]
    except Exception:  # noqa: BLE001
        raise SystemExit(f"could not list models at {url}; pass --model")


__all__ = ["logprob_distribution", "BedrockBackend", "JsonApiBackend", "OpenAICompatBackend", "PROBABILITIES_SOURCE",
           "answer_pick", "bedrock_region", "decision_schema", "detect_backend_type", "extract_json",
           "first_model", "one_hot_answer", "api_root", "api_url"]
