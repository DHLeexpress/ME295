"""Dependency-free LLM client for the agent layer (C).

Built for a public repo whose agent will eventually drive a robot:

* the API key comes only from the environment or a git-ignored ``.env``; it is
  never printed, logged, or put into error messages;
* Anthropic Messages API by default; any OpenAI-compatible endpoint with
  ``LLM_PROVIDER=openai`` (+ ``OPENAI_API_KEY``, ``LLM_MODEL``, optional ``LLM_BASE_URL``);
* bounded time and cost: request timeout, ``max_tokens`` (Anthropic), and a capped
  number of retries with exponential backoff + jitter on 408/409/429/5xx/529 and
  network errors (``Retry-After`` is honoured);
* ``MockLLM`` gives the same interface offline, so tests and demos need no key.

Conversation history is kept in Anthropic's block format (``text`` / ``tool_use`` /
``tool_result``) and converted for OpenAI-compatible providers.
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = {"anthropic": "claude-sonnet-5-5"}
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
MAX_BACKOFF_S = 30.0


class LLMError(RuntimeError):
    """An API call failed. Messages never contain the API key."""


class LLMConfigError(LLMError):
    """Missing or invalid configuration (no key, unknown provider, no model)."""


class HTTPStatusError(Exception):
    def __init__(self, status: int, body: str = "", retry_after: Optional[str] = None):
        super().__init__(f"HTTP {status}")
        self.status, self.body, self.retry_after = status, body, retry_after


class NetworkError(Exception):
    pass


@dataclass
class LLMReply:
    blocks: List[Dict[str, Any]]             # Anthropic-style content blocks
    stop_reason: str = ""
    usage: Dict[str, int] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(b.get("text", "") for b in self.blocks if b.get("type") == "text").strip()

    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        return [b for b in self.blocks if b.get("type") == "tool_use"]


# --------------------------------------------------------------------------- config
def load_dotenv(path: os.PathLike | str = ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines, # comments, optional quotes.
    Existing environment variables win (they are never overwritten)."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        if key and val:
            os.environ.setdefault(key, val)


# --------------------------------------------------------------------------- transport
Transport = Callable[[str, Dict[str, str], Dict[str, Any], float], Dict[str, Any]]


def urllib_transport(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise HTTPStatusError(e.code, e.read().decode(errors="replace"), e.headers.get("retry-after"))
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise NetworkError(str(getattr(e, "reason", e)))


def _retry_delay(attempt: int, retry_after: Optional[str]) -> float:
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), MAX_BACKOFF_S)
        except ValueError:
            pass
    return min(MAX_BACKOFF_S, (2 ** attempt) * (0.5 + 0.5 * random.random()))


def _error_summary(body: str) -> str:
    try:
        err = json.loads(body).get("error", {})
        if isinstance(err, dict):
            return f"{err.get('type', 'error')}: {err.get('message', '')}"[:240]
        return str(err)[:240]
    except (ValueError, AttributeError):
        return body[:240]


# --------------------------------------------------------------------------- OpenAI-compatible format
def to_openai_tools(tools: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                              "parameters": t.get("input_schema", {"type": "object"})}}
            for t in tools]


def to_openai_messages(messages: Sequence[Dict[str, Any]], system: Optional[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
        elif m["role"] == "assistant":
            text = " ".join(b.get("text", "") for b in content if b.get("type") == "text").strip()
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}))}}
                     for b in content if b.get("type") == "tool_use"]
            msg: Dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:
            for b in content:
                if b.get("type") == "tool_result":
                    body = b.get("content")
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                                "content": body if isinstance(body, str) else json.dumps(body)})
                elif b.get("type") == "text":
                    out.append({"role": "user", "content": b["text"]})
    return out


def from_openai_reply(data: Dict[str, Any]) -> LLMReply:
    choice = data["choices"][0]
    msg = choice.get("message", {})
    blocks: List[Dict[str, Any]] = []
    if msg.get("content"):
        blocks.append({"type": "text", "text": msg["content"]})
    for call in msg.get("tool_calls") or []:
        fn = call.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"_unparsed_arguments": fn.get("arguments")}
        blocks.append({"type": "tool_use", "id": call["id"], "name": fn.get("name", ""), "input": args})
    usage = data.get("usage") or {}
    stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else choice.get("finish_reason", "")
    return LLMReply(blocks, stop, {"input_tokens": usage.get("prompt_tokens", 0),
                                   "output_tokens": usage.get("completion_tokens", 0)})


# --------------------------------------------------------------------------- client
class LLMClient:
    """Chat-with-tools client. ``chat(messages, tools, system) -> LLMReply``."""

    def __init__(self, provider: str = "anthropic", model: Optional[str] = None,
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 max_tokens: int = 1024, timeout: float = 60.0, max_retries: int = 4,
                 transport: Optional[Transport] = None, sleep: Callable[[float], None] = time.sleep):
        self.provider = provider.strip().lower()
        if self.provider not in KEY_ENV:
            raise LLMConfigError(f"unknown LLM_PROVIDER {provider!r} (use 'anthropic' or 'openai')")
        self.model = model or DEFAULT_MODEL.get(self.provider)
        if not self.model:
            raise LLMConfigError("set LLM_MODEL for the 'openai' provider (no default model is assumed)")
        if not api_key:
            raise LLMConfigError(f"no API key: set {KEY_ENV[self.provider]} in .env (copy .env.example)")
        self._key = api_key
        self.base_url = (base_url or OPENAI_BASE_URL).rstrip("/")
        self.max_tokens, self.timeout, self.max_retries = int(max_tokens), float(timeout), int(max_retries)
        self._transport = transport or urllib_transport
        self._sleep = sleep

    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider!r}, model={self.model!r}, api_key=***)"

    @classmethod
    def from_env(cls, env_file: os.PathLike | str = ".env", **overrides) -> "LLMClient":
        load_dotenv(env_file)
        provider = os.environ.get("LLM_PROVIDER", "anthropic").strip().lower() or "anthropic"
        cfg = dict(
            provider=provider,
            model=os.environ.get("LLM_MODEL") or None,
            api_key=os.environ.get(KEY_ENV.get(provider, "ANTHROPIC_API_KEY")),
            base_url=os.environ.get("LLM_BASE_URL") or None,
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "1024")),
        )
        cfg.update(overrides)
        return cls(**cfg)

    # public -------------------------------------------------------------------
    def chat(self, messages: Sequence[Dict[str, Any]], tools: Sequence[Dict[str, Any]] = (),
             system: Optional[str] = None) -> LLMReply:
        if self.provider == "anthropic":
            payload: Dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens,
                                       "messages": list(messages)}
            if system:
                payload["system"] = system
            if tools:
                payload["tools"] = list(tools)
            headers = {"x-api-key": self._key, "anthropic-version": ANTHROPIC_VERSION,
                       "content-type": "application/json"}
            data = self._post(ANTHROPIC_URL, headers, payload)
            usage = data.get("usage") or {}
            return LLMReply(list(data.get("content", [])), data.get("stop_reason", ""),
                            {"input_tokens": usage.get("input_tokens", 0),
                             "output_tokens": usage.get("output_tokens", 0)})
        payload = {"model": self.model, "messages": to_openai_messages(messages, system)}
        if tools:
            payload["tools"] = to_openai_tools(tools)
        headers = {"authorization": f"Bearer {self._key}", "content-type": "application/json"}
        return from_openai_reply(self._post(f"{self.base_url}/chat/completions", headers, payload))

    # internals ----------------------------------------------------------------
    def _redact(self, text: str) -> str:
        return text.replace(self._key, "***") if self._key else text

    def _post(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                return self._transport(url, headers, payload, self.timeout)
            except HTTPStatusError as e:
                if e.status not in RETRY_STATUS or attempt == self.max_retries:
                    raise LLMError(self._redact(
                        f"{self.provider} API returned HTTP {e.status} ({_error_summary(e.body)})")) from None
                delay = _retry_delay(attempt, e.retry_after)
            except NetworkError as e:
                if attempt == self.max_retries:
                    raise LLMError(self._redact(f"network error talking to {self.provider}: {e}")) from None
                delay = _retry_delay(attempt, None)
            self._sleep(delay)
        raise LLMError("unreachable")  # pragma: no cover


# --------------------------------------------------------------------------- offline stand-in
class MockLLM:
    """Scripted stand-in with the same ``chat`` interface: no key, no network.

    Policy for one book pull: insert -> pinch -> contact check -> pull -> margin
    check -> slip check -> (if slip) raise Q* and regrasp -> margin check -> report.
    """

    provider, model = "mock", "scripted-policy"
    PLAN = [
        ("execute_skill", {"skill": "insert", "book_id": "7"}),
        ("execute_skill", {"skill": "pinch", "book_id": "7"}),
        ("contact_state", {}),
        ("execute_skill", {"skill": "pull", "book_id": "7"}),
        ("shear_margin", {}),
        ("incipient_slip", {}),
    ]

    def chat(self, messages, tools=(), system=None) -> LLMReply:
        calls = [b for m in messages if m["role"] == "assistant" and isinstance(m["content"], list)
                 for b in m["content"] if b.get("type") == "tool_use"]
        results = [json.loads(b["content"]) for m in messages if m["role"] == "user"
                   and isinstance(m["content"], list) for b in m["content"] if b.get("type") == "tool_result"]
        k = len(calls)
        if k < len(self.PLAN):
            return self._call(k, *self.PLAN[k])
        if k == len(self.PLAN):
            if results and results[-1].get("incipient_slip"):
                return self._call(k, "set_margin", {"q_star": 0.3})
            return self._say("Book 7 is coming out with no slip; keep pulling at the same speed.")
        if k == len(self.PLAN) + 1:
            return self._call(k, "execute_skill", {"skill": "regrasp", "book_id": "7"})
        if k == len(self.PLAN) + 2:
            return self._call(k, "shear_margin", {})
        q = results[-1].get("Q") if results else None
        return self._say(f"Slip was starting at the contact edge, so I raised Q* to 0.3 and regrasped "
                         f"(Q = {q}). Continue pulling book 7 at the same speed.")

    @staticmethod
    def _call(k: int, name: str, args: Dict[str, Any]) -> LLMReply:
        return LLMReply([{"type": "tool_use", "id": f"mock_{k}", "name": name, "input": args}], "tool_use")

    @staticmethod
    def _say(text: str) -> LLMReply:
        return LLMReply([{"type": "text", "text": text}], "end_turn")
