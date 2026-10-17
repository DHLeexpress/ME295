#!/usr/bin/env python3
"""Check LLM API access with one real tool-use round trip (stdlib only).

1. sends a task plus our tactile tool schemas to the model,
2. executes whatever tool it calls on the offline MockBackend,
3. returns the result and prints the model's final answer.

Setup: cp .env.example .env and fill in ANTHROPIC_API_KEY (LLM_MODEL optional).
Exit code 0 = key, network and tool use all work.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shear_agent.agent import TOOLS, MockBackend, dispatch  # noqa: E402

API_URL = "https://api.anthropic.com/v1/messages"
TASK = ("You control a parallel-jaw gripper with tactile fingers that is holding book #7 mid-pull. "
        "Use the tools to check whether the grasp is about to slip and say what you would do next.")


def load_env(path: Path) -> None:
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


def post(payload: dict, key: str) -> dict:
    req = urllib.request.Request(
        API_URL, data=json.dumps(payload).encode(), method="POST",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def main() -> int:
    load_env(ROOT / ".env")
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "claude-haiku-5-5")
    if not key:
        print("ANTHROPIC_API_KEY not set: cp .env.example .env and add your key.")
        return 1

    backend = MockBackend()
    backend.execute_skill("pinch", "7")
    backend.execute_skill("pull", "7")
    messages = [{"role": "user", "content": TASK}]
    try:
        for _ in range(4):  # a few tool turns at most
            reply = post({"model": model, "max_tokens": 400, "tools": TOOLS, "messages": messages}, key)
            messages.append({"role": "assistant", "content": reply["content"]})
            calls = [b for b in reply["content"] if b.get("type") == "tool_use"]
            if not calls:
                text = " ".join(b.get("text", "") for b in reply["content"] if b.get("type") == "text")
                print(f"[ok] model={model}\nfinal answer: {text.strip()}")
                return 0
            results = []
            for c in calls:
                out = dispatch(backend, c["name"], c.get("input"))
                print(f"tool call: {c['name']}({c.get('input')}) -> {out}")
                results.append({"type": "tool_result", "tool_use_id": c["id"], "content": json.dumps(out)})
            messages.append({"role": "user", "content": results})
    except urllib.error.HTTPError as e:
        print(f"[fail] HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
        return 1
    except urllib.error.URLError as e:
        print(f"[fail] network error: {e.reason}")
        return 1
    print("[warn] tool loop did not finish within 4 turns")
    return 1


if __name__ == "__main__":
    sys.exit(main())
