#!/usr/bin/env python3
"""Check LLM API access end to end: key configured, endpoint reachable, tool use works.

Setup once:  cp .env.example .env   and fill in the key (the file is git-ignored).
Runs a short agent episode (at most 4 turns) against the offline mock gripper.
Exit code 0 means the key, the network and tool calling all work.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shear_agent.agent import LLMClient, LLMConfigError, LLMError, MockBackend, run_agent  # noqa: E402

TASK = ("Book #7 is mid-pull. Use one or two tools to check whether the grasp is about to slip, "
        "then say in one sentence what you would do next.")


def main() -> int:
    try:
        llm = LLMClient.from_env(ROOT / ".env", max_retries=2)
    except LLMConfigError as e:
        print(f"[setup] {e}")
        return 1
    print(f"[check] provider={llm.provider} model={llm.model} key=set (not shown)")

    backend = MockBackend()
    for skill in ("insert", "pinch", "pull"):
        backend.execute_skill(skill, "7")
    try:
        result = run_agent(TASK, backend, llm, max_turns=4)
    except LLMError as e:
        print(f"[fail] {e}")
        return 1
    for step in result.steps:
        print(f"tool call: {step['tool']}({step['input']}) -> {step['output']}")
    print(f"answer: {result.final_text}")
    print(f"usage: {result.usage}")
    if not result.steps:
        print("[warn] the model answered without calling a tool")
        return 1
    print("[ok] key, network and tool use all work")
    return 0


if __name__ == "__main__":
    sys.exit(main())
