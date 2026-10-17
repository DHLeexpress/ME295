#!/usr/bin/env python3
"""Run the tactile agent on the offline mock gripper.

    python scripts/run_agent.py          # real LLM if a key is configured in .env, else the offline mock
    python scripts/run_agent.py --mock   # force the scripted offline policy (no key, no network)

Each run is logged to results/agent_runs/<timestamp>.jsonl (git-ignored); logs
contain provider, model, tool calls, answer and token usage, never the key.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shear_agent.agent import LLMClient, LLMConfigError, LLMError, MockBackend, MockLLM, run_agent  # noqa: E402

DEFAULT_TASK = ("Take book #7 out of the packed shelf: slide the fingers into the gaps beside it, "
                "pinch, and pull it out horizontally. Watch for slip while pulling.")


def show(step) -> None:
    args = ", ".join(f"{k}={v!r}" for k, v in step["input"].items())
    print(f"> {step['tool']}({args})".ljust(46) + " -> " + json.dumps(step["output"]))


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the tactile agent on the mock gripper.")
    ap.add_argument("--mock", action="store_true", help="use the scripted offline policy")
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--no-log", action="store_true", help="do not write results/agent_runs/*.jsonl")
    args = ap.parse_args()

    if args.mock:
        llm = MockLLM()
    else:
        try:
            llm = LLMClient.from_env(ROOT / ".env")
        except LLMConfigError as e:
            print(f"[info] {e} -> running the offline mock policy instead")
            llm = MockLLM()
    print(f"[agent] provider={llm.provider} model={llm.model}")

    try:
        result = run_agent(args.task, MockBackend(), llm, max_turns=args.max_turns, on_step=show)
    except LLMError as e:
        print(f"[fail] {e}")
        return 1
    print(f"\n[{result.stopped}] {result.final_text}")
    print(f"[usage] turns={result.turns} input_tokens={result.usage['input_tokens']} "
          f"output_tokens={result.usage['output_tokens']}")

    if not args.no_log:
        out_dir = ROOT / "results" / "agent_runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        record = {"provider": llm.provider, "model": llm.model, "task": args.task, "steps": result.steps,
                  "final": result.final_text, "stopped": result.stopped, "usage": result.usage}
        path.write_text(json.dumps(record) + "\n")
        print(f"[log] {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
