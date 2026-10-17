"""Tool-use loop for the tactile agent (layer C).

The loop is provider-agnostic: ``llm`` is anything with
``chat(messages, tools, system) -> LLMReply`` (``LLMClient`` or ``MockLLM``).
Turns, tokens and time are bounded by the caller (``max_turns``) and the client.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .tools import Q_STAR_MAX, TOOLS, dispatch

SYSTEM_PROMPT = f"""You control a UR10e arm with a Robotiq 2F-140 parallel-jaw gripper. Its fingers carry \
a thin tactile skin that measures shear at the contact.
You never command forces directly: you choose manipulation skills and the grasp-margin target Q* \
(0 to {Q_STAR_MAX}); a low-level controller keeps the grip force inside the gripper's limits.
Task context: take one book out of a tightly packed shelf. There is no headroom to lift it, so slide \
the thin fingers into the gaps beside it (insert), pinch, and pull it out horizontally.
Rules: check contact before pulling. While pulling, check shear_margin and incipient_slip. If slip is \
starting, raise Q* (at most {Q_STAR_MAX}) and regrasp before continuing. If a tool returns an error twice, \
stop and explain. Keep answers short."""


@dataclass
class AgentResult:
    final_text: str
    steps: List[Dict[str, Any]]             # [{"tool", "input", "output"}]
    turns: int
    stopped: str                            # "done" | "max_turns"
    usage: Dict[str, int] = field(default_factory=dict)


def run_agent(task: str, backend: Any, llm: Any, tools: Sequence[Dict[str, Any]] = TOOLS,
              system: Optional[str] = SYSTEM_PROMPT, max_turns: int = 10,
              on_step: Optional[Callable[[Dict[str, Any]], None]] = None) -> AgentResult:
    messages: List[Dict[str, Any]] = [{"role": "user", "content": task}]
    steps: List[Dict[str, Any]] = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    for turn in range(1, max_turns + 1):
        reply = llm.chat(messages, tools=tools, system=system)
        for k in usage:
            usage[k] += int(reply.usage.get(k, 0) or 0)
        messages.append({"role": "assistant", "content": reply.blocks})
        calls = reply.tool_calls
        if not calls:
            return AgentResult(reply.text, steps, turn, "done", usage)
        results = []
        for call in calls:
            args = call.get("input") or {}
            out = dispatch(backend, call["name"], args)
            step = {"tool": call["name"], "input": args, "output": out}
            steps.append(step)
            if on_step:
                on_step(step)
            result = {"type": "tool_result", "tool_use_id": call["id"], "content": json.dumps(out)}
            if out.get("ok") is False:
                result["is_error"] = True
            results.append(result)
        messages.append({"role": "user", "content": results})
    return AgentResult("stopped: turn limit reached", steps, max_turns, "max_turns", usage)
