#!/usr/bin/env python3
"""Offline dry run of the agent's tactile tool loop (no LLM, no hardware).

A scripted policy stands in for the LLM so the tool layer can be exercised end to
end today: pull a book, notice incipient slip by touch, raise the margin, regrasp.
``scripts/check_llm_api.py`` runs the same tools through a real model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shear_agent.agent import MockBackend, dispatch  # noqa: E402


def call(backend, name, **args):
    out = dispatch(backend, name, args)
    arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    print(f"> {name}({arg_str})".ljust(46) + " -> " + json.dumps(out))
    return out


def main() -> None:
    be = MockBackend()
    call(be, "execute_skill", skill="tilt", book_id="7")
    call(be, "execute_skill", skill="pinch", book_id="7")
    call(be, "contact_state")
    call(be, "execute_skill", skill="pull", book_id="7")
    call(be, "shear_margin")
    if call(be, "incipient_slip")["incipient_slip"]:
        print("# touch says the book is about to slip -> raise the margin and regrasp")
        call(be, "set_margin", q_star=0.3)
        call(be, "execute_skill", skill="regrasp", book_id="7")
        call(be, "shear_margin")
    call(be, "execute_skill", skill="release", book_id="7")


if __name__ == "__main__":
    main()
