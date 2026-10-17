<p align="center">
  <img src="docs/figures/hero.png" width="100%" alt="A parallel-jaw gripper with thin tactile-skin fingers pulls one book out of a packed shelf; side panels show the measured shear field, the friction-cone margin controller, and an LLM agent calling tactile tools.">
</p>

# Shear-Aware Agentic Grasping

**A thin tactile skin lets a parallel-jaw gripper feel shear; an LLM agent uses that touch as a tool to grasp gently and pull books off a shelf.**

AM/ME/CE 295 research project (Caltech, Fall 2026) · Dohyun ([@DHLeexpress](https://github.com/DHLeexpress)) · Jihoon ([@jhahn2](https://github.com/jhahn2))

> **Status: week 2, laying the foundation.** The topic is drafted, not locked. Latest report: [Lab Report 1](reports/LR1_week02.md) · running log: [LOG.md](LOG.md) · [draft proposal](docs/proposal_draft.md) · [related work](docs/related_work.md)

## Idea
Parallel-jaw grippers know how far they closed and roughly how hard they squeeze, but not how close each contact is to slipping. Vision cannot see that either, least of all inside a packed shelf. Our thin tactile skin on a Robotiq 2F-140 finger measures the shear-strain field at the contact (~50 ms per frame today). We use it at three levels:

| | Layer | What we build | Compared against |
|---|---|---|---|
| **A** | Control | friction-margin grasp quality Q from measured shear → *squeeze just enough* + slip reflex | fixed force · slip-reactive regrasp · LLM-prior force (DeliGrasp-style) |
| **B** | Application | pull one book out of a tightly packed shelf with a single parallel-jaw gripper | scripted vision-only tilt → grasp → pull (Trixi-style) |
| **C** | Agent | LLM agent that calls tactile tools (`contact_state`, `shear_margin`, `incipient_slip`, …) to plan, monitor and recover | same agent with vision-only tools; learned visuo-tactile policy (RDP-style, stretch) |

## Repository layout
```
src/shear_agent/
  grasp_quality.py     friction-cone margin Q, required grip force, gentlest closure
  sim/toy_grasp.py     toy 1-D slip simulator + baseline controllers (evaluation harness)
  agent/tools.py       tactile tool schemas for the LLM agent + offline mock backend
scripts/               run_toy_benchmark.py · agent_dry_run.py · check_llm_api.py · make_hero.py
tests/                 unit tests (stdlib unittest; pytest also works)
docs/                  proposal draft, related-work note, refs.bib, figures
reports/               biweekly lab reports
results/               small generated tables (raw data stays out of git)
LOG.md                 running log
```
Hardware-side code (sensor acquisition, ROS 2 wrapper, UR10e/Robotiq drivers) will move into `sensing/` and `ros2_ws/`.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt               # numpy, matplotlib (+ pytest); or: pip install -e .
python -m unittest discover -s tests          # 13 tests
python scripts/run_toy_benchmark.py           # -> docs/figures/lr1_toy_benchmark.png, results/toy_benchmark.md
python scripts/agent_dry_run.py               # tactile tool loop, offline, no LLM
cp .env.example .env                          # add ANTHROPIC_API_KEY, then:
python scripts/check_llm_api.py               # one real tool-use round trip
```
Scripts and tests add `src/` to the path themselves, so no install is required.

## Hardware (current)
UR10e · Robotiq 2F-140 (10–125 N grip force) · thin tactile skin on one finger, second side pending · sensing PC ↔ robot PC over Ethernet/ROS.

## Data and confidentiality
Raw sensor frames, recordings and internal lab material are **not** tracked (`data/`, `recordings/`, `internal/` are git-ignored). Sensor-design details stay out of this repo until they are cleared for publication.
