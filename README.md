# embodied-agent

A closed perceive → reason → act loop: a vision-language model looks at a MuJoCo camera
frame, reasons about it, calls tools, and acts through skill controllers — then sees the
result and goes again.

```
                    ┌──────────────── System 2 · ~0.3–1 Hz ────────────────┐
   body state ─────▶│  VLM (HF Inference)   think → choose tools → args    │
   memory     ─────▶│                                                      │
   image      ─────▶└───────────────────────┬──────────────────────────────┘
        ▲                                   │ move_to / grasp / measure / …
        │                                   ▼
        │           ┌──────────── System 1 · simulation rate ──────────────┐
        └───────────┤  IK + trajectory controllers, MuJoCo physics,        │
          measured  │  collision detection, grasp verification             │
          outcome   └──────────────────────────────────────────────────────┘
```

## Why it is built this way

Three design choices are countermeasures to published failure modes, not preferences.
Each is marked in the code where it appears.

**The model is told its own body state; it never infers it from pixels.**
[HumanCLAW](https://arxiv.org/abs/2607.27180) evaluated exactly this architecture — an
off-the-shelf VLM issuing skill commands that a controller executes — over 1,218 episodes
and nine frontier VLMs. The best scored 16.8%. The bottleneck was not perception: up to
81% of interaction-stage failures came from egocentric self-localisation and body
awareness, i.e. the model could not tell where its body was, whether it had arrived, or
whether it had collided. So every observation leads with a measured `BODY STATE` block
(`src/embodied_agent/state.py`), and skills report what actually happened rather than
returning a success flag (`SkillResult`).

**Memory lives outside the context window.**
[VISTA](https://vista-research.github.io/) notes that a VLM's native memory is the KV
cache: attended over only implicitly, compressed, lossy, short-horizon. Once old frames
are dropped to bound the context, whatever they taught is gone. `memory.py` is a small
explicit store, re-rendered verbatim into every step.

**Tool descriptions say when *not* to call.**
[Act Wisely](https://arxiv.org/abs/2604.08545) found agentic multimodal models invoke
tools reflexively even when the answer is already visible — redundant-call rates near
98% under a naive objective — adding latency and noise that derails reasoning. The
abstention condition sits in each schema, and `trace.py` counts the redundant-call rate
so the regression cannot happen quietly.

**Plans are trajectories, not prose.** Following
[ACoT-VLA](https://arxiv.org/abs/2601.11404), `move_to` takes a waypoint list so the
model expresses its plan in the language of actions.

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env      # add your HF_TOKEN
```

## Run

```bash
# 1. Does this model/provider actually do vision + tool calling?
python scripts/check_model.py --shortlist   # probes the candidates, prints the winner
python scripts/check_model.py --model <id>  # or check one

# 2. One episode.
python scripts/run_episode.py --task "Put the red cube on the blue plate."

# 3. Read what happened.
python scripts/replay_trace.py runs/<timestamp>   # writes replay.html
```

If `check_model.py` reports no native tool calling, add `--json-mode`: the model is then
constrained with `response_format` to emit the same call shape as JSON, which feeds the
identical dispatcher.

Useful flags: `--privileged` enables `list_objects` (ground-truth poses — good for
bootstrapping, bypasses vision, so keep it off when measuring), `--steps N`,
`--image-window N`, `--no-grid`, `--randomize`.

## What is measured

`runs/<ts>/summary.json` separates what the agent *claimed* from what the world *shows*:

| field | meaning |
|---|---|
| `agent_claimed_success` | what the model asserted via `done()` |
| `verified_success` | ground-truth check against object poses |
| `failure_kind` | HumanCLAW-style bucket: localisation / body_awareness / goal_detection / interaction |
| `redundant_rate` | repeat perception calls ÷ total calls (Act Wisely) |

An agent that calls `done(success=true)` having achieved nothing is a goal-detection
failure, and it is only visible because the two fields are kept apart.

## Tests

```bash
pytest
```

35 tests, no API key needed — a scripted reasoner stands in for the model, so the loop
(including the closing of it) is verified without touching a provider.

## Layout

| path | role |
|---|---|
| `envs/mujoco_tabletop.py` | scene, physics, skills, measured outcomes |
| `skills/ik.py` | damped least-squares IK, gripper held pointing down |
| `perception/` | frame encoding, pixel grid overlay, crop/zoom |
| `state.py` | the body-state block |
| `memory.py` | external lossless memory |
| `reasoner/` | HF Inference client, `<think>` extraction, prompts |
| `tools/` | schemas and dispatch |
| `history.py` | message list with a bounded image window |
| `loop.py` | the loop itself |
| `trace.py` | recording and metrics |

## Notes on the API

The HF router is OpenAI-compatible, so the client is the official `openai` package with a
different `base_url`; a local vLLM server can be swapped in via `HF_BASE_URL`. One
constraint shapes the loop: a `role:"tool"` message carries **text only**. The frame
resulting from an action therefore has to be delivered as a separate `user` message —
that injection is what actually closes the loop, and omitting it leaves an agent that
acts and never sees the consequence.
