# embodied-agent

A closed perceive → reason → act loop: a vision-language model looks at a MuJoCo camera
frame, reasons about it, calls tools, and acts through skill controllers — then sees the
result and goes again.

```
                    ┌──────────────── System 2 · ~0.3–1 Hz ────────────────┐
   body state ─────▶│                                                      │
   plan       ─────▶│  VLM (HF Inference)   think → choose tools → args    │
   memory     ─────▶│                                                      │
   image      ─────▶└───────────────────────┬──────────────────────────────┘
        ▲                                   │ move_to / grasp / commit / …
        │                                   ▼
        │           ┌──────────── System 1 · simulation rate ──────────────┐
        └───────────┤  IK + trajectory controllers, MuJoCo physics,        │
          measured  │  collision detection, grasp verification             │
          outcome   └──────────────────────────────────────────────────────┘
```

## Why it is built this way

Five design choices are countermeasures to published failure modes, not preferences.
Each is marked in the code where it appears.

**Proposed actions are verified before they execute.**
This is the single highest-leverage component in [HumanCLAW](https://arxiv.org/abs/2607.27180)'s
harness. Their ablation removes only the verifier:

| | FindSR | NavSR | InteractSR |
|---|---|---|---|
| baseline | 58.0% | 27.0% | 18.9% |
| no verifier | 51.0% | **2.0%** | **0.0%** |

Seeing barely moves; acting collapses. Their reasoning is that VLM spatial judgement
degrades as the rollout grows, so the planner starts hallucinating progress — claiming it
reached something still far away. `verifier.py` checks each proposal against the body's
actual state and rejects it *without executing*, returning the reason as the tool result.
Unlike theirs, most checks are exact code rather than a second model call: reachability is
decided by running IK on a scratch copy, so a waypoint outside the workspace costs a
correction instead of a wasted motion. It also guards termination — `done(success=true)`
while still holding the object is rejected, because that is the goal-detection failure in
its purest form.

**The model is told its own body state; it never infers it from pixels.**
HumanCLAW ran 1,218 episodes across nine frontier VLMs; the best scored 16.8%. The
bottleneck was not perception — once a target was genuinely visible, the strongest model's
reported sighting rate came within 5 points of ground truth. Failures concentrated after
seeing: body awareness underlies 34% of navigation failures and **81%** of interaction
failures. Their own limitations section names the likely cause:

> the agent receives only egocentric RGB and text history, with no proprioceptive body
> state or contact signal … a body-state or contact signal could be the missing input
> rather than a missing faculty.

That input is exactly what `state.py` supplies and what `SkillResult` measures, so this
repo is a direct test of the question their paper leaves open.

**The plan is written down, and its status is measured rather than claimed.**
[EvoHarness-RL](https://arxiv.org/abs/2608.05446) (UIUC + Meta AI) splits an agent's
external harness into three policy-facing roles — Belief, Progress, Experience — and
ablates each away on a frozen Qwen3-8B:

| | full BPE | w/o Belief | w/o Progress | w/o Experience |
|---|---|---|---|---|
| average | **56.4%** | 50.0% | 50.7% | 48.6% |
| Pick2 (two dependent sub-goals) | **41.7%** | 37.5% | 37.5% | 41.7% |

Two of the three already existed here — `state.py` is Belief, `memory.py` is Experience —
and Progress did not exist at all. `progress.py` is that component: a bounded list of
sub-goals the model writes with a single `commit`, re-rendered into context every step.
Committing the next sub-goal closes the previous one, so the plan advances without a
second verb.

The status transitions are made by the loop, not the model. A sub-goal is marked blocked
because a skill *actually* failed, with the simulator's own words attached:

```
=== PROGRESS (your committed plan) ===
1. [done   ] find and pick up the red cube
2. [blocked] carry it to the blue plate
     blocked by: move_to -> FAILED: waypoint 2 outside workspace
=== END PROGRESS ===
```

The held-out split here is mostly Pick2-shaped — `stack_red_on_green`, `both_on_plate`
and `green_on_plate_keep_red` each need one sub-goal finished before the next means
anything — so their numbers predict this is the component whose absence would cost most.
`--no-progress` reproduces the ablation.

**Memory lives outside the context window — and stays short.**
[VISTA](https://vista-research.github.io/) notes that a VLM's native memory is the KV
cache: attended over only implicitly, compressed, lossy, short-horizon. Once old frames
are dropped to bound the context, whatever they taught is gone. `memory.py` is a small
explicit store, re-rendered verbatim into every step.

Small is the operative word, and it is counterintuitive. HumanCLAW's ablation shows both
kinds of context saturating and then *hurting*:

| text history | 0 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| InteractSR | 0.0% | **18.9%** | 7.5% | 7.5% | 11.3% |

| images in context | 1 | 2 | 5 | 10 |
|---|---|---|---|---|
| NavSR | **27.0%** | 31.0% | 28.0% | 13.0% |
| InteractSR | **18.9%** | 9.4% | 7.5% | 3.8% |

Some history is necessary — with none, interaction success is zero — but more is worse.
Hence `image_window` defaults to 2 and the memory block shows the last 8 actions.

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
`--image-window N`, `--no-grid`, `--randomize`, and the two component ablations
`--no-verifier` (HumanCLAW) and `--no-progress` (EvoHarness-RL).

## Evaluating a harness honestly

[Rethinking the Evaluation of Harness Evolution](https://arxiv.org/abs/2607.12227) (AI2 +
UW) showed that published gains from evolving an agent's harness mostly survive neither of
two controls. On Terminal-Bench 2.1, harness evolution scored *below* the unmodified
baseline without unit tests (67.4 vs 68.2) while plain parallel sampling reached 72.3; and
on tasks disjoint from the ones it was tuned on it transferred +0.6 points on average, and
+0.0 on GPT-5.4. Their diagnosis: the edits memorise fixes rather than distil strategies.

Their closing conditions are also why measuring this domain is worth doing. Harness
evolution should matter where there is real headroom *and* performance genuinely depends
on the harness — Terminal-Bench met neither ("a shell tool and a basic prompt already
suffices"). Embodied manipulation meets both: HumanCLAW's best model reached 16.8%, and
removing one harness component took interaction success from 18.9% to 0%.

So `bench.py` builds in both controls:

```bash
python scripts/bench.py --split test --seeds 3 --budget 3   # held-out, pass@1 vs pass@3
python scripts/bench.py --split test --seeds 3 --no-verifier # ablate the verifier
python scripts/bench.py --split test --seeds 3 --no-progress # ablate the plan
```

`tasks.py` holds 8 tasks split 4 train / 4 test, disjoint, with seeded layout variation so
a policy cannot pass on memorised coordinates. Every number is reported per split, and
pass@1 always sits next to pass@k at the same budget: a change that moves pass@1 but not
pass@k bought retries, not capability.

## What is measured

`runs/<ts>/summary.json` separates what the agent *claimed* from what the world *shows*:

| field | meaning |
|---|---|
| `agent_claimed_success` | what the model asserted via `done()` |
| `verified_success` | ground-truth check against object poses |
| `failure_kind` | HumanCLAW-style bucket: localisation / body_awareness / goal_detection / interaction |
| `redundant_rate` | repeat perception calls ÷ total calls (Act Wisely) |
| `verifier_rejections` | proposals caught before execution, by tool |
| `overclaim_rate` | attempts claiming success the world does not support |
| `commits_per_episode` | how often the plan is rewritten (EvoHarness-RL's annealing rate) |

An agent that calls `done(success=true)` having achieved nothing is a goal-detection
failure, and it is only visible because the two fields are kept apart.

## Tests

```bash
pytest
```

71 tests, no API key needed — a scripted reasoner stands in for the model, so the loop
(including the closing of it) is verified without touching a provider.

## Layout

| path | role |
|---|---|
| `envs/mujoco_tabletop.py` | scene, physics, skills, measured outcomes |
| `skills/ik.py` | damped least-squares IK, gripper held pointing down |
| `perception/` | frame encoding, pixel grid overlay, crop/zoom |
| `state.py` | the body-state block (Belief) |
| `progress.py` | the committed plan, status set by measured outcomes |
| `verifier.py` | pre-execution checks on proposed calls |
| `memory.py` | external lossless memory (Experience) |
| `reasoner/` | HF Inference client, `<think>` extraction, prompts |
| `tools/` | schemas and dispatch |
| `history.py` | message list with a bounded image window |
| `loop.py` | the loop itself |
| `trace.py` | recording and metrics |
| `tasks.py` | task family with a held-out split |
| `bench.py` | benchmark runner, pass@1 vs pass@k |

## Notes on the API

The HF router is OpenAI-compatible, so the client is the official `openai` package with a
different `base_url`; a local vLLM server can be swapped in via `HF_BASE_URL`. One
constraint shapes the loop: a `role:"tool"` message carries **text only**. The frame
resulting from an action therefore has to be delivered as a separate `user` message —
that injection is what actually closes the loop, and omitting it leaves an agent that
acts and never sees the consequence.
