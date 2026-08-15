"""Progress is the component EvoHarness-RL (arXiv 2608.05446) ablates away for its
largest loss on multi-subgoal tasks, so its two load-bearing properties are tested
rather than assumed: committing advances the plan, and status comes from measured
outcomes rather than from the model's own account."""

from __future__ import annotations

from embodied_agent.memory import Memory
from embodied_agent.progress import ACTIVE, BLOCKED, DONE, Progress
from embodied_agent.reasoner.base import AgentStep, ToolCall
from embodied_agent.tools.registry import Registry, Tool, ToolResult


# ------------------------------------------------------------------------- the verb


def test_committing_the_next_subgoal_closes_the_previous_one():
    """EvoHarness-RL exposes a single write verb; advancing is what marks work finished."""
    progress = Progress()
    progress.commit("find the red cube")
    progress.commit("pick it up")

    assert [s.status for s in progress.subgoals] == [DONE, ACTIVE]
    assert progress.active is not None and progress.active.text == "pick it up"


def test_recommitting_the_same_subgoal_is_a_no_op():
    progress = Progress()
    progress.commit("find the red cube")
    message = progress.commit("find the red cube")

    assert len(progress.subgoals) == 1
    assert "already working on" in message


def test_empty_subgoal_is_rejected():
    progress = Progress()
    assert progress.commit("   ").startswith("error:")
    assert not progress.subgoals


# ------------------------------------------------------------- the adapter's updates


def test_a_failed_action_blocks_the_active_subgoal_with_the_measured_reason():
    """Status is set by what the simulator reported, never by the model asserting it."""
    progress = Progress()
    progress.commit("pick up the red cube")
    progress.note_action(False, "grasp -> FAILED: nothing between the fingers")

    active = progress.active
    assert active is not None
    assert active.status == BLOCKED
    assert "nothing between the fingers" in active.detail
    assert progress.blocked is active
    assert "blocked by:" in progress.render()


def test_a_later_success_clears_the_block():
    progress = Progress()
    progress.commit("pick up the red cube")
    progress.note_action(False, "grasp -> FAILED")
    progress.note_action(True, "grasp -> ok, holding red_cube")

    assert progress.active is not None
    assert progress.active.status == ACTIVE
    assert progress.active.detail == ""
    assert progress.blocked is None


def test_success_does_not_close_a_subgoal():
    """One subgoal spans several skills, so only an explicit commit advances the plan."""
    progress = Progress()
    progress.commit("pick up the red cube")
    progress.note_action(True, "move_to -> ok")

    assert progress.active is not None and progress.active.status == ACTIVE
    assert len(progress.subgoals) == 1


def test_note_action_without_a_committed_subgoal_is_harmless():
    progress = Progress()
    progress.note_action(False, "move_to -> FAILED")
    assert not progress.subgoals


# ------------------------------------------------------------------------- the bound


def test_the_plan_stays_bounded_and_drops_finished_work_first():
    """Their cap is 8 entries; the live subgoal must survive eviction."""
    progress = Progress(max_subgoals=3)
    for i in range(6):
        progress.commit(f"step {i}")

    assert len(progress.subgoals) == 3
    assert progress.active is not None and progress.active.text == "step 5"


def test_a_blocked_subgoal_survives_eviction_ahead_of_a_finished_one():
    progress = Progress(max_subgoals=2)
    progress.commit("step 0")
    progress.commit("step 1")
    progress.note_action(False, "move_to -> FAILED")
    progress.commit("step 2")

    texts = [s.text for s in progress.subgoals]
    assert "step 0" not in texts, "the finished subgoal should go first"
    assert "step 1" in texts, "the blocked subgoal is the record of what went wrong"


# ------------------------------------------------------------------------ rendering


def test_render_states_the_plan_is_empty_rather_than_showing_nothing():
    text = Progress().render()
    assert "nothing committed yet" in text


def test_commits_are_counted_for_the_annealing_metric():
    progress = Progress()
    progress.commit("a")
    progress.commit("b")
    progress.commit("b")  # a no-op for the plan, but still a spent step
    assert progress.commits == 3


# ----------------------------------------------------------------------- the tool


def test_commit_tool_dispatches_into_the_plan():
    """The component has to be ablatable, which means the tool disappears with it."""
    registry = Registry()
    memory = Memory()
    progress = Progress()

    # Mirror what build_registry does, without needing a live simulator.
    registry.register(
        Tool(
            name="commit",
            description="",
            parameters={"type": "object", "properties": {}, "required": []},
            fn=lambda subgoal: ToolResult(progress.commit(str(subgoal))),
        )
    )
    call = ToolCall("c1", "commit", {"subgoal": "find the red cube"}, "{}")
    result = registry.dispatch(call)

    assert not result.is_error
    assert progress.commits == 1
    assert memory.render()  # memory stays independent of progress


def test_agent_step_carrying_a_commit_parses_like_any_other_call():
    step = AgentStep("thinking", "", [ToolCall("c1", "commit", {"subgoal": "grasp"}, "{}")])
    assert step.tool_calls[0].name == "commit"


# --------------------------------------------------------------- through the real loop


class _Scripted:
    """Replays turns and keeps the context it was handed, so we can assert on it."""

    def __init__(self, turns):
        self.turns, self.seen = turns, []

    def act(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        index = len(self.seen) - 1
        if index >= len(self.turns):
            return AgentStep("done", "", [])
        name, args = self.turns[index]
        return AgentStep(f"turn {index}", "", [ToolCall(f"c{index}", name, args, "{}")])


def _user_text(messages):
    """Every piece of text the model was actually shown."""
    out = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            out.extend(p.get("text", "") for p in content if isinstance(p, dict))
    return "\n".join(out)


def test_progress_block_reaches_the_model_and_tracks_a_real_failure():
    """The whole point is that the plan is re-rendered into context every step, and that
    a skill failure measured by the simulator shows up there without the model saying so."""
    from embodied_agent.envs.mujoco_tabletop import TabletopEnv
    from embodied_agent.loop import run_episode
    from embodied_agent.tools.builtin import build_registry

    env = TabletopEnv(image_size=(240, 320))
    progress = Progress()
    memory = Memory()
    try:
        reasoner = _Scripted(
            [
                ("commit", {"subgoal": "pick up the red cube"}),
                # Nothing is between the fingers here, so the simulator reports a failure.
                ("grasp", {}),
                ("done", {"success": False, "reason": "stopping"}),
            ]
        )
        run_episode(
            env,
            reasoner,
            build_registry(env, memory, progress=progress, allow_privileged=True),
            memory,
            task="Put the red cube on the blue plate.",
            max_steps=4,
            progress=progress,
            verifier=None,
            verbose=False,
        )
    finally:
        env.close()

    assert progress.commits == 1
    blocked = progress.blocked
    assert blocked is not None, "the failed grasp should have blocked the active subgoal"
    assert blocked.text == "pick up the red cube"

    shown = _user_text(reasoner.seen[-1])
    assert "=== PROGRESS" in shown, "the plan must be rendered into the model's context"
    assert "pick up the red cube" in shown
    assert "blocked by:" in shown


def test_progress_is_ablatable_end_to_end():
    """--no-progress must remove both the block and the tool, or the ablation is not one."""
    from embodied_agent.envs.mujoco_tabletop import TabletopEnv
    from embodied_agent.loop import run_episode
    from embodied_agent.tools.builtin import build_registry

    env = TabletopEnv(image_size=(240, 320))
    memory = Memory()
    try:
        registry = build_registry(env, memory, progress=None, allow_privileged=True)
        assert "commit" not in [t.name for t in registry.available()]

        reasoner = _Scripted([("done", {"success": False, "reason": "stopping"})])
        summary = run_episode(
            env, reasoner, registry, memory,
            task="Put the red cube on the blue plate.",
            max_steps=2, progress=None, verifier=None, verbose=False,
        )
    finally:
        env.close()

    assert summary["progress_commits"] == 0
    assert "=== PROGRESS" not in _user_text(reasoner.seen[0])
