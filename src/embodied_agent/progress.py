"""The agent's committed plan, externalised.

EvoHarness-RL (arXiv 2608.05446) organises an agent's external harness into three
policy-facing roles -- Belief, Progress and Experience -- and ablates each one away on
ALFWorld. Two of the three already existed here: `state.py` is Belief (the measured body
state) and `memory.py` is Experience (facts that outlive the image window). Progress did
not exist at all.

Their ablation on a frozen Qwen3-8B says what that costs. Removing Progress drops the
average from 56.4% to 50.7%, and the damage concentrates exactly where you would expect:
on Pick2, the task family with two dependent sub-goals, where it is the largest single
loss of any component. Our held-out split is mostly that shape -- `stack_red_on_green`,
`both_on_plate`, `green_on_plate_keep_red` all require finishing one sub-goal before the
next becomes meaningful -- so this is the component whose absence their numbers predict
we would feel most.

The design follows theirs: a bounded list of (sub-goal, status) records that the model
writes with a single `commit` action, re-rendered into context every step. Committing a
new sub-goal closes the previous one, so the plan advances without needing a second verb.

Status transitions are made by the environment adapter, not by the model. That mirrors
their belief tracker, which is "a rule-based parser over action-observation pairs" with
no LLM call: a sub-goal is marked blocked because a skill actually failed, never because
the model said so. The plan is the model's; the status is measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: EvoHarness-RL caps ALFWorld's committed plan at 8 entries. The same bound applies for
#: the same reason -- their context ablation, and HumanCLAW's, both show long history
#: hurting rather than helping, so the plan has to stay short enough to read at a glance.
MAX_SUBGOALS = 8

ACTIVE = "active"
DONE = "done"
BLOCKED = "blocked"


@dataclass
class Subgoal:
    text: str
    status: str = ACTIVE
    #: Why it is blocked, taken verbatim from the measured skill result.
    detail: str = ""


@dataclass
class Progress:
    """A bounded, ordered record of what the agent set out to do and how it went."""

    subgoals: list[Subgoal] = field(default_factory=list)
    max_subgoals: int = MAX_SUBGOALS
    #: How many times the model called `commit`. The analogue of EvoHarness-RL's harness
    #: annealing plot: after training, their agent settles at roughly one harness call per
    #: episode, so the rate is the thing to watch, not just whether the tool exists.
    commits: int = 0

    # ------------------------------------------------------------------ the one verb

    def commit(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "error: subgoal must not be empty"

        self.commits += 1
        previous = self.active
        if previous is not None:
            if previous.text == text:
                return f"already working on {text!r}; it is still the active subgoal"
            # Committing the next step is what closes the current one. A blocked sub-goal
            # stays blocked -- it is the record of something that did not work.
            if previous.status == ACTIVE:
                previous.status = DONE

        self.subgoals.append(Subgoal(text=text))
        self._evict()
        return f"committed subgoal {len(self.subgoals)}: {text!r}"

    # ---------------------------------------------------------- the adapter's updates

    def note_action(self, ok: bool, reason: str) -> None:
        """Update the active sub-goal from a measured skill result.

        Called by the loop after any world-mutating action. Deterministic and free: no
        model call is involved in deciding whether progress stalled.
        """
        active = self.active
        if active is None:
            return
        if ok:
            # Success does not close a sub-goal -- one sub-goal usually spans several
            # skills. It only clears a block, because the arm evidently moved again.
            if active.status == BLOCKED:
                active.status = ACTIVE
                active.detail = ""
        else:
            active.status = BLOCKED
            active.detail = reason

    # -------------------------------------------------------------------- inspection

    @property
    def active(self) -> Subgoal | None:
        for subgoal in reversed(self.subgoals):
            if subgoal.status in (ACTIVE, BLOCKED):
                return subgoal
        return None

    @property
    def blocked(self) -> Subgoal | None:
        active = self.active
        return active if active is not None and active.status == BLOCKED else None

    def _evict(self) -> None:
        """Keep the plan bounded, dropping finished work first so the live part survives."""
        while len(self.subgoals) > self.max_subgoals:
            for index, subgoal in enumerate(self.subgoals):
                if subgoal.status == DONE:
                    del self.subgoals[index]
                    break
            else:
                del self.subgoals[0]

    def render(self) -> str:
        lines = ["=== PROGRESS (your committed plan) ==="]
        if not self.subgoals:
            lines.append(
                "(nothing committed yet -- call commit() with your first subgoal so you "
                "can tell later what you have already finished)"
            )
        else:
            width = max(len(s.status) for s in self.subgoals)
            for index, subgoal in enumerate(self.subgoals, start=1):
                line = f"{index}. [{subgoal.status:<{width}}] {subgoal.text}"
                if subgoal.status == BLOCKED and subgoal.detail:
                    line += f"\n     blocked by: {subgoal.detail}"
                lines.append(line)
        lines.append("=== END PROGRESS ===")
        return "\n".join(lines)
