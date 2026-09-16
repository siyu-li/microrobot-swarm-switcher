"""
Classical best-first plan-to-goal baselines: A*, LevinTS, PHS_h, PHS*.

Canonical BFS (Best-First Search) after Orseau & Lelis, "Policy-Guided
Heuristic Search with Guarantees" (AAAI 2021), Algorithm 1: a priority queue
ordered by an evaluation function φ, popping the minimum, returning the first
*popped* solution node, and otherwise expanding — generating and evaluating
every child.  All four baselines are instances of one engine differing only
in φ:

* **A***       φ(n) = g(n) + h(n)
* **LevinTS**  φ(n) = g(n) / π(n)                  (PHS with η ≡ 1)
* **PHS_h**    φ(n) = (g(n) + h(n)) / π(n)         (η_h = (g+h)/g)
* **PHS***     φ(n) = (g(n) + h(n)) / π(n)^(1 + h(n)/g(n))

where π(n) is the product of the prior's conditional probabilities along the
root path and h is the leaf cost-to-go (``model.cost_to_go`` — the learned
value net when configured, summed over unreached robots).  The π-based φ's
are compared in log space (π underflows within ~10 decisions).

**Loss unit.**  The paper's framework allows arbitrary non-negative losses
ℓ(n); its experiments use ℓ ≡ 1 (g = depth, search loss = expansions).  Here
ℓ = the switcher's exact per-decision step cost, so g is accumulated plan
cost in the same units as the value net's h — the PHS heuristic factor
η = (g+h)/g requires the two to share units, and the paper's headline metric
(episode cost) is what g then optimises.  ``levints-depth`` keeps the
canonical unit-loss LevinTS (φ = d₀/π) as a reference.

**Domain mapping** (matching the lazy-tree semantics of ``tree.py``):
a shield-refuted coarse edge is an illegal action discovered on generation —
skipped, with the vet still charged to the model's transition counters; a
predicted-collision child (possible only for precise, which is never vetted)
is a dead end and never enqueued; there is no duplicate-state detection
(continuous states — and the paper's analysis assumes ``state(n) = n``).
π is the softmax over all branch *stubs*, before legality is known: mass on
refuted edges is lost, which the paper explicitly permits (Σ π(n'|n) ≤ 1).

**Plan-to-goal, not receding horizon.**  ``BestFirstSearch14.run`` searches
from the root state to an in-model ``all_reached`` node and returns the whole
decision path.  Every decision carries the sub-step controls recorded when
its edge was materialised (coarse: the vetted frames; precise: the rollout's
executed actions), so ``PlanToGoalSwitcher14`` replays the plan through the
env **verbatim** — one decision per ``decide`` call, no GAT forward at
execution time — and re-plans from the live state only when the plan is
exhausted before the episode ends.  A
transition cap bounds each planning call; on cap-hit the best generated goal
node (if any — still an exact in-model plan) or the best partial path by
g + h is returned and the shortfall is flagged.  Search effort is reported in
*materialised edges*: the model's own ``n_coarse_vets`` /
``n_precise_expansions`` counters.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from robot_nav.models.MARL.capswitcher.rl.cost import SwitcherCost
from robot_nav.models.MARL.capswitcher.rl.reward import PRECISE
from robot_nav.models.MARL.capswitcher_14.rl.forward_model import (
    ForwardModel14,
    build_forward_model,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.common import expand_stubs
from robot_nav.models.MARL.capswitcher_14.rl.search.priors import UniformPrior


# ---------------------------------------------------------------------------
# Evaluation functions φ (log space where π is involved)
# ---------------------------------------------------------------------------

def _phi_astar(g: float, h: float, log_pi: float, depth: int) -> float:
    return g + h


def _phi_levints(g: float, h: float, log_pi: float, depth: int) -> float:
    return math.log(g) - log_pi


def _phi_levints_depth(g: float, h: float, log_pi: float, depth: int) -> float:
    # Canonical unit-loss LevinTS: φ = d0(n)/π(n) with d0 = depth + 1.
    return math.log(depth + 1) - log_pi


def _phi_phs(g: float, h: float, log_pi: float, depth: int) -> float:
    return math.log(g + h) - log_pi


def _phi_phs_star(g: float, h: float, log_pi: float, depth: int) -> float:
    return math.log(g + h) - (1.0 + h / g) * log_pi


# Registry: CLI algo name -> (evaluation function, uses_heuristic).
EVALUATIONS = {
    "astar": _phi_astar,
    "levints": _phi_levints,
    "levints-depth": _phi_levints_depth,
    "phs": _phi_phs,
    "phs-star": _phi_phs_star,
}


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    return z - math.log(float(np.sum(np.exp(z))))


# ---------------------------------------------------------------------------
# Search engine
# ---------------------------------------------------------------------------

class BFSNode:
    """One generated node: state + path bookkeeping (parent chain = the plan)."""

    __slots__ = ("ms", "g", "h", "log_pi", "depth", "parent", "branch",
                 "terminal", "aidx")

    def __init__(self, ms, g, h, log_pi, depth, parent, branch, terminal,
                 aidx: int = -1) -> None:
        self.ms = ms
        self.g = g                    # exact accumulated step cost from root
        self.h = h                    # leaf cost-to-go estimate (0 if terminal)
        self.log_pi = log_pi          # log Π π(a|·) along the root path
        self.depth = depth
        self.parent = parent          # BFSNode | None
        self.branch = branch          # Branch taken from parent (None at root)
        self.terminal = terminal      # all_reached in-model
        self.aidx = aidx              # branch index within parent's stubs


@dataclass
class PlanResult:
    """Outcome of one plan-to-goal search."""

    decisions: list[dict]             # per decision: mode / group / frames
    solved: bool                      # decisions end at an in-model goal node
    plan_cost: float                  # exact in-model cost of the plan (g)
    expansions: int                   # nodes expanded (incl. root)
    n_coarse_vets: int                # materialised coarse edges (incl. refuted)
    n_precise_expansions: int         # materialised precise edges
    cap_hit: bool                     # transition cap stopped the search


def _extract_decisions(node: BFSNode) -> list[dict]:
    """Root→node decision list off the parent chain (vetted frames included).

    ``next_poses`` is the search's predicted post-decision pose block — the
    drift monitor compares it against the live state during replay.
    """
    out: list[dict] = []
    while node.parent is not None:
        b = node.branch
        out.append({"mode": b.mode, "group": b.group, "pgroup": b.pgroup,
                    "frames": b.frames,
                    "next_poses": np.array(node.ms.poses, copy=True)})
        node = node.parent
    out.reverse()
    return out


class BestFirstSearch14:
    """
    Canonical best-first plan-to-goal search over a :class:`ForwardModel14`.

    Args:
        evaluate:        φ(g, h, log_pi, depth) — one of :data:`EVALUATIONS`.
        prior:           ``prior(model, ms, branches) -> logits`` over stubs
                         (π; default uniform).  Only π-based φ's consult it.
        max_transitions: cap on ``model.n_transitions`` per :meth:`run`; the
                         expansion in progress is finished, so the cap can
                         overshoot by at most one node's edges (≤ 23).
        recorder:        optional :class:`~.trace.TraceRecorder` — logs every
                         expansion + plan outcome to npz shards (redesign §6).
    """

    def __init__(self, evaluate, prior=None, max_transitions: int = 20000,
                 recorder=None) -> None:
        self.evaluate = evaluate
        self.prior = prior if prior is not None else UniformPrior()
        self.max_transitions = int(max_transitions)
        self.recorder = recorder

    def run(self, model, ms, trace: list | None = None) -> PlanResult:
        """
        Search from ``ms`` to an in-model goal.  ``trace``, if given, collects
        the popped-for-expansion nodes in order (test instrumentation).
        """
        if model.all_reached(ms):
            return PlanResult([], True, 0.0, 0, model.n_coarse_vets,
                              model.n_precise_expansions, False)
        if self.recorder is not None:
            self.recorder.begin_plan(model, ms)
        root = BFSNode(
            ms, g=0.0, h=float(model.cost_to_go(ms)), log_pi=0.0, depth=0,
            parent=None, branch=None, terminal=False,
        )

        heap: list[tuple[float, int, BFSNode]] = []
        tiebreak = 0                     # FIFO among equal φ
        expansions = 0
        best_goal: BFSNode | None = None      # cheapest generated goal node
        best_partial: BFSNode = root          # min g + h over generated nodes
        cap_hit = False

        def expand(node: BFSNode) -> None:
            nonlocal tiebreak, expansions, best_goal, best_partial
            if trace is not None:
                trace.append(node)
            branches = expand_stubs(model, node.ms)
            log_pi_cond = _log_softmax(
                np.asarray(self.prior(model, node.ms, branches), dtype=np.float64)
            )
            branch_rows: list[tuple] = []       # recorder rows, stub order
            branch_states = []                  # child ModelState/None, stub order

            def rec(b, safe, clearance, child_g, child_h, terminal, collision,
                    child_ms=None):
                branch_rows.append((
                    b.mode,
                    -1 if b.group is None else b.group,
                    -1 if b.pgroup is None else b.pgroup,
                    b.step_cost, safe, clearance,
                    child_g, child_h, terminal, collision,
                ))
                branch_states.append(child_ms)

            nan = float("nan")
            for a, b in enumerate(branches):
                if b.mode == PRECISE:
                    # Record the rollout's executed controls on the stub: the
                    # plan is replayed verbatim (the CUDA GAT forward is
                    # non-deterministic, so a live re-decide at execution time
                    # could not reproduce the searched transition).
                    if b.pgroup is None:
                        child_ms, b.frames = model.precise_next(
                            node.ms, return_frames=True
                        )
                    else:
                        child_ms, b.frames = model.precise_group_next(
                            node.ms, b.pgroup, return_frames=True
                        )
                    clearance = nan
                else:
                    mv = model.coarse_move(node.ms, b.group)
                    clearance = float(mv.candidate.clearance)
                    if not mv.candidate.safe:
                        rec(b, False, clearance, nan, nan, False, False)
                        continue          # illegal edge; the vet was charged
                    b.frames = mv.candidate.frames
                    b.candidate = mv.candidate
                    child_ms = mv.next_state
                # Collision wins over goal-reached (the sim ends a colliding
                # episode as a failure): with sub-step truncation a precise
                # child's end state is the colliding state itself.
                if model.collision_pred(child_ms):
                    rec(b, True, clearance, node.g + b.step_cost, nan,
                        False, True, child_ms)
                    continue              # dead end (no d_safe margin on precise)
                terminal = model.all_reached(child_ms)
                child = BFSNode(
                    child_ms,
                    g=node.g + b.step_cost,
                    h=0.0 if terminal else float(model.cost_to_go(child_ms)),
                    log_pi=node.log_pi + float(log_pi_cond[a]),
                    depth=node.depth + 1,
                    parent=node,
                    branch=b,
                    terminal=terminal,
                    aidx=a,
                )
                rec(b, True, clearance, child.g, child.h, terminal, False,
                    child_ms)
                if terminal and (best_goal is None or child.g < best_goal.g):
                    best_goal = child
                if child.g + child.h < best_partial.g + best_partial.h:
                    best_partial = child
                key = self.evaluate(child.g, child.h, child.log_pi, child.depth)
                heapq.heappush(heap, (key, tiebreak, child))
                tiebreak += 1
            expansions += 1
            if self.recorder is not None:
                self.recorder.record_expansion(node, branch_rows, branch_states)

        def finish(result: PlanResult, goal_node: BFSNode | None) -> PlanResult:
            if self.recorder is not None:
                self.recorder.end_plan(result, goal_node)
            return result

        expand(root)                      # the root is always expanded first
        while heap:
            _, _, node = heapq.heappop(heap)
            if node.terminal:             # first popped solution node wins
                return finish(PlanResult(
                    _extract_decisions(node), True, node.g, expansions,
                    model.n_coarse_vets, model.n_precise_expansions, False,
                ), node)
            if model.n_transitions >= self.max_transitions:
                cap_hit = True
                break
            expand(node)

        # Cap hit or queue exhausted without popping a goal.  A generated goal
        # node is still an exact in-model plan — prefer it over any estimate.
        if best_goal is not None:
            return finish(PlanResult(
                _extract_decisions(best_goal), True, best_goal.g, expansions,
                model.n_coarse_vets, model.n_precise_expansions, cap_hit,
            ), best_goal)
        return finish(PlanResult(
            _extract_decisions(best_partial), False,
            best_partial.g + best_partial.h, expansions,
            model.n_coarse_vets, model.n_precise_expansions, cap_hit,
        ), best_partial)


# ---------------------------------------------------------------------------
# Env-facing switcher: plan once, replay, re-plan on exhaustion
# ---------------------------------------------------------------------------

class PlanToGoalSwitcher14:
    """
    Drop-in switcher (same ``decide(robot_state) -> dict`` contract as
    ``GumbelSwitcher14``) that plans to the goal and replays the plan.

    A fresh plan is computed on the first decision of an episode and whenever
    the previous plan runs out before the sim episode ends.  If a planning
    call yields an empty plan (cap hit with the root as best node), the
    always-legal precise action is executed as a fallback so the episode can
    progress (``n_fallbacks`` counts these).

    Cumulative counters (``total_*``, ``n_plans``, ``n_solved_plans``,
    ``n_cap_hits``, ``n_fallbacks``) are diffed per episode by the eval
    harness; ``decision_transitions`` gets the planning call's transition
    count on planning decisions and 0 on replay decisions.

    **Execution is verbatim**: every decision — coarse *and* precise — carries
    the sub-step controls recorded at search materialisation, and the env
    replays them exactly (``SwitcherEnv.step(frames=...)``).  No GAT forward
    runs at execution time and the executed trajectory is the searched one to
    float32-snapshot precision (~5e-7 m).  Precise rollouts collision-check
    every sub-step endpoint (the same states the sim evaluates), so a replayed
    plan cannot collide except through that ~5e-7 snapshot error on a
    razor-thin margin.  Only the empty-plan precise fallback executes live.
    """

    def __init__(
        self,
        backbone,
        coarse,
        sim,
        evaluate,
        prior=None,
        max_transitions: int = 20000,
        d_safe: float = 0.3,
        selection_interval: int = 5,
        goal_threshold: float = 0.3,
        cost: SwitcherCost | None = None,
        default_rho: float = 0.2,
        leaf_value=None,
        coupling=None,
        precise_groups: list | None = None,
        recorder=None,
        replan_drift: float | None = None,
    ) -> None:
        if cost is None:
            raise ValueError(
                "PlanToGoalSwitcher14 requires a SwitcherCost (load "
                "cost_14robots.yaml with SwitcherCost.from_yaml)"
            )
        self.coupling = coupling
        self.precise_groups = precise_groups
        self.recorder = recorder
        self.backbone = backbone
        self.coarse = coarse
        self.sim = sim
        self.d_safe = float(d_safe)
        self.selection_interval = int(selection_interval)
        self.goal_threshold = float(goal_threshold)
        self.cost = cost
        self.default_rho = float(default_rho)
        self.leaf_value = leaf_value
        self.search = BestFirstSearch14(
            evaluate, prior=prior, max_transitions=max_transitions,
            recorder=recorder,
        )

        # Drift-triggered replanning (heading-noise robustness): drop the
        # plan suffix when the live poses deviate from the search's predicted
        # post-decision poses by more than ``replan_drift`` metres.  None =
        # legacy open-loop replay (replan only on plan exhaustion).
        self.replan_drift = (None if replan_drift is None
                             else float(replan_drift))
        self._expected_poses: np.ndarray | None = None

        self._plan: deque[dict] = deque()
        self.decision_transitions: list[int] = []
        self.total_coarse_vets = 0
        self.total_precise_expansions = 0
        self.total_expansions = 0
        self.n_plans = 0
        self.n_solved_plans = 0
        self.n_cap_hits = 0
        self.n_fallbacks = 0
        self.n_drift_replans = 0

    def _build(self, robot_state: np.ndarray) -> tuple[ForwardModel14, object]:
        model = build_forward_model(
            self.backbone,
            self.coarse,
            self.sim,
            robot_state,
            d_safe=self.d_safe,
            selection_interval=self.selection_interval,
            goal_threshold=self.goal_threshold,
            cost=self.cost,
            default_rho=self.default_rho,
            leaf_value=self.leaf_value,
            coupling=self.coupling,
            precise_groups=self.precise_groups,
        )
        return model, ForwardModel14.state_from_robot_state(robot_state)

    def decide(self, robot_state: np.ndarray) -> dict:
        if (self._plan and self.replan_drift is not None
                and self._expected_poses is not None):
            poses = ForwardModel14.state_from_robot_state(robot_state).poses
            err = float(np.max(np.linalg.norm(
                poses[:, :2] - self._expected_poses[:, :2], axis=1)))
            if err > self.replan_drift:
                self._plan.clear()          # reality left the plan — replan
                self.n_drift_replans += 1
        if not self._plan:
            model, ms = self._build(robot_state)
            res = self.search.run(model, ms)
            self.n_plans += 1
            self.n_solved_plans += int(res.solved)
            self.n_cap_hits += int(res.cap_hit)
            self.total_coarse_vets += res.n_coarse_vets
            self.total_precise_expansions += res.n_precise_expansions
            self.total_expansions += res.expansions
            self.decision_transitions.append(model.n_transitions)
            self._plan = deque(res.decisions)
            if not self._plan:
                self.n_fallbacks += 1
                return {
                    "mode": PRECISE, "group": None, "pgroup": None,
                    "frames": None, "candidates": [],
                }
        else:
            self.decision_transitions.append(0)
        d = self._plan.popleft()
        self._expected_poses = d.get("next_poses")
        return {
            "mode": d["mode"], "group": d["group"],
            "pgroup": d.get("pgroup"), "frames": d["frames"],
            "candidates": [],
        }

    def reset_plan(self) -> None:
        """Drop any unexecuted plan suffix (call at episode boundaries)."""
        self._plan.clear()
        self._expected_poses = None

    def snapshot(self) -> dict:
        """Cumulative counters — diffed per episode by the eval harness."""
        return {
            "coarse_vets": self.total_coarse_vets,
            "precise_expansions": self.total_precise_expansions,
            "expansions": self.total_expansions,
            "plans": self.n_plans,
            "solved_plans": self.n_solved_plans,
            "cap_hits": self.n_cap_hits,
            "fallbacks": self.n_fallbacks,
            "drift_replans": self.n_drift_replans,
        }
