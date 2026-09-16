"""
Analytic forward model with **single-edge transitions** (no ``sim.step`` in the
tree) — the 14-robot counterpart of ``capswitcher.rl.forward_model``.

The decisive difference from the 6-robot model: there is no ``coarse_moves``
that vets *every* group at a node.  The lazy search materialises one edge at a
time, so the model's unit of work is :meth:`coarse_move` — vet **one** group
(frames + swept clearance + progress) and, if the shield admits it, build its
deterministic next state.  Verifying a coarse action and computing its
transition are the same computation, so verify-on-descent costs nothing beyond
the transition the search must pay anyway.

Budget accounting
-----------------
``n_transitions = n_coarse_vets + n_precise_expansions`` counts every expensive
model operation: each single-group vet (including ones the shield refutes) and
each precise-all rollout.  This is the search budget unit — a per-node "expand
everything" scheme would be charged its true cost of ~23 transitions here.

Determinism
-----------
``CoarseSteering14`` never drops actuation columns, so a coarse control is a
pure function of (state, group) — the seed threading of the 6-robot model
(``_state_group_seed``) is gone entirely.

Reused from ``capswitcher`` (N-generic): shield sweep geometry
(``swept_positions`` / ``min_member_clearance`` / ``predicted_progress`` /
``CoarseCandidate`` / ``ShieldGeometry``) and the ``SwitcherCost`` table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_nav.models.MARL.capswitcher.rl.cost import SwitcherCost
from robot_nav.models.MARL.capswitcher.rl.reward import COARSE
from robot_nav.models.MARL.capswitcher.rl.shield import (
    CoarseCandidate,
    ShieldGeometry,
    min_member_clearance,
    predicted_progress,
    swept_positions,
)


def _angle_wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap angles into (−π, π]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def analytic_alpha(precise_unit: float, lin_max: float, step_time: float) -> float:
    """
    Cost per metre per robot of the precise-completion heuristic.

    ``precise_unit`` is charged per robot per sub-step, and one sub-step
    advances a robot ``lin_max · step_time`` metres.
    """
    return float(precise_unit) / (float(lin_max) * float(step_time))


def analytic_cost_to_go(
    goal_distances: np.ndarray, goal_threshold: float, alpha: float
) -> float:
    """
    The analytic leaf heuristic ``α · Σ_{i unreached} ‖p_i − goal_i‖``.

    Reached robots contribute 0 — they are skipped by the precise rollout and
    never charged.

    Module-level so anything measuring the heuristic's bias (see
    ``robot_nav/collect_leaf_data.py``) scores the *same* function the search
    plans with, rather than a re-derivation that can drift from it.

    Note the pricing this bakes in: it charges the whole completion at the
    **precise** rate, so it is blind to coarse moves being 4–60× cheaper per
    robot-metre.  It is therefore expected to overestimate — the direction the
    Gumbel searches do not self-correct (see ``search/gumbel_eager.py``).
    Quantifying that gap is what the leaf-data collection exists to do.
    """
    d = np.asarray(goal_distances, dtype=np.float64)
    return float(alpha) * float(np.sum(d[d > goal_threshold]))


@dataclass
class ModelState:
    """Dynamic planner state for one lookahead node."""

    poses: np.ndarray         # (N, 3) [x, y, theta]
    last_actions: np.ndarray  # (N, 2) last applied sim-input [lin, ang]


@dataclass
class CoarseMove:
    """One vetted coarse edge: shield candidate + (if safe) the state it leads to."""

    candidate: CoarseCandidate       # group, frames, clearance, progress, safe
    next_state: ModelState | None    # None iff the shield refuted the move


class ForwardModel14:
    """
    Deterministic analytic model over robot poses for the lazy tree search.

    Args:
        backbone:           Frozen ``GATBackbone`` (precise actions + state prep).
        coarse:             ``CoarseSteering14`` primitive (shared with the env).
        goals:              (N, 2) fixed goal positions for the episode.
        obstacle_states:    (M, 4) obstacle observations for the GAT forward
                            (frozen within a decision).
        geom:               ``ShieldGeometry`` for the safety vet and collision
                            prediction.
        step_time:          Simulator sub-step duration (s).
        selection_interval: Sub-steps each robot is driven per precise decision.
        lin_max:            Max linear velocity (m/s) — used by the cost-to-go α.
        d_safe:             Clearance margin (m) a coarse move must keep to be safe.
        goal_threshold:     Per-robot goal-arrival radius (m) for ``all_reached``.
        cost:               :class:`SwitcherCost` decision-pricing table (load
                            with ``SwitcherCost.from_yaml``).
        leaf_value:         Optional learned leaf evaluator ``(model, ms) -> float``.
        coupling:           Optional rotation realiser — the physics fix
                            (redesign §2): precise rotation is realised through
                            the actuation matrix.  Either ``PreciseCoupling``
                            (minimum-norm ``pinv(A_S)``; every bystander
                            side-rotates) or ``GroupRotation`` (the driven
                            robot's fixed size-7 block rotates uniformly with
                            it; the other block holds still).  Both expose
                            ``coupled_ang(members, omega)``.  ``None`` keeps
                            the legacy (physically inconsistent)
                            independent-rotation model.
        precise_groups:     Optional list of member-index lists — the precise
                            action set (redesign §3).  ``None`` keeps the single
                            legacy precise-all edge; a list replaces it with one
                            edge per group (see :meth:`precise_group_next`).
    """

    def __init__(
        self,
        backbone,
        coarse,
        goals: np.ndarray,
        obstacle_states: np.ndarray,
        geom: ShieldGeometry,
        step_time: float,
        selection_interval: int = 5,
        lin_max: float = 0.5,
        d_safe: float = 0.3,
        goal_threshold: float = 0.3,
        cost: SwitcherCost | None = None,
        leaf_value=None,
        coupling=None,
        precise_groups: list | None = None,
    ) -> None:
        if cost is None:
            raise ValueError(
                "ForwardModel14 requires a SwitcherCost (load the system's "
                "cost YAML with SwitcherCost.from_yaml)"
            )
        self.backbone = backbone
        self.coarse = coarse
        self.goals = np.asarray(goals, dtype=np.float64)          # (N, 2)
        self.obstacle_states = np.asarray(obstacle_states)         # (M, 4)
        self.geom = geom
        self.step_time = float(step_time)
        self.selection_interval = int(selection_interval)
        self.lin_max = float(lin_max)
        self.d_safe = float(d_safe)
        self.goal_threshold = float(goal_threshold)
        self.cost = cost
        self.leaf_value = leaf_value
        self.coupling = coupling
        self.precise_groups = (
            None if precise_groups is None
            else [np.asarray(g, dtype=int) for g in precise_groups]
        )

        self.N = self.goals.shape[0]

        # Budget counters: every single-group vet and every precise rollout.
        self.n_coarse_vets = 0
        self.n_precise_expansions = 0

    @property
    def n_transitions(self) -> int:
        """Total expensive model operations — the search budget unit."""
        return self.n_coarse_vets + self.n_precise_expansions

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    @staticmethod
    def state_from_robot_state(robot_state: np.ndarray) -> ModelState:
        """
        Recover a :class:`ModelState` from an 11-col ``robot_state`` row block
        (inverse of the GAT node-feature scalings — see the 6-robot model).
        """
        s = np.asarray(robot_state, dtype=np.float64)
        px, py = s[:, 0], s[:, 1]
        theta = np.arctan2(s[:, 3], s[:, 2])
        poses = np.stack([px, py, theta], axis=1)                  # (N, 3)
        lin = s[:, 7] / 2.0
        ang = s[:, 8] * 2.0 - 1.0
        last_actions = np.stack([lin, ang], axis=1)                # (N, 2)
        return ModelState(poses=poses, last_actions=last_actions)

    def robot_state(self, ms: ModelState) -> np.ndarray:
        """
        Build the (N, 11) GAT state for ``ms`` — mirrors ``sim.step`` geometry
        then ``backbone.prepare_state`` so layout/scaling match training.
        """
        pos = ms.poses[:, :2]
        theta = ms.poses[:, 2]
        goal_vecs = self.goals - pos                               # (N, 2)
        dist = np.linalg.norm(goal_vecs, axis=1)                   # (N,)
        dist_safe = np.where(dist > 1e-10, dist, 1e-10)
        h = np.stack([np.cos(theta), np.sin(theta)], axis=1)       # (N, 2) unit
        g = goal_vecs / dist_safe[:, None]
        cos_e = np.sum(h * g, axis=1)                              # (N,)
        sin_e = h[:, 0] * g[:, 1] - h[:, 1] * g[:, 0]             # (N,)

        states, _ = self.backbone.prepare_state(
            ms.poses.tolist(),
            dist.tolist(),
            cos_e.tolist(),
            sin_e.tolist(),
            [False] * self.N,
            ms.last_actions.tolist(),
            self.goals.tolist(),
        )
        return np.asarray(states, dtype=np.float32)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def goal_distances(self, ms: ModelState) -> np.ndarray:
        """Per-robot Euclidean distance to goal."""
        return np.linalg.norm(self.goals - ms.poses[:, :2], axis=1)

    def all_reached(self, ms: ModelState) -> bool:
        """True iff every robot is within ``goal_threshold`` of its goal."""
        return bool(np.all(self.goal_distances(ms) <= self.goal_threshold))

    def collision_pred(self, ms: ModelState) -> bool:
        """
        Geometric collision prediction for a node's state (robot–robot and
        robot–obstacle, same geometry as the shield).
        """
        pos = ms.poses[:, :2]
        rho = self.geom.rho
        diff = pos[:, None, :] - pos[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        if d.min() < 2.0 * rho:
            return True
        if self.geom.obstacle_xy.shape[0] > 0:
            od = np.linalg.norm(
                pos[:, None, :] - self.geom.obstacle_xy[None, :, :], axis=2
            )
            clr = od - self.geom.obstacle_r[None, :] - rho
            if clr.min() < 0.0:
                return True
        return False

    def _substep_collision(self, poses: np.ndarray, driven: np.ndarray) -> bool:
        """
        :meth:`collision_pred` restricted to the robots that moved this
        sub-step.  Bystanders never translate inside a precise rollout, so a
        pair not involving a driven robot cannot have changed since the entry
        state (which the search already knows is collision-free) — checking
        driven-vs-all covers every pair the sub-step could have broken.
        """
        pos = poses[:, :2]
        dpos = pos[driven]
        d = np.linalg.norm(dpos[:, None, :] - pos[None, :, :], axis=2)
        d[np.arange(len(driven)), driven] = np.inf
        if d.min() < 2.0 * self.geom.rho:
            return True
        if self.geom.obstacle_xy.shape[0] > 0:
            od = np.linalg.norm(
                dpos[:, None, :] - self.geom.obstacle_xy[None, :, :], axis=2
            )
            if (od - self.geom.obstacle_r[None, :] - self.geom.rho).min() < 0.0:
                return True
        return False

    # ------------------------------------------------------------------
    # Transitions (single-edge — the budgeted operations)
    # ------------------------------------------------------------------

    def coarse_move(self, ms: ModelState, group: int) -> CoarseMove:
        """
        Vet **one** coarse group at ``ms`` and, if the shield admits it, build
        its deterministic next state.

        This is a budgeted operation (``n_coarse_vets``), charged whether or not
        the vet succeeds — a refuted vet consumed the same sweep computation.
        A refuted move returns ``next_state=None``; the caller prunes the edge
        from the node's legal action set (no cost, no penalty — the action is
        simply not available there).
        """
        self.n_coarse_vets += 1
        rs = self.robot_state(ms)
        rot, trans = self.coarse.compute_actions(rs, group)
        samples = swept_positions(rs, rot, trans, self.step_time)  # (T+1, N, 2)
        members = self.coarse.members_of(group)
        clearance = min_member_clearance(samples, members, self.geom)
        progress = predicted_progress(rs, members, samples[-1])
        safe = clearance >= self.d_safe
        candidate = CoarseCandidate(
            group=group,
            frames=rot + trans,
            clearance=clearance,
            progress=progress,
            safe=safe,
        )
        next_state = (
            self._coarse_next_state(ms, rot, trans, samples[-1]) if safe else None
        )
        return CoarseMove(candidate=candidate, next_state=next_state)

    def _coarse_next_state(
        self,
        ms: ModelState,
        rotation_frames: list,
        translation_frames: list,
        end_positions: np.ndarray,
    ) -> ModelState:
        """
        Next state after a coarse control: swept end positions + post-rotation
        headings; ``last_actions`` = the final executed frame.
        """
        theta = ms.poses[:, 2].copy()
        for frame in rotation_frames:
            theta = theta + np.asarray(frame, dtype=np.float64)[:, 1] * self.step_time
        theta = _angle_wrap(theta)

        poses = np.empty_like(ms.poses)
        poses[:, :2] = end_positions
        poses[:, 2] = theta

        all_frames = rotation_frames + translation_frames
        last = (
            np.asarray(all_frames[-1], dtype=np.float64)
            if all_frames
            else np.zeros((self.N, 2))
        )
        return ModelState(poses=poses, last_actions=last)

    def precise_next(self, ms: ModelState, return_frames: bool = False):
        """
        Next state after one precise-all decision: resolve robots one at a time,
        each driven by its frozen GAT action for ``selection_interval`` sub-steps
        while the others hold still, integrating unicycle dynamics.  The rollout
        truncates at the first colliding sub-step, so a mid-decision clip
        surfaces as a colliding end state (caught by the search's endpoint
        ``collision_pred``) instead of being integrated through.

        This is the budgeted transition (``n_precise_expansions``); the
        unbudgeted :meth:`precise_rollout` shares its body.

        With ``return_frames=True`` returns ``(next_state, frames)``, the
        frames being the executed sub-step controls of **this exact rollout**
        (the CUDA GAT forward is non-deterministic, so a re-roll would not
        reproduce it) — replayable verbatim by ``SwitcherEnv.step(1,
        frames=...)``.
        """
        self.n_precise_expansions += 1
        st, _, frames = self._precise_rollout(
            ms, record=False, record_frames=return_frames
        )
        return (st, frames) if return_frames else st

    def precise_rollout(self, ms: ModelState) -> tuple[ModelState, list]:
        """
        Same transition as :meth:`precise_next`, additionally returning the
        swept path — **not** charged to the transition budget (diagnostics only).

        Precise is not shield-vetted (no ``d_safe`` margin), but every sub-step
        endpoint is collision-checked and the rollout truncates at the first
        hit — exactly the states the sim evaluates during verbatim replay, so
        an in-model-clean rollout cannot collide in the sim.  The path exposes
        the sub-steps for offline scoring with the shield's swept-clearance
        criterion (``robot_nav/eval_feasibility_14.py``).

        Returns:
            ``(next_state, path)`` where ``path`` is a list of
            ``(driven_robot, positions)`` pairs — one per sub-step, positions
            an ``(N, 2)`` array — led by the entry state as ``(-1, positions)``.
            ``driven_robot`` is the only robot that moved on that sub-step.
        """
        st, path, _ = self._precise_rollout(ms, record=True)
        return st, path

    def precise_frames(self, ms: ModelState) -> tuple[ModelState, list]:
        """
        Same transition as :meth:`precise_next`, additionally returning the
        executed sub-step controls — **not** charged to the transition budget.

        The frames let ``SwitcherEnv.step(1, frames=...)`` replay this exact
        rollout verbatim (the sim's integrator matches :meth:`_integrate`
        bit-for-bit), instead of re-deciding skip sets and actions live —
        removing model→sim replay divergence entirely.

        Returns:
            ``(next_state, frames)`` — ``frames`` is one ``(driven, actions)``
            pair per sub-step: ``driven`` the list of robots commanded this
            sub-step, ``actions`` the full (N, 2) [lin, ang] sim input as
            nested lists.
        """
        st, _, frames = self._precise_rollout(
            ms, record=False, record_frames=True
        )
        return st, frames

    def _precise_rollout(
        self, ms: ModelState, record: bool, record_frames: bool = False
    ) -> tuple[ModelState, list, list]:
        """Shared body of ``precise_next`` / ``precise_rollout`` / ``precise_frames``."""
        poses = ms.poses.copy()
        last = ms.last_actions.copy()
        path: list = [(-1, poses[:, :2].copy())] if record else []
        frames: list = []
        # Robots already at goal are skipped (mirrors the env; they are also
        # not charged by the precise pricing).  Membership is frozen at entry —
        # a robot arriving mid-decision still finishes its own sub-steps.
        unreached = np.flatnonzero(self.goal_distances(ms) > self.goal_threshold)
        for r in unreached:
            for _ in range(self.selection_interval):
                rs = self.robot_state(ModelState(poses=poses, last_actions=last))
                raw, _ = self.backbone.get_embedding_and_actions(
                    rs, self.obstacle_states
                )
                lin = (float(raw[r, 0]) + 1.0) / 4.0   # actor→sim: [0, 0.5]
                ang = float(raw[r, 1])
                sim_actions = np.zeros((self.N, 2), dtype=np.float64)
                if self.coupling is not None:
                    # Physics fix: the driven robot's turn couples into every
                    # bystander through the actuation matrix (bystanders still
                    # do not translate).
                    sim_actions[:, 1] = self.coupling.coupled_ang([int(r)], [ang])
                else:
                    sim_actions[r, 1] = ang
                sim_actions[r, 0] = lin
                poses = self._integrate(poses, sim_actions)
                last = sim_actions
                if record:
                    path.append((int(r), poses[:, :2].copy()))
                if record_frames:
                    frames.append(([int(r)], sim_actions.tolist()))
                # The sim evaluates collision at every sub-step state; a
                # rollout that clips a robot/obstacle mid-decision truncates
                # here, so the returned end state *is* the colliding state and
                # the search's endpoint collision_pred flags the branch.
                if self._substep_collision(poses, np.array([int(r)])):
                    return ModelState(poses=poses, last_actions=last), path, frames
        return ModelState(poses=poses, last_actions=last), path, frames

    # ------------------------------------------------------------------
    # Per-group precise transitions (redesign §3 — configs B/C)
    # ------------------------------------------------------------------

    def precise_group_next(
        self, ms: ModelState, pgroup: int, return_frames: bool = False
    ):
        """
        Next state after one precise decision driving precise-group ``pgroup``:
        the group's *unreached* members are driven **simultaneously** by their
        frozen GAT actions for ``selection_interval`` sub-steps.  With a
        configured ``coupling`` the members' angular commands are realised
        through the actuation matrix (bystanders side-rotate but do not
        translate); without one, bystanders hold still (legacy physics).

        A budgeted transition (``n_precise_expansions``), one per edge — note
        it costs ``|driven| × selection_interval`` GAT-driven sub-steps versus
        ``n_unreached × selection_interval`` for the legacy precise-all edge.

        ``return_frames=True`` → ``(next_state, frames)``; see
        :meth:`precise_next`.
        """
        self.n_precise_expansions += 1
        st, _, frames = self._precise_group_rollout(
            ms, pgroup, record=False, record_frames=return_frames
        )
        return (st, frames) if return_frames else st

    def precise_group_rollout(
        self, ms: ModelState, pgroup: int
    ) -> tuple[ModelState, list]:
        """Un-budgeted variant returning the swept path (diagnostics only)."""
        st, path, _ = self._precise_group_rollout(ms, pgroup, record=True)
        return st, path

    def precise_group_frames(
        self, ms: ModelState, pgroup: int
    ) -> tuple[ModelState, list]:
        """
        Un-budgeted variant returning the executed sub-step controls — the
        per-group counterpart of :meth:`precise_frames`, same frame format.
        """
        st, _, frames = self._precise_group_rollout(
            ms, pgroup, record=False, record_frames=True
        )
        return st, frames

    def driven_members(self, ms: ModelState, pgroup: int) -> np.ndarray:
        """Unreached members of precise-group ``pgroup`` at ``ms``."""
        members = self.precise_groups[pgroup]
        dist = self.goal_distances(ms)
        return members[dist[members] > self.goal_threshold]

    def _precise_group_rollout(
        self, ms: ModelState, pgroup: int, record: bool,
        record_frames: bool = False,
    ) -> tuple[ModelState, list, list]:
        """Shared body of ``precise_group_next`` / ``_rollout`` / ``_frames``."""
        if self.precise_groups is None:
            raise ValueError("precise_group_next requires precise_groups")
        poses = ms.poses.copy()
        last = ms.last_actions.copy()
        path: list = [(-1, poses[:, :2].copy())] if record else []
        frames: list = []
        driven = self.driven_members(ms, pgroup)
        if driven.size == 0:          # all members reached — a no-op edge
            return ModelState(poses=poses, last_actions=last), path, frames
        driven_list = [int(r) for r in driven]
        for _ in range(self.selection_interval):
            rs = self.robot_state(ModelState(poses=poses, last_actions=last))
            raw, _ = self.backbone.get_embedding_and_actions(
                rs, self.obstacle_states
            )
            raw = np.asarray(raw)
            lin = (raw[driven, 0] + 1.0) / 4.0        # actor→sim: [0, 0.5]
            ang = raw[driven, 1].astype(np.float64)
            sim_actions = np.zeros((self.N, 2), dtype=np.float64)
            if self.coupling is not None:
                sim_actions[:, 1] = self.coupling.coupled_ang(driven_list, ang)
            else:
                sim_actions[driven, 1] = ang
            sim_actions[driven, 0] = lin
            poses = self._integrate(poses, sim_actions)
            last = sim_actions
            if record:
                path.append((driven_list, poses[:, :2].copy()))
            if record_frames:
                frames.append((driven_list, sim_actions.tolist()))
            # Truncate at the first colliding sub-step (see _precise_rollout);
            # driven-vs-all also covers driven-vs-driven pairs.
            if self._substep_collision(poses, driven):
                break
        return ModelState(poses=poses, last_actions=last), path, frames

    def _integrate(self, poses: np.ndarray, sim_actions: np.ndarray) -> np.ndarray:
        """One unicycle sub-step for all robots (forward Euler, pre-update heading)."""
        dt = self.step_time
        v = sim_actions[:, 0]
        w = sim_actions[:, 1]
        theta = poses[:, 2]
        out = poses.copy()
        out[:, 0] = poses[:, 0] + v * np.cos(theta) * dt
        out[:, 1] = poses[:, 1] + v * np.sin(theta) * dt
        out[:, 2] = _angle_wrap(theta + w * dt)
        return out

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------

    def n_unreached(self, ms: ModelState) -> int:
        """Number of robots still outside ``goal_threshold`` at ``ms``."""
        return int(np.sum(self.goal_distances(ms) > self.goal_threshold))

    def step_cost(
        self,
        action: int,
        ms: ModelState,
        group: int | None = None,
        pgroup: int | None = None,
    ) -> float:
        """
        Per-decision motion cost from the :class:`SwitcherCost` table.

        Coarse ⇒ the group's configured constant.  Precise-all ⇒
        ``precise_unit × n_unreached(ms) × selection_interval`` — the nominal
        price of the rollout that skips reached robots (the env charges the
        sub-steps actually executed; they differ only on terminal truncation).
        Precise-group (``pgroup``) ⇒ the same formula over the group's
        unreached members only.  Known without vetting — lazy branch stubs
        carry exact step costs from creation.
        """
        if action == COARSE:
            return float(self.cost.coarse_cost(group))
        if pgroup is not None:
            n_driven = int(self.driven_members(ms, pgroup).size)
            return float(
                self.cost.precise_cost(n_driven, self.selection_interval)
            )
        return float(
            self.cost.precise_cost(self.n_unreached(ms), self.selection_interval)
        )

    def cost_to_go(self, ms: ModelState, alpha: float | None = None) -> float:
        """
        Leaf cost-to-go: the configured ``leaf_value`` if set, else the analytic
        precise-completion heuristic ``α · Σ_{i unreached} ‖p_i − goal_i‖`` with
        default ``α = precise_unit / (lin_max · step_time)`` — ``precise_unit``
        charged per robot per sub-step, one sub-step advancing a robot
        ``lin_max · step_time`` metres.  Reached robots contribute 0 (they are
        skipped and never charged).
        """
        if self.leaf_value is not None:
            return float(self.leaf_value(self, ms))
        if alpha is None:
            alpha = self.analytic_alpha()
        return analytic_cost_to_go(
            self.goal_distances(ms), self.goal_threshold, alpha
        )

    def analytic_alpha(self) -> float:
        """This model's ``α`` for :func:`analytic_cost_to_go`."""
        return analytic_alpha(
            self.cost.precise_unit, self.lin_max, self.step_time
        )


# robot_state goal columns (same layout as the 6-robot system).
_GX, _GY = 9, 10


def build_forward_model(
    backbone,
    coarse,
    sim,
    robot_state: np.ndarray,
    *,
    d_safe: float,
    selection_interval: int,
    goal_threshold: float,
    cost: SwitcherCost,
    default_rho: float,
    leaf_value=None,
    coupling=None,
    precise_groups: list | None = None,
) -> ForwardModel14:
    """
    Rebuild the deterministic forward model from the live sim + root state —
    the one place a switcher turns (backbone, coarse, sim, root robot_state)
    into a :class:`ForwardModel14`.
    """
    s = np.asarray(robot_state, dtype=np.float64)
    goals = s[:, [_GX, _GY]]
    return ForwardModel14(
        backbone=backbone,
        coarse=coarse,
        goals=goals,
        obstacle_states=sim.get_obstacle_states(),
        geom=ShieldGeometry.from_sim(sim, default_rho=default_rho),
        step_time=coarse.step_time,
        selection_interval=selection_interval,
        lin_max=coarse.lin_max,
        d_safe=d_safe,
        goal_threshold=goal_threshold,
        cost=cost,
        leaf_value=leaf_value,
        coupling=coupling,
        precise_groups=precise_groups,
    )
