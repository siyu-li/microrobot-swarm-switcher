"""
CoarseSteering: coarse group steering over a rank-deficient actuation matrix.

A single coarse control of group ``g`` has two physical parts:

1. **Rotation.** All robots rotate according to the actuation matrix ``A``
   (the complement of the membership matrix ``M``).  Because ``A`` is
   rank-deficient, the reachable rotation subspace is low-dimensional.  We pick
   rotation parameters ``t`` either by

       least-squares heading:   t* = pinv(A_reduced) @ dθ_desired
       nonlinear progress:      t* = argmax  Σ_members  Δdistance_to_goal(t)

   and apply ``dθ_actual = A_reduced @ t*`` to every robot.

2. **Move.** Only the *members* of the chosen group (``M[:, g] == 1``) advance
   forward by a fixed **translation length** ``move_distance`` (metres); non-members
   hold still.

Two linear quantities — keep them distinct
------------------------------------------
``move_distance`` (metres) is the coarse translation length: how far the group's
members travel per coarse control.  ``lin_max`` (m/s) is the simulator's fixed
per-step linear-velocity bound on the action command.  They relate by
``distance = velocity · step_time``, so a single env step can realise at most
``lin_max · step_time`` metres.  When ``move_distance`` exceeds that, the move is
split into ``n_trans = ceil(move_distance / (lin_max · step_time))`` equal
sub-steps — the exact mirror of how a large rotation ``dθ`` is split into
``n_rot`` sub-steps under the ``ang_max`` bound.

Group-dependent rank reduction
-------------------------------
``A_reduced`` is rebuilt every call from the chosen group:

  * drop the chosen group's own column from ``A`` (it actuates the move, not
    the steering rotation),
  * randomly drop one of the remaining three columns (the all-ones group-4
    column is eligible),

leaving exactly two columns — an artificially rank-2 actuation matrix that
mimics the rank-deficiency problem.

Physical interpretation
-----------------------
The membership matrix M (6 robots × 4 groups) defines which robots belong to
each group.  A[i, s] = 1 means robot i rotates when group s is activated (i.e.
robot i is NOT a member of group s).

Full A (6 robots × 4 groups):
    Robot 1: [1, 1, 0, 1]
    Robot 2: [1, 0, 1, 1]
    Robot 3: [1, 0, 0, 1]
    Robot 4: [0, 1, 1, 1]
    Robot 5: [0, 1, 0, 1]
    Robot 6: [0, 0, 1, 1]

Selectable move groups are 1, 2, 3 (group 4 has no members, so nothing moves).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Full actuation matrix (6 robots × 4 groups).
# Complement of the membership matrix M from the project summary:
#     A[i, s] = 1 - M[s, i]
# ---------------------------------------------------------------------------
_A_FULL_6 = np.array(
    [
        [1, 1, 0, 1],  # Robot 0
        [1, 0, 1, 1],  # Robot 1
        [1, 0, 0, 1],  # Robot 2
        [0, 1, 1, 1],  # Robot 3
        [0, 1, 0, 1],  # Robot 4
        [0, 0, 1, 1],  # Robot 5
    ],
    dtype=np.float64,
)

_NUM_GROUPS = 4
_RANK = 2  # reduced actuation rank we artificially constrain to


def _angle_wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap angles into (−π, π]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class CoarseSteering:
    """
    Coarse group steering for a swarm of coupled unicycle robots.

    Each call produces the sim-input action frames for one coarse control of a
    chosen group: a sequence of rotation sub-steps (so even large rotations are
    fully realised within the per-step angular-velocity bound) followed by a
    sequence of translation sub-steps (so the full ``move_distance`` is realised
    within the per-step linear-velocity bound) for the group's members.

    Args:
        num_robots:    Number of robots (must be 6 for the default matrix).
        move_distance: Coarse translation length (metres): how far the chosen
                       group's members advance per coarse control — a scalar for
                       every group, or a ``{group_id: distance}`` mapping (1-based
                       ids) covering every selectable group. Independent of the
                       simulator velocity cap — large values are split across
                       several sub-steps. Default 0.5.
        method:        ``"least_squares"`` (pinv heading) or ``"nonlinear"``
                       (maximise members' distance-to-goal reduction). Default
                       ``"least_squares"``.
        step_time:     Simulator step duration (s). Converts target rotation /
                       translation into per-step velocities. Default 0.3.
        ang_max:       Maximum angular velocity magnitude (rad/s) the simulator
                       accepts. Default 1.0.
        lin_max:       Maximum linear velocity (m/s) the simulator accepts. Used
                       only to decide how many translation sub-steps realise
                       ``move_distance``. Default 0.5.
        seed:          Optional seed for the random column drop (reproducibility).
    """

    def __init__(
        self,
        num_robots: int = 6,
        move_distance: float | dict = 0.5,
        method: str = "least_squares",
        step_time: float = 0.3,
        ang_max: float = 1.0,
        lin_max: float = 0.5,
        seed: int | None = None,
    ) -> None:
        if num_robots != 6:
            raise ValueError(
                "CoarseSteering currently only supports num_robots=6 "
                f"(got {num_robots})."
            )
        if method not in ("least_squares", "nonlinear"):
            raise ValueError(
                f"method must be 'least_squares' or 'nonlinear' (got {method!r})."
            )
        self.num_robots = num_robots
        self.method = method
        self.step_time = step_time
        self.ang_max = ang_max
        self.lin_max = lin_max
        self.rng = np.random.default_rng(seed)

        self.A_full = _A_FULL_6.copy()
        # Members of group g (0-based column): robots with A[:, g] == 0.
        self._members = [
            np.flatnonzero(self.A_full[:, g] == 0) for g in range(_NUM_GROUPS)
        ]

        # Per-group translation lengths, keyed by 1-based selectable group id.
        selectable = [g + 1 for g in range(_NUM_GROUPS) if self._members[g].size > 0]
        if isinstance(move_distance, dict):
            missing = [g for g in selectable if g not in move_distance]
            if missing:
                raise ValueError(f"move_distance mapping missing groups {missing}")
            self.move_distances = {g: float(move_distance[g]) for g in selectable}
        else:
            self.move_distances = {g: float(move_distance) for g in selectable}
        if any(d < 0.0 for d in self.move_distances.values()):
            raise ValueError("move_distance must be non-negative")

        # Wall-clock seconds spent in the most recent rotation solve (the
        # least-squares projection or the nonlinear optimisation). Used for the
        # LS-vs-nonlinear runtime comparison.
        self.last_solve_time_s: float = 0.0

    # ------------------------------------------------------------------

    def selectable_groups(self) -> list[int]:
        """Return 1-based group ids that have at least one member (movable)."""
        return [g + 1 for g in range(_NUM_GROUPS) if self._members[g].size > 0]

    def members_of(self, group: int) -> np.ndarray:
        """Robot indices that move when ``group`` (1-based) is activated."""
        return self._members[group - 1]

    # ------------------------------------------------------------------

    def _reduce_A(
        self, group: int, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """
        Build the rank-2 reduced actuation matrix for the chosen group.

        Drops the chosen group's own column, then randomly drops one of the
        remaining three columns (group-4 all-ones column included), leaving two.

        Args:
            group: 1-based chosen group id.
            rng:   Optional RNG for the random column drop.  When ``None`` the
                   instance RNG ``self.rng`` is used (the legacy behaviour);
                   passing a seeded generator makes the drop — and therefore the
                   whole coarse control — deterministic given (state, group),
                   which the MPC lookahead relies on.

        Returns:
            A_reduced: (N, 2) actuation matrix.
        """
        r = rng if rng is not None else self.rng
        g = group - 1
        remaining = [c for c in range(_NUM_GROUPS) if c != g]  # 3 columns
        drop = int(r.choice(remaining))
        keep = [c for c in remaining if c != drop]  # 2 columns
        return self.A_full[:, keep]

    def _solve_rotation(
        self,
        A_reduced: np.ndarray,
        theta_current: np.ndarray,
        d_theta_desired: np.ndarray,
        px: np.ndarray,
        py: np.ndarray,
        gx: np.ndarray,
        gy: np.ndarray,
        members: np.ndarray,
        move_distance: float,
    ) -> np.ndarray:
        """
        Solve for per-robot actual rotation ``dθ_actual = A_reduced @ t*``.

        Times the solve into ``self.last_solve_time_s``.

        least_squares:
            t* = pinv(A_reduced) @ dθ_desired   (closest reachable headings).
        nonlinear:
            t* = argmax Σ_{i∈members} (‖goal_i − p_i‖ − ‖goal_i − p_new_i‖)
            where p_new_i is p_i advanced by the fixed translation length
            ``move_distance`` along the rotated heading. Optimises *location*
            progress toward the goal, not heading alignment. Initialised at the
            least-squares solution.

        Returns:
            d_theta_actual: (N,) rotation applied to every robot.
        """
        t0_clock = time.perf_counter()

        pinv_A = np.linalg.pinv(A_reduced)
        t_ls = pinv_A @ d_theta_desired  # (2,)

        if self.method == "least_squares":
            t_star = t_ls
        else:
            # Fixed translation length (only members actually translate).
            d = move_distance
            pm = np.stack([px[members], py[members]], axis=1)        # (Nm, 2)
            gm = np.stack([gx[members], gy[members]], axis=1)        # (Nm, 2)
            theta_m = theta_current[members]                          # (Nm,)
            A_m = A_reduced[members, :]                               # (Nm, 2)
            dist0 = np.linalg.norm(gm - pm, axis=1)                  # (Nm,)

            def neg_progress(t: np.ndarray) -> float:
                theta_new = theta_m + A_m @ t                         # (Nm,)
                p_new = pm + d * np.stack(
                    [np.cos(theta_new), np.sin(theta_new)], axis=1
                )
                dist1 = np.linalg.norm(gm - p_new, axis=1)
                return float(-np.sum(dist0 - dist1))                  # maximise progress

            res = minimize(neg_progress, t_ls, method="BFGS")
            t_star = res.x

        d_theta_actual = A_reduced @ t_star  # (N,)
        self.last_solve_time_s = time.perf_counter() - t0_clock
        return d_theta_actual

    # ------------------------------------------------------------------

    def compute_actions(
        self, robot_state: np.ndarray, group: int, seed: int | None = None
    ) -> tuple[list[list[list[float]]], list[list[float]]]:
        """
        Compute the sim-input action frames for one coarse control of ``group``.

        Args:
            robot_state: Per-robot state array of shape (N, 11), same column
                layout as ``TD3Obstacle.prepare_state`` output:
                    0, 1  : px, py       — robot position
                    2, 3  : cos θ, sin θ — robot heading
                    4     : dist_to_goal
                    5, 6  : cos_err, sin_err
                    7, 8  : lin_vel_obs, ang_vel_obs
                    9, 10 : gx, gy       — goal position
            group: 1-based id of the activated coarse group (must be selectable;
                see :meth:`selectable_groups`).
            seed: Optional seed for the random rank-2 column drop.  When ``None``
                the instance RNG is used (legacy stochastic behaviour); passing a
                seed makes the control deterministic given (state, group) so the
                MPC lookahead vets exactly the plan it will later execute.

        Returns:
            rotation_frames: list of ``n_rot`` frames, each a list of N
                ``[0.0, ang_vel_i]`` sim-input pairs.  Applying them in sequence
                fully realises ``dθ_actual`` for every robot within the angular
                velocity bound.  Empty if no rotation is needed.
            translation_frames: list of ``n_trans`` frames, each a list of N
                ``[lin_vel_or_0, 0.0]`` pairs.  Applying them in sequence advances
                the chosen group's members by the full ``move_distance`` within the
                linear velocity bound; non-members stay still.  Empty if
                ``move_distance`` is zero.
        """
        if group not in self.selectable_groups():
            raise ValueError(
                f"group {group} is not selectable; "
                f"choose from {self.selectable_groups()}."
            )

        s = np.asarray(robot_state, dtype=np.float64)  # (N, 11)
        px, py = s[:, 0], s[:, 1]
        cos_h, sin_h = s[:, 2], s[:, 3]
        gx, gy = s[:, 9], s[:, 10]

        theta_current = np.arctan2(sin_h, cos_h)              # (N,)
        theta_desired = np.arctan2(gy - py, gx - px)          # (N,)
        d_theta_desired = _angle_wrap(theta_desired - theta_current)  # (N,)

        members = self._members[group - 1]
        move_distance = self.move_distances[group]
        rng = np.random.default_rng(seed) if seed is not None else None
        A_reduced = self._reduce_A(group, rng=rng)

        d_theta_actual = self._solve_rotation(
            A_reduced, theta_current, d_theta_desired, px, py, gx, gy, members,
            move_distance,
        )

        # ---- Rotation sub-steps -------------------------------------------
        # Pick the smallest number of equal sub-steps so that the largest
        # rotation stays within ang_max, then drive every robot at a constant
        # angular velocity that completes its own rotation over those steps.
        max_abs = float(np.max(np.abs(d_theta_actual))) if d_theta_actual.size else 0.0
        max_per_step = self.ang_max * self.step_time  # rad realisable in one step
        rotation_frames: list[list[list[float]]] = []
        if max_abs > 1e-9:
            n_rot = int(np.ceil(max_abs / max_per_step))
            ang_vels = d_theta_actual / (n_rot * self.step_time)  # (N,), |·| ≤ ang_max
            ang_vels = np.clip(ang_vels, -self.ang_max, self.ang_max)
            frame = [[0.0, float(ang_vels[i])] for i in range(self.num_robots)]
            rotation_frames = [list(map(list, frame)) for _ in range(n_rot)]

        # ---- Translation sub-steps ----------------------------------------
        # Mirror of the rotation split: pick the smallest number of equal
        # sub-steps so the per-step linear velocity stays within lin_max, then
        # advance the members at a constant velocity that covers the full
        # move_distance over those steps.  Non-members stay still throughout.
        member_set = set(int(i) for i in members)
        max_per_step_lin = self.lin_max * self.step_time  # metres realisable per step
        translation_frames: list[list[list[float]]] = []
        if move_distance > 1e-9:
            n_trans = int(np.ceil(move_distance / max_per_step_lin))
            lin_cmd = move_distance / (n_trans * self.step_time)  # ≤ lin_max
            lin_cmd = min(lin_cmd, self.lin_max)
            frame = [
                [lin_cmd if i in member_set else 0.0, 0.0]
                for i in range(self.num_robots)
            ]
            translation_frames = [list(map(list, frame)) for _ in range(n_trans)]

        return rotation_frames, translation_frames
