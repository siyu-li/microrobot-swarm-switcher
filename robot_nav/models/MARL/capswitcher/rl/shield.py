"""
Hard safety shield for the CAPSwitcher (lookahead over the coarse primitive).

The coarse control of a group is fully determined once its frames are built
(``CoarseSteering.compute_actions``): a rotation phase (no translation — only
headings change) followed by a translation phase in which **only the group's
members** advance in a straight line along their post-rotation heading, while
every other robot and every obstacle stays put.  That makes the swept geometry
of a coarse decision *exactly* reconstructable from the frames, so we can vet a
candidate coarse move geometrically **before** committing it.

A coarse decision is declared *safe* iff, at every sub-step the simulator would
integrate, the moving members keep a clearance ``>= d_safe`` from

  * every other robot   (centre distance − 2·ρ), and
  * every obstacle       (centre distance − r_obs − ρ),

where ρ is the robot radius.  Only member-involving pairs are scored, so the
shield judges *the risk introduced by moving this group*, not pre-existing
proximity the move does not change.

The same reconstruction yields the members' end positions, hence the predicted
goal-distance reduction (progress) of the move — used by the P1 efficiency rule.

Because coarse is deterministic given its frames and all other bodies are frozen
over the decision horizon, this lookahead matches what the simulator will
actually do at its collision-check points; ``d_safe`` is the single conservatism
knob (raise it to cover model/sensing slack).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# robot_state column layout (see CoarseSteering.compute_actions docstring).
_PX, _PY, _COS, _SIN, _GX, _GY = 0, 1, 2, 3, 9, 10


@dataclass
class ShieldGeometry:
    """Static collision geometry for one decision (refresh per decision)."""

    rho: float                 # robot radius (m)
    obstacle_xy: np.ndarray    # (M, 2) obstacle centres
    obstacle_r: np.ndarray     # (M,)   obstacle radii

    @classmethod
    def from_sim(cls, sim, default_rho: float = 0.2) -> "ShieldGeometry":
        """Read robot radius and obstacle centres/radii from the live sim."""
        robots = sim.env.robot_list
        rho = float(getattr(robots[0], "radius", default_rho)) if robots else default_rho
        obs = sim.env.obstacle_list
        if obs:
            xy = np.array([o.position.flatten()[:2] for o in obs], dtype=np.float64)
            r = np.array([float(o.radius) for o in obs], dtype=np.float64)
        else:
            xy = np.empty((0, 2), dtype=np.float64)
            r = np.empty((0,), dtype=np.float64)
        return cls(rho=rho, obstacle_xy=xy, obstacle_r=r)


def swept_positions(
    robot_state: np.ndarray,
    rotation_frames: list,
    translation_frames: list,
    step_time: float,
) -> np.ndarray:
    """
    Reconstruct the per-sub-step robot positions of a coarse control.

    Rotation frames only change headings (lin velocity is zero), so positions
    are constant through the rotation phase.  Each translation frame advances a
    robot by ``lin_i · step_time`` along its post-rotation heading; non-members
    have ``lin_i = 0`` and stay put.  Sampling at every translation sub-step
    boundary (including the start) reproduces exactly the points at which the
    simulator integrates and checks collisions.

    Args:
        robot_state:        (N, 11) decision-time state.
        rotation_frames:    list of (N, 2) [lin, ang] frames (lin == 0).
        translation_frames: list of (N, 2) [lin, ang] frames (ang == 0).
        step_time:          simulator sub-step duration (s).

    Returns:
        (T+1, N, 2) positions, T = number of translation frames.
    """
    s = np.asarray(robot_state, dtype=np.float64)
    p0 = np.stack([s[:, _PX], s[:, _PY]], axis=1)           # (N, 2)
    theta = np.arctan2(s[:, _SIN], s[:, _COS]).copy()        # (N,)

    # Post-rotation heading: accumulate the angular increments of every frame.
    for frame in rotation_frames:
        f = np.asarray(frame, dtype=np.float64)              # (N, 2)
        theta = theta + f[:, 1] * step_time

    heading = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (N, 2)

    samples = [p0]
    pos = p0.copy()
    for frame in translation_frames:
        f = np.asarray(frame, dtype=np.float64)              # (N, 2)
        step_len = (f[:, 0] * step_time)[:, None]            # (N, 1) per robot
        pos = pos + step_len * heading
        samples.append(pos.copy())
    return np.stack(samples, axis=0)                         # (T+1, N, 2)


def min_member_clearance(
    samples: np.ndarray,
    members: np.ndarray,
    geom: ShieldGeometry,
) -> float:
    """
    Minimum clearance the moving members keep over the whole swept path.

    Only member-involving pairs are scored (robot–robot against all other
    robots, robot–obstacle against all obstacles), so pre-existing proximity
    between two frozen bodies does not penalise the move.

    Returns:
        Smallest clearance (m) over all sub-steps and members; +inf if there is
        nothing to hit.
    """
    members = np.asarray(members, dtype=int)
    if members.size == 0:
        return float("inf")

    rho = geom.rho
    worst = float("inf")
    for P in samples:                                        # (N, 2) per sub-step
        mp = P[members]                                      # (Nm, 2)

        # robot–robot: each member vs every other robot.
        diff = mp[:, None, :] - P[None, :, :]                # (Nm, N, 2)
        d = np.linalg.norm(diff, axis=2)                     # (Nm, N)
        for k, m in enumerate(members):
            d[k, m] = np.inf                                 # exclude self
        rr = float(d.min()) - 2.0 * rho
        worst = min(worst, rr)

        # robot–obstacle: each member vs every obstacle boundary.
        if geom.obstacle_xy.shape[0] > 0:
            od = np.linalg.norm(
                mp[:, None, :] - geom.obstacle_xy[None, :, :], axis=2
            )                                                # (Nm, M)
            ro = float((od - geom.obstacle_r[None, :] - rho).min())
            worst = min(worst, ro)
    return worst


def predicted_progress(
    robot_state: np.ndarray,
    members: np.ndarray,
    end_positions: np.ndarray,
) -> float:
    """
    Goal-distance reduction (summed over members) the coarse move would yield.

    Non-members do not translate, so only members contribute.  Positive means
    the move makes net progress toward the goals.

    Args:
        robot_state:   (N, 11) decision-time state (goal at cols 9, 10).
        members:       member robot indices.
        end_positions: (N, 2) positions after the move (last swept sample).
    """
    members = np.asarray(members, dtype=int)
    if members.size == 0:
        return 0.0
    s = np.asarray(robot_state, dtype=np.float64)
    goals = np.stack([s[:, _GX], s[:, _GY]], axis=1)         # (N, 2)
    start = np.stack([s[:, _PX], s[:, _PY]], axis=1)         # (N, 2)
    d_start = np.linalg.norm(goals[members] - start[members], axis=1)
    d_end = np.linalg.norm(goals[members] - end_positions[members], axis=1)
    return float(np.sum(d_start - d_end))


@dataclass
class CoarseCandidate:
    """Result of vetting one coarse group at the current state."""

    group: int
    frames: list          # rotation_frames + translation_frames (ready to run)
    clearance: float      # min member clearance over the swept path (m)
    progress: float       # predicted summed goal-distance reduction (m)
    safe: bool            # clearance >= d_safe


def evaluate_coarse_group(
    coarse,
    robot_state: np.ndarray,
    group: int,
    geom: ShieldGeometry,
    d_safe: float,
    seed: int | None = None,
) -> CoarseCandidate:
    """
    Build the coarse frames for ``group`` and vet them with the lookahead.

    The returned ``frames`` are exactly the ones that must be executed (the
    primitive's column drop is random, so re-generating would vet a different
    plan than the one run).  Pass ``seed`` to make the drop deterministic given
    (state, group) — the MPC lookahead uses this so the plan it scores at a node
    is the same plan it would execute there.
    """
    rotation_frames, translation_frames = coarse.compute_actions(
        robot_state, group, seed=seed
    )
    samples = swept_positions(
        robot_state, rotation_frames, translation_frames, coarse.step_time
    )
    clearance = min_member_clearance(samples, coarse.members_of(group), geom)
    progress = predicted_progress(
        robot_state, coarse.members_of(group), samples[-1]
    )
    return CoarseCandidate(
        group=group,
        frames=rotation_frames + translation_frames,
        clearance=clearance,
        progress=progress,
        safe=clearance >= d_safe,
    )
