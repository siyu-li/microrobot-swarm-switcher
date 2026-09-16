"""
Static feature builder for the learned prior — numpy-only, sim-free.

Everything here is computable from the node's :class:`ModelState` (poses) plus
constants frozen per decision (goals, obstacle geometry, the move-group
table).  **Nothing may depend on shield vetting output**: the swept clearance
is the quantity the feasibility head is trained to *predict*; feeding a vet
result in would silently reintroduce the per-edge verification cost the prior
exists to remove.  ``min_static_clearance`` below is therefore the *current*
point clearance of the members, not the swept clearance of the move.

Per-group features (one row per coarse move-group, ``GROUP_FEAT_DIM``):

    0-2   size one-hot over {3, 4, 7}
    3     mean member goal distance          / dist_scale
    4     min  member goal distance          / dist_scale
    5     std  member goal distance          / dist_scale
    6     mean |heading-to-goal error|       / π
    7     max  |heading-to-goal error|       / π    (one badly aligned member
                                                     caps what rank-5 rotation
                                                     can fix)
    8     min static clearance of members    (capped, / clearance_cap)
    9     min "ahead" clearance of members   (obstacles within ±60° of the
                                              member's goal bearing; capped)

Global context (``GLOBAL_FEAT_DIM``): fraction of robots unreached, mean goal
distance / dist_scale, global min static clearance (capped) — shared by every
group row and feeding the precise edge's own head.

Attention statistics from the frozen GAT (scalar max / mean-of-max weights)
are a deliberate v1 omission: they would cost one GAT forward per expanded
node.  If added later as an ablation, they belong here behind a flag.
"""

from __future__ import annotations

import numpy as np

GROUP_FEAT_DIM = 10
GLOBAL_FEAT_DIM = 3

_SIZE_CLASSES = (3, 4, 7)
_AHEAD_HALF_ANGLE = np.pi / 3.0  # ±60° cone toward the goal bearing


def _angle_wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class GroupFeatureBuilder:
    """
    Build (per-group, global) feature matrices for one node state.

    Args:
        move_groups:   the 22 member-index lists (order = coarse action ids).
        dist_scale:    distance normaliser (m) — roughly the arena diameter.
        clearance_cap: clearance saturation (m); beyond this, more open space
                       carries no extra information.
    """

    def __init__(
        self,
        move_groups: list,
        dist_scale: float = 10.0,
        clearance_cap: float = 2.0,
    ) -> None:
        self.move_groups = [np.asarray(g, dtype=int) for g in move_groups]
        self.dist_scale = float(dist_scale)
        self.clearance_cap = float(clearance_cap)

    # ------------------------------------------------------------------

    def _static_clearance(self, model, pos: np.ndarray) -> np.ndarray:
        """
        Per-robot current point clearance (m): nearest other robot minus 2ρ vs
        nearest obstacle boundary minus ρ, whichever is closer.  Capped.
        """
        n = pos.shape[0]
        rho = model.geom.rho
        diff = pos[:, None, :] - pos[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        clear = d.min(axis=1) - 2.0 * rho
        if model.geom.obstacle_xy.shape[0] > 0:
            od = np.linalg.norm(
                pos[:, None, :] - model.geom.obstacle_xy[None, :, :], axis=2
            )
            oc = (od - model.geom.obstacle_r[None, :] - rho).min(axis=1)
            clear = np.minimum(clear, oc)
        return np.minimum(clear, self.clearance_cap)

    def _ahead_clearance(
        self, model, pos: np.ndarray, goal_bearing: np.ndarray
    ) -> np.ndarray:
        """
        Per-robot clearance restricted to obstacles within ±60° of the robot's
        goal bearing — what the robot would sweep toward.  Cap when nothing is
        ahead.
        """
        n = pos.shape[0]
        rho = model.geom.rho
        ahead = np.full(n, self.clearance_cap, dtype=np.float64)
        if model.geom.obstacle_xy.shape[0] == 0:
            return ahead
        vec = model.geom.obstacle_xy[None, :, :] - pos[:, None, :]   # (N, M, 2)
        od = np.linalg.norm(vec, axis=2)                              # (N, M)
        bearing = np.arctan2(vec[..., 1], vec[..., 0])                # (N, M)
        in_cone = np.abs(_angle_wrap(bearing - goal_bearing[:, None])) <= _AHEAD_HALF_ANGLE
        oc = od - model.geom.obstacle_r[None, :] - rho
        oc = np.where(in_cone, oc, np.inf)
        return np.minimum(oc.min(axis=1), self.clearance_cap)

    # ------------------------------------------------------------------

    def __call__(self, model, ms) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            group_feats:  (n_groups, GROUP_FEAT_DIM) float32
            global_feats: (GLOBAL_FEAT_DIM,) float32
        """
        pos = ms.poses[:, :2]
        theta = ms.poses[:, 2]
        goal_vec = model.goals - pos
        dist = np.linalg.norm(goal_vec, axis=1)
        goal_bearing = np.arctan2(goal_vec[:, 1], goal_vec[:, 0])
        heading_err = np.abs(_angle_wrap(goal_bearing - theta))

        static_clear = self._static_clearance(model, pos)
        ahead_clear = self._ahead_clearance(model, pos, goal_bearing)

        rows = np.zeros((len(self.move_groups), GROUP_FEAT_DIM), dtype=np.float32)
        for i, members in enumerate(self.move_groups):
            md = dist[members]
            rows[i, 0:3] = [
                1.0 if members.size == s else 0.0 for s in _SIZE_CLASSES
            ]
            rows[i, 3] = md.mean() / self.dist_scale
            rows[i, 4] = md.min() / self.dist_scale
            rows[i, 5] = md.std() / self.dist_scale
            rows[i, 6] = heading_err[members].mean() / np.pi
            rows[i, 7] = heading_err[members].max() / np.pi
            rows[i, 8] = static_clear[members].min() / self.clearance_cap
            rows[i, 9] = ahead_clear[members].min() / self.clearance_cap

        frac_unreached = float(np.mean(dist > model.goal_threshold))
        global_feats = np.array(
            [
                frac_unreached,
                dist.mean() / self.dist_scale,
                static_clear.min() / self.clearance_cap,
            ],
            dtype=np.float32,
        )
        return rows, global_feats
