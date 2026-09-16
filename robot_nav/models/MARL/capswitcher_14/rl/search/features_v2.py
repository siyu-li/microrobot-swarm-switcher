"""
V2 feature builder for the split feasibility / group-scheduling prior —
numpy-only, sim-free, vet-free.

Replaces the single 10-d row of ``features.py`` with two purpose-built views
of the same node state:

* **Feasibility rows** (K, ``FEAS_FEAT_DIM``) — geometry of the move the vet
  would check: member clearances (static, and "ahead" toward the goal
  bearing), member↔member and member↔bystander proximity, and the rotation
  sweep size.  Labels are the root shield vets harvested for free by
  ``eval_gaz14_lazy --log-pi-targets``.
* **Policy rows** (K, ``POLICY_BASE_DIM``) — scheduling view: member goal
  distances, alignment of the *unreached* members, useful-work vs harm
  fractions, and two group-membership features (rarity-weighted coverage,
  best-fit count) that need the static table maps below.  The feasibility
  net's predicted probability is appended as the 13th input at net time
  (``feas_policy_net.py``), never here — this module stays vet-free and
  net-free.

Global context (``GLOBAL_FEAT_DIM``): unreached fraction, mean goal distance,
the stuck proxy (unreached robots no group can help without disturbing more
reached members than it advances) and the greedy rarity-cover count — shared
by every policy row and the sole input of the precise edge's head.

Static per-robot table maps, precomputed once from ``move_groups``:

* ``ngroups[i]`` — number of coarse groups containing robot ``i`` (the
  rarity weight is ``1 / ngroups[i]``);
* ``minsize[i]`` — size of the smallest group containing ``i`` (a group is
  "best fit" for ``i`` when it has exactly that size).

Same contract as ``features.py``: nothing here may depend on shield vetting
output — the swept clearance is what the feasibility net *predicts*.
"""

from __future__ import annotations

import numpy as np

FEAS_FEAT_DIM = 11
POLICY_BASE_DIM = 12          # + feasibility probability = 13 at the net
GLOBAL_FEAT_DIM = 4

_SIZE_CLASSES = (3, 4, 7)
_AHEAD_HALF_ANGLE = np.pi / 3.0   # ±60° cone toward the goal bearing


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class GroupFeatureBuilderV2:
    """
    Build (feasibility rows, policy rows, global context) for one node state.

    Args:
        move_groups:    the 22 member-index lists (order = coarse action ids).
        move_distances: coarse translation length (m) per group id — the
                        "step length" of feasibility feature 6 (take it from
                        ``CoarseSteering14.move_distances``).
        goal_threshold: arrival radius (m).
        dist_scale:     distance normaliser (m).
        clearance_cap:  clearance saturation (m).
    """

    def __init__(
        self,
        move_groups: list,
        move_distances: dict[int, float] | float = 0.5,
        goal_threshold: float = 0.3,
        dist_scale: float = 10.0,
        clearance_cap: float = 2.0,
    ) -> None:
        self.move_groups = [np.asarray(g, dtype=int) for g in move_groups]
        K = len(self.move_groups)
        if isinstance(move_distances, dict):
            self.step_len = np.array(
                [float(move_distances[k]) for k in range(K)])
        else:
            self.step_len = np.full(K, float(move_distances))
        self.goal_threshold = float(goal_threshold)
        self.L = float(dist_scale)
        self.cap = float(clearance_cap)

        n = int(max(g.max() for g in self.move_groups)) + 1
        self.n_robots = n
        # (K, N) boolean membership; static per-robot table maps.
        self.member = np.zeros((K, n), dtype=bool)
        for k, g in enumerate(self.move_groups):
            self.member[k, g] = True
        self.sizes = self.member.sum(axis=1)                       # (K,)
        self.size_onehot = np.stack(
            [(self.sizes == s) for s in _SIZE_CLASSES], axis=1
        ).astype(np.float32)                                       # (K, 3)
        self.ngroups = self.member.sum(axis=0)                     # (N,)
        if (self.ngroups == 0).any():
            raise ValueError("every robot must appear in >= 1 move-group")
        self.rarity = 1.0 / self.ngroups                           # (N,)
        sizes_col = np.where(self.member, self.sizes[:, None], np.inf)
        self.minsize = sizes_col.min(axis=0)                       # (N,)

    # ------------------------------------------------------------------

    def __call__(self, model, ms) -> dict:
        """Deploy/collect adapter: forward-model state → feature dict."""
        return self.raw(
            ms.poses, model.goals,
            model.geom.obstacle_xy, model.geom.obstacle_r, model.geom.rho,
        )

    def raw(
        self,
        poses: np.ndarray,
        goals: np.ndarray,
        obstacle_xy: np.ndarray,
        obstacle_r: np.ndarray,
        rho: float,
    ) -> dict:
        """
        Returns:
            ``feas``   (K, FEAS_FEAT_DIM)  float32
            ``policy`` (K, POLICY_BASE_DIM) float32
            ``glob``   (GLOBAL_FEAT_DIM,)  float32
        """
        poses = np.asarray(poses, dtype=np.float64)
        goals = np.asarray(goals, dtype=np.float64)
        oxy = np.asarray(obstacle_xy, dtype=np.float64).reshape(-1, 2)
        orad = np.asarray(obstacle_r, dtype=np.float64).reshape(-1)
        rho = float(rho)
        pos, theta = poses[:, :2], poses[:, 2]
        n, K = pos.shape[0], len(self.move_groups)

        gvec = goals - pos
        dist = np.linalg.norm(gvec, axis=1)
        unreached = dist > self.goal_threshold
        # Arrived robots keep their heading (bearing degenerate) → err = 0.
        phi = np.where(unreached, np.arctan2(gvec[:, 1], gvec[:, 0]), theta)
        err = np.abs(_wrap(theta - phi)) / np.pi                   # (N,)

        # -- per-robot clearances (current geometry, never the vet) --------
        D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)                                # (N, N)
        static_clear = D.min(axis=1) - 2.0 * rho
        ahead_clear = np.full(n, self.cap)
        if oxy.shape[0] > 0:
            od = np.linalg.norm(pos[:, None, :] - oxy[None, :, :], axis=2)
            oc = od - orad[None, :] - rho
            static_clear = np.minimum(static_clear, oc.min(axis=1))
            vec = oxy[None, :, :] - pos[:, None, :]
            bearing = np.arctan2(vec[..., 1], vec[..., 0])
            in_cone = (np.abs(_wrap(bearing - phi[:, None]))
                       <= _AHEAD_HALF_ANGLE)
            ahead_clear = np.minimum(
                np.where(in_cone, oc, np.inf).min(axis=1), self.cap)
        static_clear = np.minimum(static_clear, self.cap)

        # -- per-group masked reductions ----------------------------------
        m = self.member                                            # (K, N) bool
        sz = self.sizes.astype(np.float64)                         # (K,)
        um = m & unreached[None, :]                                # (K, N)
        n_un = um.sum(axis=1).astype(np.float64)                   # (K,)

        def gmean(x, mask, count):
            return np.where(count > 0,
                            (x[None, :] * mask).sum(axis=1)
                            / np.maximum(count, 1), 0.0)

        def gmin(x, mask):
            return np.where(mask, x[None, :], np.inf).min(axis=1)

        def gmax(x, mask):
            return np.where(mask, x[None, :], -np.inf).max(axis=1)

        mean_d = gmean(dist, m, sz)
        min_d = gmin(dist, m)
        std_d = np.sqrt(np.maximum(
            gmean(dist ** 2, m, sz) - mean_d ** 2, 0.0))

        # member↔member / member↔bystander proximity (K,)
        pair = np.where(m[:, :, None] & m[:, None, :], D[None, :, :], np.inf)
        min_pair = pair.min(axis=(1, 2))
        cross = np.where(m[:, :, None] & ~m[:, None, :], D[None, :, :], np.inf)
        min_cross = cross.min(axis=(1, 2))

        feas = np.zeros((K, FEAS_FEAT_DIM), dtype=np.float32)
        feas[:, 0:3] = self.size_onehot
        feas[:, 3] = gmin(static_clear, m) / self.cap
        feas[:, 4] = gmin(ahead_clear, m) / self.cap
        feas[:, 5] = gmean(ahead_clear, m, sz) / self.cap
        feas[:, 6] = ((ahead_clear[None, :] < self.step_len[:, None])
                      & m).sum(axis=1) / sz
        feas[:, 7] = min_pair / self.L
        feas[:, 8] = min_cross / self.L
        feas[:, 9] = np.maximum(gmax(err, m), 0.0)
        feas[:, 10] = gmean(err, m, sz)

        policy = np.zeros((K, POLICY_BASE_DIM), dtype=np.float32)
        policy[:, 0:3] = self.size_onehot
        policy[:, 3] = mean_d / self.L
        policy[:, 4] = min_d / self.L
        policy[:, 5] = std_d / self.L
        policy[:, 6] = gmean(err, um, n_un)
        policy[:, 7] = np.maximum(gmax(err, um), 0.0)
        policy[:, 8] = n_un / sz
        policy[:, 9] = 1.0 - n_un / sz
        policy[:, 10] = (self.rarity[None, :] * um).sum(axis=1) / sz
        policy[:, 11] = (um & (sz[:, None] == self.minsize[None, :])
                         ).sum(axis=1) / sz

        return {"feas": feas, "policy": policy,
                "glob": self._globals(dist, unreached)}

    def globals_only(self, poses: np.ndarray, goals: np.ndarray) -> np.ndarray:
        """
        Just the (GLOBAL_FEAT_DIM,) context — the value residual's whole
        input.  Obstacle-free by construction, and skips the per-group
        feas/policy rows, so it is cheap enough for per-expansion leaf calls.
        """
        pos = np.asarray(poses, dtype=np.float64)[:, :2]
        dist = np.linalg.norm(np.asarray(goals, dtype=np.float64) - pos,
                              axis=1)
        return self._globals(dist, dist > self.goal_threshold)

    def _globals(self, dist: np.ndarray, unreached: np.ndarray) -> np.ndarray:
        # Stuck proxy: unreached robots whose every containing group has more
        # reached than unreached members.
        m = self.member
        n_un = (m & unreached[None, :]).sum(axis=1)
        helpless_group = (self.sizes - n_un) > n_un                # (K,)
        stuck = unreached & (~m | helpless_group[:, None]).all(axis=0)
        return np.array([
            unreached.mean(),
            dist.mean() / self.L,
            stuck.sum() / dist.shape[0],
            self._greedy_cover(unreached) / len(self.move_groups),
        ], dtype=np.float32)

    def _greedy_cover(self, unreached: np.ndarray) -> int:
        """Groups picked greedily by rarity-weighted coverage until every
        unreached robot is covered — a cheap lower-bound schedule length."""
        w = np.where(unreached, self.rarity, 0.0)
        count = 0
        while True:
            gain = (self.member * w[None, :]).sum(axis=1)
            best = int(gain.argmax())
            if gain[best] <= 0.0:
                return count
            count += 1
            w[self.move_groups[best]] = 0.0
