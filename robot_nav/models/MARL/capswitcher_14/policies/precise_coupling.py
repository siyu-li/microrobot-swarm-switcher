"""
PreciseCoupling: physically-consistent rotation for precise control.

The physics fix of the value/prior redesign (``docs/value_prior_redesign.md``
§2): independent single-robot rotation does not exist in the coupled system.
Any rotation is realised by driving the actuation columns, so rotating a
driven set ``S`` couples into every robot:

    t  = pinv(A_S) @ omega_S          (minimum-norm exact solve)
    w  = A @ t                        (everyone rotates)

``A_S`` is the driven rows of the actuation matrix.  When ``rank(A_S) = |S|``
— true for every singleton and every pair of the canonical 14-robot matrix —
the driven robots receive *exactly* their commanded angular velocities and the
bystanders receive the unavoidable side-rotation ``A @ pinv(A_S) @ omega_S``
(~0.6 rad mean per rad of target on the canonical matrix).  For larger sets
with ``rank(A_S) < |S|`` the solve degrades gracefully to the least-squares
best effort.

Bystander commands are clipped to the simulator's ``ang_max``; driven-robot
commands are assumed within bounds already (the GT policy's action space is
``[-1, 1]`` = ``[-ang_max, ang_max]``).  Clipping deviates from the linear
model only in the rare case a combined pair command pushes a bystander past
the cap; both the env and the forward model apply the same clip, so they stay
consistent with each other.

Translation is *not* coupled: only the driven set advances (membership gates
who translates, exactly as in coarse control).
"""

from __future__ import annotations

import numpy as np


class PreciseCoupling:
    """
    Coupled-rotation solver over a fixed actuation matrix.

    Args:
        A_full:  (N, K) actuation matrix (e.g. ``configs.A_FULL``).
        ang_max: simulator angular-velocity cap (rad/s); applied to the
                 resulting per-robot commands.
    """

    def __init__(self, A_full: np.ndarray, ang_max: float = 1.0) -> None:
        self.A = np.asarray(A_full, dtype=np.float64)
        if self.A.ndim != 2:
            raise ValueError(f"A_full must be 2-D (got shape {self.A.shape}).")
        self.num_robots = int(self.A.shape[0])
        self.ang_max = float(ang_max)
        # ``C[S] = A @ pinv(A_S)``: (N, |S|) map from driven commands to
        # everyone's rotation, cached per driven set (22 move-groups + a few
        # precise groups recur constantly).
        self._C: dict[tuple[int, ...], np.ndarray] = {}

    def _coupling_map(self, members: tuple[int, ...]) -> np.ndarray:
        C = self._C.get(members)
        if C is None:
            A_S = self.A[list(members), :]                    # (|S|, K)
            C = self.A @ np.linalg.pinv(A_S)                  # (N, |S|)
            self._C[members] = C
        return C

    def coupled_ang(
        self, members: np.ndarray | list, omega_members: np.ndarray | list
    ) -> np.ndarray:
        """
        Per-robot angular velocities realising the driven set's commands.

        Args:
            members:       driven robot indices (len |S|).
            omega_members: their commanded angular velocities (len |S|).

        Returns:
            (N,) angular velocity for every robot, clipped to ``ang_max``.
            Entries at ``members`` equal ``omega_members`` exactly whenever
            ``rank(A_S) = |S|`` and the commands are within ``ang_max``.
        """
        key = tuple(int(i) for i in members)
        omega = np.asarray(omega_members, dtype=np.float64)
        if omega.shape != (len(key),):
            raise ValueError(
                f"omega_members shape {omega.shape} does not match "
                f"{len(key)} driven robots"
            )
        w = self._coupling_map(key) @ omega                   # (N,)
        return np.clip(w, -self.ang_max, self.ang_max)
