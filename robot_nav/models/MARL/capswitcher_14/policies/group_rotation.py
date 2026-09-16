"""
GroupRotation: uniform block rotation for precise control.

The group-based alternative to ``PreciseCoupling`` (pinv).  Both answer the
same physical constraint — independent single-robot rotation does not exist,
any rotation is ``w = A @ t`` — but they pick different realisable patterns:

* ``PreciseCoupling`` makes the driven robot exact via the minimum-norm solve
  ``t = pinv(A_S) @ omega_S``; the price is a dense side-rotation touching all
  13 bystanders (Σ|w| per unit command 7.75–9.50 depending on the robot).
* ``GroupRotation`` rotates a whole *uniform block* containing the driven
  robot at exactly the commanded rate.  The only sets that can rotate
  uniformly while everyone else holds exactly still are the ones whose
  indicator lies in ``col(A)`` — for the canonical matrix these are the 8
  size-7 bit-groups (each original group ``g_j`` and its complement ``h_j``)
  and all-14; nothing smaller exists.  Rotating a block costs Σ|w| = 6.00 per
  unit command and touches only 6 bystanders — strictly less disturbance than
  pinv, and the 8 untouched robots hold perfectly still.

Block rule (fixed, stateless)
-----------------------------
One global ``bit`` (default 0) splits the 14 robots into the two blocks
``g_bit`` / ``h_bit``; a driven robot rotates *its own block*.  The rule is a
pure function of the robot index — no state enters the choice, so the forward
model and the simulator can never disagree on which block rotates (no
threshold-flip drift site), and the same 2 blocks are used for the whole run.
Which bit anchors the partition is a convention; any of the 4 bits gives an
equally valid 7+7 split.

With the default bit the blocks are the even- and odd-indexed robots (code
``i+1`` has bit 0 set exactly when ``i`` is even), so the canonical pairs
config ``[[0,1], [2,3], ...]`` always drives one robot per block and both
commands are met exactly.  Driven robots sharing a block have their commands
summed over that block (then clipped) — the least-squares analogue does not
arise because the block pattern is exact by construction.

As with ``PreciseCoupling``: translation is not coupled (only the driven set
advances), commands are clipped to ``ang_max``, and the interface is
``coupled_ang(members, omega_members) -> (N,)`` so the class is a drop-in at
every existing ``coupling`` call site.  The block's rotation lands in
``sim_actions[:, 1]`` for all members, hence in every member's last-action
input to the next GAT forward.
"""

from __future__ import annotations

import numpy as np


class GroupRotation:
    """
    Fixed-block uniform rotation over a given actuation matrix.

    Args:
        A_full:  (N, K) actuation matrix (e.g. ``configs.A_FULL``).  Used to
                 *verify* the blocks are realisable (indicator in ``col(A)``);
                 the returned commands are the uniform block patterns
                 themselves.
        ang_max: simulator angular-velocity cap (rad/s), applied to the result.
        bit:     which code bit anchors the two blocks (0-based).  Robot ``i``
                 (binary code ``i+1``) is in block ``g_bit`` if the bit is set,
                 else in block ``h_bit``.
    """

    def __init__(self, A_full: np.ndarray, ang_max: float = 1.0, bit: int = 0) -> None:
        self.A = np.asarray(A_full, dtype=np.float64)
        if self.A.ndim != 2:
            raise ValueError(f"A_full must be 2-D (got shape {self.A.shape}).")
        self.num_robots = int(self.A.shape[0])
        self.ang_max = float(ang_max)
        self.bit = int(bit)

        codes = np.arange(1, self.num_robots + 1)
        in_g = (codes >> self.bit) & 1 == 1
        # blocks[i] = indicator (N,) of the block robot i rotates with.
        g = in_g.astype(np.float64)
        h = 1.0 - g
        self._block = np.where(in_g[:, None], g[None, :], h[None, :])  # (N, N)

        # Physics check: both block indicators must be realisable, i.e. lie in
        # col(A) — otherwise "rotate the block, everyone else exactly still"
        # is not a pattern this actuation matrix can produce.
        proj = self.A @ np.linalg.pinv(self.A)
        for name, ind in (("g", g), ("h", h)):
            if not np.allclose(proj @ ind, ind, atol=1e-9):
                raise ValueError(
                    f"block {name}_{self.bit} is not realisable through A_full "
                    "(indicator not in col(A))"
                )

    def block_of(self, robot: int) -> np.ndarray:
        """Indices of the block robot ``robot`` rotates with (includes it)."""
        return np.flatnonzero(self._block[int(robot)])

    def coupled_ang(
        self, members: np.ndarray | list, omega_members: np.ndarray | list
    ) -> np.ndarray:
        """
        Per-robot angular velocities realising the driven set's commands.

        Args:
            members:       driven robot indices (len |S|).
            omega_members: their commanded angular velocities (len |S|).

        Returns:
            (N,) angular velocity for every robot, clipped to ``ang_max``:
            each driven robot's whole block at its commanded rate (summed
            where driven robots share a block).
        """
        idx = [int(i) for i in members]
        omega = np.asarray(omega_members, dtype=np.float64)
        if omega.shape != (len(idx),):
            raise ValueError(
                f"omega_members shape {omega.shape} does not match "
                f"{len(idx)} driven robots"
            )
        w = np.zeros(self.num_robots, dtype=np.float64)
        for r, om in zip(idx, omega):
            w += om * self._block[r]
        return np.clip(w, -self.ang_max, self.ang_max)
