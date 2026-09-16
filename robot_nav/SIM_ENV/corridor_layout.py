"""
Banded ("corridor") episode layout: sparse → dense → sparse along x.

The randomized-obstacle world scatters obstacles, starts and goals uniformly
over the whole arena, which makes a rendered episode hard to read: obstacle
density is the same everywhere, so *why* the switcher left one coarse group for
another is confounded with wherever the robots happened to be.  This layout
removes that confound by making density a function of one coordinate only:

    │ start band │   free   │ obstacle band │   free   │ goal band │
    └── sparse ──┴──────────┴──── dense ────┴──────────┴── sparse ─┘
      x = x0                    (the gate)                   x = x1

Every robot therefore runs the same three-phase traverse — open field, gate,
open field — and a group switch can be read against the phase it happened in.

Determinism
-----------
Obstacle positions draw from ``numpy.random`` and robot starts from ``random``,
the same two globals the unbanded path uses, so
:func:`~robot_nav.models.MARL.capswitcher.rl.switcher_env.seed_episode` still
pins a corridor episode exactly as it pins a scattered one.  Goals are drawn by
irsim's ``set_random_goal`` (also ``numpy.random``); this module only supplies
the band it must draw inside — see :meth:`CorridorLayout.goal_range_limits`.

Everything here is pure geometry: no irsim, no torch.  Obstacles are treated as
circles of a caller-supplied radius, which is exact for the circular obstacles
of the 14-robot worlds and a conservative bounding radius otherwise.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

__all__ = ["CorridorLayout", "sample_obstacles", "sample_starts"]

# Robots traverse left→right (+1) or right→left (-1).
LEFT_TO_RIGHT = 1
RIGHT_TO_LEFT = -1


@dataclass(frozen=True)
class CorridorLayout:
    """
    Widths of the three bands, as fractions of the world's x-extent.

    Attributes:
        start_width: Start band width; robots spawn in ``[x0+margin, x0+w·W]``.
        goal_width:  Goal band width, mirrored against the right wall.
        band:        ``(lo, hi)`` fractions bounding the obstacle band.  The
                     default straddles the midline, leaving a free run either
                     side of the gate so the sparse phases are visible.  Bounds
                     the obstacle *centres*, so the occupied strip is wider than
                     the band by one obstacle radius on each side.
        margin:      Clearance kept from the world boundary, metres.
        min_gap:     Minimum *free surface* gap between two obstacles, metres.
                     This is what keeps the band a gate rather than a wall: at
                     the default 1.0 any pair is ``2(ρ+d_safe)`` apart for the
                     0.2 m robots at ``d_safe=0.3``, so a shielded coarse move
                     can thread between any two of them.
        bidirectional: If True the second half of the robots traverse
                     right→left, so the swarm has to interpenetrate inside the
                     gate.  If False (default) every robot goes left→right.
        obstacle_free: Sanity mode: instead of forming a gate, the whole
                     obstacle set is parked in a strip along the top wall,
                     outside :meth:`y_band`.  The arena is effectively empty,
                     but the world keeps its obstacle count — the frozen GAT
                     backbone is built with ``num_obstacles ==
                     len(env.obstacle_list)``, so the objects cannot simply be
                     dropped from the YAML.
        aligned_goals: Sanity mode: robots start on evenly spaced y-lanes
                     facing their direction of travel, and each goal is pinned
                     to its robot's start y (see :meth:`goal_range_limits`).
                     The nominal solution is then "drive straight ahead".
                     Composes with ``bidirectional`` (opposing halves still
                     get distinct lanes) and with ``obstacle_free``.
        max_attempts: Rejection-sampling attempts per object before the
                     best-so-far candidate is accepted.
        restarts:    Whole-band redraws when greedy placement leaves the last
                     obstacle no room; the widest-gated draw wins.
    """

    start_width: float = 0.25
    goal_width: float = 0.25
    band: tuple[float, float] = (0.40, 0.60)
    margin: float = 1.0
    min_gap: float = 1.0
    bidirectional: bool = False
    obstacle_free: bool = False
    aligned_goals: bool = False
    max_attempts: int = 200
    restarts: int = 20

    # Vertical strip (m) reserved along the top wall for parked obstacles in
    # obstacle_free mode.  With the default margin this keeps a robot at the
    # top of `y_band` >= 1.5 m (centre to surface) from a parked 0.7 m
    # obstacle, so the GAT's obstacle-proximity shaping (threshold 1.5 m)
    # never fires in the traffic band.
    PARK_RESERVE = 2.0

    # ------------------------------------------------------------------
    # Bands, in metres
    # ------------------------------------------------------------------

    def _span(self, x_range) -> float:
        return float(x_range[1] - x_range[0])

    def obstacle_band(self, x_range) -> tuple[float, float]:
        """x-interval the obstacles are drawn in."""
        x0, w = float(x_range[0]), self._span(x_range)
        return x0 + self.band[0] * w, x0 + self.band[1] * w

    def start_band(self, x_range, direction: int) -> tuple[float, float]:
        """x-interval a robot heading ``direction`` starts in."""
        x0, x1, w = float(x_range[0]), float(x_range[1]), self._span(x_range)
        if direction == LEFT_TO_RIGHT:
            return x0 + self.margin, x0 + self.start_width * w
        return x1 - self.start_width * w, x1 - self.margin

    def goal_band(self, x_range, direction: int) -> tuple[float, float]:
        """x-interval the goal of a robot heading ``direction`` lies in."""
        x0, x1, w = float(x_range[0]), float(x_range[1]), self._span(x_range)
        if direction == LEFT_TO_RIGHT:
            return x1 - self.goal_width * w, x1 - self.margin
        return x0 + self.margin, x0 + self.goal_width * w

    def y_band(self, y_range) -> tuple[float, float]:
        """
        y-interval everything is drawn in: the full height less margin, and in
        obstacle_free mode also less the parking strip along the top wall.
        """
        y1 = float(y_range[1]) - self.margin
        if self.obstacle_free:
            y1 -= self.PARK_RESERVE
        return float(y_range[0]) + self.margin, y1

    def lanes(self, n: int, y_range) -> np.ndarray:
        """Evenly spaced start y per robot, used by the aligned sanity mode."""
        y0, y1 = self.y_band(y_range)
        return np.linspace(y0, y1, n)

    def parked_obstacles(self, radii, x_range, y_range) -> np.ndarray:
        """
        Deterministic parking row hugging the top wall, one slot per obstacle.

        Used instead of :func:`sample_obstacles`'s banded draw when
        ``obstacle_free`` — the row sits above :meth:`y_band` (which reserves
        ``PARK_RESERVE`` below the wall), so nothing in the traffic band ever
        interacts with it.  Deterministic on purpose: parked scenery consumes
        no RNG, so it cannot shift the seeded start/goal draws between runs.

        Returns:
            ``(n, 3)`` array of ``(x, y, theta)``.
        """
        radii = np.asarray(radii, dtype=np.float64)
        r = float(radii.max(initial=0.0))
        xs = np.linspace(
            float(x_range[0]) + r + 0.1, float(x_range[1]) - r - 0.1, len(radii)
        )
        out = np.zeros((len(radii), 3), dtype=np.float64)
        out[:, 0] = xs
        out[:, 1] = float(y_range[1]) - r - 0.1
        return out

    def directions(self, n: int) -> np.ndarray:
        """Per-robot traverse direction; the second half is flipped if bidirectional."""
        d = np.full(n, LEFT_TO_RIGHT, dtype=int)
        if self.bidirectional:
            d[n // 2:] = RIGHT_TO_LEFT
        return d

    def goal_range_limits(
        self, x_range, y_range, direction: int, start_y: float | None = None
    ) -> list:
        """
        ``range_limits`` for irsim's ``set_random_goal`` — ``[[lo], [hi]]`` over
        ``(x, y, theta)``, restricted to this robot's goal band.

        With ``aligned_goals``, ``start_y`` (the robot's start — or, on a
        respawn, current — y) collapses the y-interval to that lane; without
        it, or when not aligned, the full :meth:`y_band` is used.
        """
        gx0, gx1 = self.goal_band(x_range, direction)
        y0, y1 = self.y_band(y_range)
        if self.aligned_goals and start_y is not None:
            y0 = y1 = float(np.clip(start_y, y0, y1))
        return [[gx0, y0, -np.pi], [gx1, y1, np.pi]]

    def validate(self) -> None:
        """Reject band settings that overlap or leave no free run."""
        lo, hi = self.band
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError(f"band must satisfy 0 <= lo < hi <= 1, got {self.band}")
        if self.obstacle_free:
            # No gate to keep clear of — the two sides only have to stay
            # disjoint from each other.
            if self.start_width + self.goal_width >= 1.0:
                raise ValueError(
                    f"start band ({self.start_width}) and goal band "
                    f"({self.goal_width}) overlap — widths must sum below 1"
                )
            return
        if self.start_width >= lo:
            raise ValueError(
                f"start band ({self.start_width}) reaches into the obstacle "
                f"band (starts at {lo}) — robots would spawn inside the gate"
            )
        if 1.0 - self.goal_width <= hi:
            raise ValueError(
                f"goal band (from {1.0 - self.goal_width}) reaches into the "
                f"obstacle band (ends at {hi}) — goals would sit inside the gate"
            )


def sample_obstacles(
    radii, x_range, y_range, layout: CorridorLayout
) -> np.ndarray:
    """
    Draw obstacle states inside the band, spaced by at least ``min_gap``.

    Rejection sampling on centre distance ``>= r_i + r_j + min_gap``.  Placement
    is sequential and greedy, so a bad early draw can leave no room for the last
    obstacle; the whole band is therefore redrawn up to ``layout.restarts``
    times and the widest-gated layout wins.  Within one draw, an obstacle that
    cannot be satisfied keeps its *best* candidate (the one maximising the
    tightest gap) rather than its last, so the failure mode is "as open as we
    could manage", never an arbitrary overlap.

    A layout tighter than ``min_gap`` is returned rather than raised on: a gate
    the shield cannot thread is a legitimate episode — it is what forces the
    switcher into precise mode — and refusing to generate one would bias the
    benchmark toward the easy half of the distribution.

    Args:
        radii: Per-obstacle radius, length ``n``.
        layout: The band geometry.

    Returns:
        ``(n, 3)`` array of ``(x, y, theta)``.
    """
    radii = np.asarray(radii, dtype=np.float64)
    if layout.obstacle_free:
        return layout.parked_obstacles(radii, x_range, y_range)
    bx0, bx1 = layout.obstacle_band(x_range)
    by0, by1 = layout.y_band(y_range)

    best_layout, best_min_gap = None, -np.inf
    for _ in range(max(1, layout.restarts)):
        out = np.zeros((len(radii), 3), dtype=np.float64)
        worst = np.inf
        for i, r in enumerate(radii):
            best, best_gap = None, -np.inf
            for _ in range(layout.max_attempts):
                cand = np.array([
                    np.random.uniform(bx0, bx1),
                    np.random.uniform(by0, by1),
                    np.random.uniform(-np.pi, np.pi),
                ])
                if i == 0:
                    best, best_gap = cand, np.inf
                    break
                d = np.linalg.norm(out[:i, :2] - cand[:2], axis=1)
                gap = float(np.min(d - radii[:i] - r))
                if gap > best_gap:
                    best, best_gap = cand, gap
                if gap >= layout.min_gap:
                    break
            out[i] = best
            worst = min(worst, best_gap)
        if worst > best_min_gap:
            best_layout, best_min_gap = out, worst
        if best_min_gap >= layout.min_gap:
            break
    return best_layout


def sample_starts(
    n: int, x_range, y_range, layout: CorridorLayout,
    obstacle_xy: np.ndarray, obstacle_r: np.ndarray,
    robot_radius: float = 0.2, spacing: float = 0.6, obstacle_clearance: float = 0.5,
) -> np.ndarray:
    """
    Draw robot start poses inside the start band(s).

    Same acceptance test as the unbanded path — ``spacing`` between robots and
    ``obstacle_clearance`` to any obstacle surface — and the same ``random``
    global, only over a restricted x-interval.  The obstacle test is kept even
    though the bands are disjoint: it costs nothing and keeps the function
    correct for layouts whose start band is widened toward the gate.

    With ``layout.aligned_goals`` each robot instead gets a fixed, evenly
    spaced y-lane and a heading facing its direction of travel; only x is
    drawn.  The acceptance tests still run, but the lanes satisfy them by
    construction for the 14-robot worlds.

    Returns:
        ``(n, 3)`` array of ``(x, y, theta)``.
    """
    y0, y1 = layout.y_band(y_range)
    directions = layout.directions(n)
    lanes = layout.lanes(n, y_range) if layout.aligned_goals else None
    out = np.zeros((n, 3), dtype=np.float64)

    for i in range(n):
        sx0, sx1 = layout.start_band(x_range, int(directions[i]))
        for _ in range(layout.max_attempts):
            if lanes is not None:
                cand = np.array([
                    random.uniform(sx0, sx1),
                    lanes[i],
                    0.0 if directions[i] == LEFT_TO_RIGHT else np.pi,
                ])
            else:
                cand = np.array([
                    random.uniform(sx0, sx1),
                    random.uniform(y0, y1),
                    random.uniform(-np.pi, np.pi),
                ])
            if i and np.min(np.linalg.norm(out[:i, :2] - cand[:2], axis=1)) < spacing:
                continue
            if len(obstacle_xy):
                surf = (
                    np.linalg.norm(obstacle_xy - cand[:2], axis=1)
                    - obstacle_r - robot_radius
                )
                if np.min(surf) < obstacle_clearance:
                    continue
            out[i] = cand
            break
        else:
            raise RuntimeError(
                f"could not place robot {i} in the start band "
                f"[{sx0:.2f}, {sx1:.2f}] x [{y0:.2f}, {y1:.2f}] in "
                f"{layout.max_attempts} attempts — widen `start_width` or "
                f"lower `spacing`"
            )
    return out
