"""
SwitcherCost: the single source of truth for CAPSwitcher decision pricing.

Loaded from a per-instantiation YAML (``cost_6robots.yaml`` /
``cost_14robots.yaml``) so the same definition serves any robot count:

* **Precise** — ``precise_unit`` is the cost of driving **one robot for one
  sim sub-step** under the precise mechanism.  A precise decision therefore
  costs ``precise_unit × (robots actually driven) × (sub-steps each)``.
  Robots already at goal are skipped by the precise rollout and contribute
  nothing.  The planner prices the nominal
  ``precise_unit × n_unreached × selection_interval``; the environment charges
  the sub-steps actually executed (they differ only when a terminal event
  truncates the rollout).
* **Coarse** — every move-group carries its own configured ``cost`` and
  ``move_distance``.  Costs are free constants (edit the YAML), **not**
  derived from ``n_members × move_distance`` — though ``cost: auto`` in the
  YAML falls back to exactly that product.

Each YAML group entry lists its ``members`` so the file is self-documenting
when editing costs; the loader treats them as a checksum and
:meth:`SwitcherCost.validate_members` compares them against the live group
algebra, so a reordering or membership drift fails loudly instead of silently
mispricing actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GroupCost:
    """Pricing of one coarse move-group."""

    group_id: int            # coarse action id (list position in the search)
    members: tuple[int, ...]  # robot indices — documentation + drift checksum
    move_distance: float     # metres each member advances per control
    cost: float              # per-decision cost constant


class SwitcherCost:
    """
    Decision-cost table for one CAPSwitcher instantiation.

    Args:
        precise_unit: Cost of one robot for one precise sub-step.
        groups:       :class:`GroupCost` entries, one per coarse action.
    """

    def __init__(self, precise_unit: float, groups: list[GroupCost]) -> None:
        self.precise_unit = float(precise_unit)
        self._groups: dict[int, GroupCost] = {g.group_id: g for g in groups}
        if len(self._groups) != len(groups):
            raise ValueError("duplicate group ids in cost config")

    # ---- coarse -------------------------------------------------------

    def coarse_cost(self, group_id: int) -> float:
        """Configured per-decision cost of coarse move-group ``group_id``."""
        return self._groups[group_id].cost

    def move_distance(self, group_id: int) -> float:
        """Configured translation length (m) of move-group ``group_id``."""
        return self._groups[group_id].move_distance

    @property
    def move_distances(self) -> dict[int, float]:
        """``{group_id: move_distance}`` — the coarse primitive's table."""
        return {gid: g.move_distance for gid, g in self._groups.items()}

    @property
    def group_ids(self) -> list[int]:
        return sorted(self._groups)

    # ---- precise ------------------------------------------------------

    def precise_cost(self, n_robots_moved: int, n_substeps: int) -> float:
        """
        Cost of a precise decision that drove ``n_robots_moved`` robots for
        ``n_substeps`` sub-steps **each** (nominal planner pricing).
        """
        return self.precise_unit * int(n_robots_moved) * int(n_substeps)

    def precise_substep_cost(self, n_substeps_total: int) -> float:
        """
        Cost of ``n_substeps_total`` executed precise sub-steps summed over
        robots (the environment's actual-count pricing — one robot moves per
        precise sub-step).
        """
        return self.precise_unit * int(n_substeps_total)

    # ---- validation ---------------------------------------------------

    def validate_members(self, expected: dict[int, list[int]]) -> None:
        """
        Check the YAML's group table against the live group algebra.

        ``expected`` maps group id -> member indices (e.g. from
        ``CoarseSteering.members_of``).  Raises ``ValueError`` on any missing
        id or membership mismatch.
        """
        if set(expected) != set(self._groups):
            raise ValueError(
                f"cost config group ids {sorted(self._groups)} != "
                f"system group ids {sorted(expected)}"
            )
        for gid, members in expected.items():
            got = tuple(int(i) for i in members)
            if self._groups[gid].members != got:
                raise ValueError(
                    f"cost config group {gid} members "
                    f"{list(self._groups[gid].members)} != system {list(got)} "
                    "— regenerate the YAML to match the group algebra"
                )

    # ---- loading ------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SwitcherCost":
        """
        Load a cost config.  Schema::

            precise_unit: <float>          # per robot per sub-step
            coarse:
              - {id: 0, members: [2, 6, 10], move_distance: 1.5, cost: 4.5}
              - {id: 1, members: [4, 6, 12], move_distance: 1.5, cost: auto}
              ...

        ``cost: auto`` resolves to ``len(members) × move_distance``.
        """
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict) or "precise_unit" not in raw or "coarse" not in raw:
            raise ValueError(f"{path}: expected keys 'precise_unit' and 'coarse'")
        groups: list[GroupCost] = []
        for entry in raw["coarse"]:
            members = tuple(int(i) for i in entry["members"])
            move_distance = float(entry["move_distance"])
            cost = entry["cost"]
            if cost == "auto":
                cost = len(members) * move_distance
            groups.append(
                GroupCost(
                    group_id=int(entry["id"]),
                    members=members,
                    move_distance=move_distance,
                    cost=float(cost),
                )
            )
        if not groups:
            raise ValueError(f"{path}: 'coarse' must list at least one group")
        return cls(precise_unit=float(raw["precise_unit"]), groups=groups)
