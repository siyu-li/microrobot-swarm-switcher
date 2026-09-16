"""
Search-trace recording for the best-first planners — the Phase-3 data source
of the value/prior redesign (``docs/value_prior_redesign.md`` §6).

Design principle: **save raw snapshots, not features.**  A shard stores every
*expanded* node's state (poses + last_actions) and, per branch, everything the
expansion learned (legality, clearance, the generated child's ``g``/``h``/
terminal/collision) plus per-plan and per-shard constants.  Any feature set —
including ones designed after collection — is rebuilt offline from these
files, so feature iterations never re-run the sim or the search.

Labels derivable offline (no re-simulation):

* **Value targets** — walk the goal reference's parent chain: on-path node
  ``n`` gets ``y_V(n) = plan_cost − g(n)`` (exact in-model remaining cost).
* **Prior targets** — the on-path chosen branch index per node; sibling
  ``child_g + child_h`` for soft/advantage-weighted variants.
* **Feasibility targets** — every coarse branch row carries the shield's
  exact ``clearance`` (vetted whether or not the edge was safe).

File layout: ``<out_dir>/meta.json`` (once per shard directory — system
constants) and one ``plan_<ep>_<dec>.npz`` per planning call:

    nodes:    parent_row, parent_aidx (int32; −1 at root), g, h (f32),
              depth (int32), poses (f32 [M,N,3]), last_actions (f32 [M,N,2])
    branches: br_node (int32 row into nodes), br_mode (int8),
              br_group / br_pgroup (int16, −1 = n/a), br_step_cost (f32),
              br_safe (bool), br_clearance (f32, NaN for precise),
              br_child_g / br_child_h (f32, NaN if refuted/dead-end),
              br_child_terminal / br_child_collision (bool),
              br_child_poses (f32 [B,N,3]) / br_child_last (f32 [B,N,2]) —
              the materialised child *state* exactly as the search scored it
              (collision dead-ends included; NaN only for refuted coarse
              vets, which never materialise).  Stored because the CUDA GAT
              forward is non-bit-reproducible: precise children cannot be
              regenerated exactly, and sub-searches / sibling resolution must
              start from the state the teacher actually evaluated.
    plan:     solved, cap_hit (bool), plan_cost (f32),
              goal_parent_row / goal_aidx (int32, −1 = none; the goal/best
              node is a *generated* child — terminals are never expanded —
              so it is addressed as (parent node row, branch index)),
              expansions, n_coarse_vets, n_precise_expansions (int32),
              episode, seed, decision_index (int32),
              goals (f32 [N,2]), obstacle_xy (f32 [M_o,2]),
              obstacle_r (f32 [M_o]), rho (f32),
              obstacle_states (f32 [M_o,4], schema 3+) — the [x, y, cosθ,
              sinθ] observations the GAT consumed.  θ is world-random and
              NOT derivable from obstacle_xy, so schema-2 shards need the
              seed_episode-replay reconstruction in
              ``collect_sibling_values.py`` to rebuild GAT inputs.

Cost: an expanded node is ~1.2 KB + ~6.4 KB of child states (23 × 280 B);
a cap-5000 A* plan is ~1–1.5 MB compressed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TraceRecorder:
    """
    Collects one plan's expansions and writes one ``.npz`` per planning call.

    Wire-up (see ``BestFirstSearch14``): ``begin_plan`` at ``run()`` entry,
    ``record_expansion`` inside ``expand()``, ``end_plan`` at every return.
    ``set_episode`` is called by the eval loop at episode boundaries;
    ``decision_index`` counts planning calls within an episode.

    Args:
        out_dir: shard directory (created; ``meta.json`` written once).
        meta:    JSON-serialisable system constants for the whole shard —
                 actuation matrix, move groups, precise groups/config,
                 coupled flag, cost parameters, algo, caps, code stamp.
    """

    def __init__(self, out_dir: str | Path, meta: dict) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.out_dir / "meta.json"
        if not meta_path.exists():
            meta_path.write_text(json.dumps(meta, indent=2, default=_jsonable))
        self.episode = -1
        self.seed = -1
        self.decision_index = -1
        self._active = False

    # -- episode context (eval loop) ------------------------------------

    def set_episode(self, episode: int, seed: int) -> None:
        self.episode = int(episode)
        self.seed = int(seed)
        self.decision_index = -1

    # -- plan lifecycle (search) ----------------------------------------

    def begin_plan(self, model, ms) -> None:
        self.decision_index += 1
        self._active = True
        self._rows: dict[int, int] = {}      # id(BFSNode) -> node row
        self._nodes: list[tuple] = []        # (parent_row, parent_aidx, g, h, depth)
        self._poses: list[np.ndarray] = []
        self._last: list[np.ndarray] = []
        self._branches: list[tuple] = []
        self._bposes: list[np.ndarray | None] = []
        self._blast: list[np.ndarray | None] = []
        self._goals = np.asarray(model.goals, dtype=np.float32)
        self._obstacle_xy = np.asarray(model.geom.obstacle_xy, dtype=np.float32)
        self._obstacle_r = np.asarray(model.geom.obstacle_r, dtype=np.float32)
        self._rho = float(model.geom.rho)
        self._obstacle_states = np.asarray(model.obstacle_states, dtype=np.float32)

    def record_expansion(
        self, node, branch_rows: list[tuple], branch_states: list | None = None
    ) -> None:
        """
        Register ``node`` (a ``BFSNode`` being expanded) and its branch table.

        ``branch_rows`` entries: ``(mode, group, pgroup, step_cost, safe,
        clearance, child_g, child_h, child_terminal, child_collision)`` —
        one per stub, in stub order (so branch indices match the search's).
        ``branch_states`` (parallel to ``branch_rows``): the materialised
        child ``ModelState`` per stub, or ``None`` where no child exists
        (refuted coarse vet).
        """
        if not self._active:
            return
        parent_row, parent_aidx = self._node_ref(node)
        row = len(self._nodes)
        self._rows[id(node)] = row
        self._nodes.append(
            (parent_row, parent_aidx, float(node.g), float(node.h), int(node.depth))
        )
        self._poses.append(np.asarray(node.ms.poses, dtype=np.float32))
        self._last.append(np.asarray(node.ms.last_actions, dtype=np.float32))
        if branch_states is None:
            branch_states = [None] * len(branch_rows)
        for b, cms in zip(branch_rows, branch_states):
            self._branches.append((row, *b))
            self._bposes.append(
                None if cms is None else np.asarray(cms.poses, dtype=np.float32)
            )
            self._blast.append(
                None if cms is None
                else np.asarray(cms.last_actions, dtype=np.float32)
            )

    def end_plan(self, result, goal_node) -> None:
        """
        Write the plan's shard file.

        ``result`` is the search's ``PlanResult``; ``goal_node`` the
        ``BFSNode`` the returned decisions lead to (goal, best generated goal,
        or best partial), or ``None``.
        """
        if not self._active:
            return
        self._active = False
        goal_parent, goal_aidx = (
            self._node_ref(goal_node) if goal_node is not None else (-1, -1)
        )
        n = np.asarray(self._nodes, dtype=np.float64).reshape(-1, 5)
        b = self._branches
        path = (
            self.out_dir
            / f"plan_ep{self.episode:04d}_d{self.decision_index:03d}.npz"
        )
        np.savez_compressed(
            path,
            parent_row=n[:, 0].astype(np.int32),
            parent_aidx=n[:, 1].astype(np.int32),
            g=n[:, 2].astype(np.float32),
            h=n[:, 3].astype(np.float32),
            depth=n[:, 4].astype(np.int32),
            poses=np.stack(self._poses) if self._poses else np.zeros((0, 0, 3), np.float32),
            last_actions=np.stack(self._last) if self._last else np.zeros((0, 0, 2), np.float32),
            br_node=np.array([r[0] for r in b], dtype=np.int32),
            br_mode=np.array([r[1] for r in b], dtype=np.int8),
            br_group=np.array([r[2] for r in b], dtype=np.int16),
            br_pgroup=np.array([r[3] for r in b], dtype=np.int16),
            br_step_cost=np.array([r[4] for r in b], dtype=np.float32),
            br_safe=np.array([r[5] for r in b], dtype=bool),
            br_clearance=np.array([r[6] for r in b], dtype=np.float32),
            br_child_g=np.array([r[7] for r in b], dtype=np.float32),
            br_child_h=np.array([r[8] for r in b], dtype=np.float32),
            br_child_terminal=np.array([r[9] for r in b], dtype=bool),
            br_child_collision=np.array([r[10] for r in b], dtype=bool),
            br_child_poses=_stack_optional(self._bposes, self._poses, 3),
            br_child_last=_stack_optional(self._blast, self._last, 2),
            solved=np.bool_(result.solved),
            cap_hit=np.bool_(result.cap_hit),
            plan_cost=np.float32(result.plan_cost),
            goal_parent_row=np.int32(goal_parent),
            goal_aidx=np.int32(goal_aidx),
            expansions=np.int32(result.expansions),
            n_coarse_vets=np.int32(result.n_coarse_vets),
            n_precise_expansions=np.int32(result.n_precise_expansions),
            episode=np.int32(self.episode),
            seed=np.int32(self.seed),
            decision_index=np.int32(self.decision_index),
            goals=self._goals,
            obstacle_xy=self._obstacle_xy,
            obstacle_r=self._obstacle_r,
            rho=np.float32(self._rho),
            obstacle_states=self._obstacle_states,
        )

    # -- helpers ---------------------------------------------------------

    def _node_ref(self, node) -> tuple[int, int]:
        """(parent_row, branch index) addressing ``node`` as its parent's child."""
        if node.parent is None:
            return -1, -1
        parent_row = self._rows.get(id(node.parent), -1)
        return parent_row, int(getattr(node, "aidx", -1))


def _stack_optional(
    items: list, node_arrays: list, dim: int
) -> np.ndarray:
    """
    Stack per-branch child arrays into ``(B, N, dim)`` f32, NaN rows where a
    branch materialised no child.  ``node_arrays`` supplies N (robot count).
    """
    n = node_arrays[0].shape[0] if node_arrays else 0
    out = np.full((len(items), n, dim), np.nan, dtype=np.float32)
    for i, x in enumerate(items):
        if x is not None:
            out[i] = x
    return out


def load_plan(path: str | Path) -> dict:
    """Load one plan shard into a plain dict of arrays/scalars."""
    with np.load(Path(path)) as z:
        return {k: z[k] for k in z.files}


def on_path_rows(plan: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Root→goal path over *expanded* node rows, from the stored goal reference.

    Returns:
        rows:  (L,) int array of node rows on the path, root first.  The goal
               state itself is a generated child (never expanded, no row).
        aidx:  (L,) branch index chosen at each row (the prior target).
    """
    rows, aidx = [], []
    r = int(plan["goal_parent_row"])
    a = int(plan["goal_aidx"])
    while r >= 0:
        rows.append(r)
        aidx.append(a)
        a = int(plan["parent_aidx"][r])
        r = int(plan["parent_row"][r])
    return np.asarray(rows[::-1], dtype=np.int64), np.asarray(aidx[::-1], dtype=np.int64)


def value_labels(plan: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    (rows, y) — on-path node rows and their exact in-model remaining cost
    ``plan_cost − g(row)``.  Valid only for solved plans (empty otherwise).
    """
    if not bool(plan["solved"]):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    rows, _ = on_path_rows(plan)
    y = float(plan["plan_cost"]) - plan["g"][rows]
    return rows, y.astype(np.float32)


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    raise TypeError(f"not JSON-serialisable: {type(x)}")
