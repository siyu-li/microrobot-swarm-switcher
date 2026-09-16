"""
Shared machinery for the lazy tree search: branch stubs and value normalisation.

The economic premise (Danihelka et al., ICLR 2022) is an asymmetry: *ranking*
actions is cheap (one prior forward pass covers all edges), *evaluating* them
is expensive (a coarse vet or a precise rollout per edge).  Accordingly a
:class:`Branch` starts as a **stub** — mode, group and exact step cost, all
known without touching the model — and its transition is materialised only when
the search actually descends the edge.  Contrast ``capswitcher.rl.search.common``,
whose eager ``expand`` vets every group and rolls out precise at every node.

Step costs are exact at stub creation (the group's configured constant from the
``SwitcherCost`` table / ``precise_unit × n_unreached × selection_interval``),
so selection can price edges before buying their transitions.

A shield-refuted coarse edge is an *illegal action discovered late*: it is
pruned from the node's legal set — no Q value, no collision penalty.  So is
any edge — coarse or precise — whose materialised child lands in collision
(``tree.materialize``); the modes keep different margins (``d_safe`` on the
coarse swept vet, zero for precise) but a colliding action is pruned either
way.  The ``COLLISION_COST`` sentinel therefore survives only for the
*entry* state of a node and for **dead ends** — nodes every one of whose
edges has been pruned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from robot_nav.models.MARL.capswitcher.rl.reward import COARSE, PRECISE

# Dominating cost for a transition into a predicted-collision state — larger
# than any realistic cost-to-go, so the search shuns it without pruning.
COLLISION_COST: float = 1e9


@dataclass
class Branch:
    """
    One action edge from a node — created as a stub, filled on materialisation.

    ``frames`` / ``candidate`` stay ``None`` until the edge is materialised
    (and ``frames`` stays ``None`` for the precise edge, which has no vetted
    frame plan).  The child node lives on the owning ``Node.children`` slot.
    """

    mode: int                 # COARSE (0) or PRECISE (1)
    group: int | None         # coarse move-group id, else None
    step_cost: float          # exact per-decision motion cost (known at stub time)
    pgroup: int | None = field(default=None)  # precise-group id (configs B/C), None = precise-all
    frames: list | None = field(default=None)     # vetted coarse frames (post-mat.)
    candidate: object | None = field(default=None)  # shield CoarseCandidate (post-mat.)
    # Goal-distance reduction (m) the edge buys, filled on materialisation by
    # the *eager* tree only (``tree_eager.expand_node_eager``): for coarse it
    # copies the shield candidate's predicted progress, for precise it is
    # measured on the rolled-out child.  Stays ``None`` throughout the lazy
    # tree, whose prior is forbidden to see materialisation output at all.
    progress: float | None = field(default=None)


def expand_stubs(model, ms) -> list[Branch]:
    """
    Build the branch stubs for ``ms``: every selectable coarse group plus the
    precise edges (last).  No model transition is run — this is the cheap half
    of an expansion; the prior forward pass is the other half.

    Precise edges depend on the model's configuration (redesign §3):

    * ``model.precise_groups is None`` — the single legacy precise-all edge.
    * otherwise — one edge per precise group that still has an unreached
      member (a fully-arrived group would be a no-op edge; it is simply not
      an action at this node, mirroring how a refuted coarse edge is not one).

    Stub *existence* is not legality: a precise edge whose rollout collides is
    pruned on materialisation like any other, so a node's legal set can go
    empty (a dead end) — see ``tree.materialize``.
    """
    branches = [
        Branch(mode=COARSE, group=g, step_cost=model.step_cost(COARSE, ms, g))
        for g in model.coarse.selectable_groups()
    ]
    pgs = getattr(model, "precise_groups", None)
    if pgs is None:
        branches.append(
            Branch(mode=PRECISE, group=None, step_cost=model.step_cost(PRECISE, ms))
        )
    else:
        for pg in range(len(pgs)):
            if model.driven_members(ms, pg).size == 0:
                continue
            branches.append(
                Branch(
                    mode=PRECISE,
                    group=None,
                    step_cost=model.step_cost(PRECISE, ms, pgroup=pg),
                    pgroup=pg,
                )
            )
    return branches


class QNormalizer:
    """
    Running min–max normaliser mapping cost-to-go values to a higher-is-better
    q ∈ [0, 1] (the MuZero/Gumbel trick; keeps ``σ`` budget-independent).

    Collision-dominated costs (``>= COLLISION_COST``) never update the bounds
    and always normalise to 0 (worst).  Before two distinct finite values have
    been seen, every finite cost normalises to 0.5.
    """

    def __init__(self) -> None:
        self.vmin = float("inf")
        self.vmax = float("-inf")

    def update(self, cost: float) -> None:
        if cost >= COLLISION_COST:
            return
        self.vmin = min(self.vmin, cost)
        self.vmax = max(self.vmax, cost)

    def normalize(self, cost: float) -> float:
        if cost >= COLLISION_COST:
            return 0.0
        if not (self.vmax > self.vmin):
            return 0.5
        q = (self.vmax - cost) / (self.vmax - self.vmin)
        return float(min(max(q, 0.0), 1.0))
