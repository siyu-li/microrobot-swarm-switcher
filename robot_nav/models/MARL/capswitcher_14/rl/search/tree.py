"""
Lazy search tree for the 14-robot Gumbel AlphaZero switcher.

Same problem class as the 6-robot searches — deterministic, known-model,
min-cost — but the expansion economics are inverted to match the Gumbel
premise: **expanding a node is cheap** (branch stubs + one prior forward pass),
**materialising an edge is the budgeted resource** (one coarse vet or one
precise rollout, ``model.n_transitions``).  Design consequences:

* **Materialise-on-descent.**  ``Node.children`` slots start ``None``; the
  first traversal of an edge buys its transition.  For a coarse edge the
  shield vet *is* the transition, so verify-on-descent costs nothing extra.
  A refuted vet **prunes** the edge from the node's legal set — an illegal
  action discovered late, with no Q value and no collision penalty.

* **Collision prunes both modes.**  An edge whose materialised child is in
  collision is pruned exactly like a refuted coarse vet — no Q value, and no
  ``COLLISION_COST`` arm left standing to soak up Sequential Halving visits.
  The two modes are held to *different margins*, deliberately: coarse must
  clear ``d_safe`` over its swept path (the shield), precise only has to not
  actually collide (zero margin, checked at every sub-step by the rollout,
  which truncates on contact so the returned child state *is* the colliding
  one).  Precise robots pass close by design; a ``d_safe`` margin there would
  prune almost every precise edge.

* **Dead ends.**  Pruning precise edges too means a node's legal set *can* go
  empty — every action out of it collides.  Such a node is a dead end,
  valued at ``COLLISION_COST``; descents that reach it back up immediately
  without buying anything.  Dead-endness is **not** propagated as a prune:
  the edge into a dead end is *dominated* (worst possible q, never selected)
  but stays legal, because in a lazy tree "every child collides" is only ever
  known for the edges actually bought — a transitive prune would be asserting
  more than the search has paid to learn.  An empty legal set at the *root*
  is different: there every edge is materialised eagerly, so it really does
  mean no executable action exists.  The search reports that
  (``unsolvable``) and the caller ends the episode.

* **Completed-Q (v_mix).**  With children materialised lazily, most edges have
  no value estimate.  Where a policy over edges is needed, unmaterialised
  legal edges are *completed* with the mixed value estimate of Danihelka et
  al. (2022): an interpolation between the node's own value and the
  prior-weighted mean of the materialised edges' values.  Evidence adjusts an
  edge's standing; absence of evidence leaves it at the prior's.

* **Bellman (value-replacement) backup over materialised edges only**:
  ``q_hat = min(materialised edge values)``, falling back to the leaf estimate
  ``h`` while nothing is materialised.  ``h`` never sits in the min next to
  real edge values — a low-lying estimate would otherwise stick as the node's
  value whenever its edge is soundly never bought (see ``bellman_refresh``).

* **Certificates** (exact upper bounds ``U`` from reached-goal suffixes) and
  the positive-step-cost branch-and-bound prune carry over unchanged from the
  6-robot tree — both restricted to materialised children.
"""

from __future__ import annotations

import numpy as np

from robot_nav.models.MARL.capswitcher.rl.reward import PRECISE
from robot_nav.models.MARL.capswitcher_14.rl.search.common import (
    COLLISION_COST,
    Branch,
    QNormalizer,
    expand_stubs,
)

INF = float("inf")


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


class Node:
    """One search-tree node over a :class:`ModelState`."""

    __slots__ = (
        "ms", "terminal", "collision", "h", "q_hat", "U",
        "branches", "children", "legal", "N", "W", "prior_logits",
    )

    def __init__(self, ms, terminal: bool, collision: bool, h: float) -> None:
        self.ms = ms
        self.terminal = terminal
        self.collision = collision
        self.h = h                     # leaf cost-to-go estimate at creation
        self.q_hat = h                 # Bellman estimate (refined as edges appear)
        self.U = 0.0 if terminal else INF  # certified upper bound on cost-to-go
        self.branches: list[Branch] | None = None
        self.children: list[Node | None] | None = None
        self.legal: np.ndarray | None = None   # bool per edge; False = pruned
        self.N: np.ndarray | None = None       # per-edge visit counts
        self.W: np.ndarray | None = None       # per-edge return sums (diagnostics)
        self.prior_logits: np.ndarray | None = None

    @property
    def expanded(self) -> bool:
        return self.branches is not None

    def value(self) -> float:
        """Effective cost-to-go: the estimate capped by the certificate."""
        return min(self.q_hat, self.U)

    def legal_actions(self) -> list[int]:
        return [a for a in range(len(self.branches)) if self.legal[a]]

    def materialized_actions(self) -> list[int]:
        return [
            a for a in range(len(self.branches))
            if self.legal[a] and self.children[a] is not None
        ]

    @property
    def precise_action(self) -> int:
        # Legacy single-precise-edge assumption; prefer ``is_precise`` when
        # multiple precise edges (configs B/C) may exist.
        return len(self.branches) - 1

    def is_precise(self, a: int) -> bool:
        return self.branches[a].mode == PRECISE

    @property
    def dead_end(self) -> bool:
        """Expanded, but every edge is pruned — no action leads out of here."""
        return self.branches is not None and not self.legal.any()


def make_node(model, ms) -> Node:
    """Create (and leaf-evaluate) a node for ``ms``."""
    if model.all_reached(ms):
        return Node(ms, terminal=True, collision=False, h=0.0)
    if model.collision_pred(ms):
        return Node(ms, terminal=False, collision=True, h=COLLISION_COST)
    return Node(ms, terminal=False, collision=False, h=float(model.cost_to_go(ms)))


def expand_node(node: Node, model, prior) -> None:
    """
    The *cheap* expansion: branch stubs for every edge + one prior forward pass.
    No model transition runs here — children stay unmaterialised.
    """
    node.branches = expand_stubs(model, node.ms)
    k = len(node.branches)
    node.children = [None] * k
    node.legal = np.ones(k, dtype=bool)
    node.N = np.zeros(k, dtype=np.int64)
    node.W = np.zeros(k, dtype=np.float64)
    node.prior_logits = np.asarray(
        prior(model, node.ms, node.branches), dtype=np.float64
    )


def materialize(node: Node, a: int, model) -> Node | None:
    """
    Buy edge ``a``'s transition (the budgeted operation).

    Coarse: run the single-group vet.  If the shield refutes the move the edge
    is pruned from the legal set and ``None`` is returned — the vet's cost was
    still charged to ``model.n_coarse_vets``.  The candidate is stashed on the
    branch either way (root label harvesting).  Precise: run the rollout,
    which truncates at the first colliding sub-step.

    Either way, an edge whose child lands **in collision** is pruned too — a
    coarse move whose non-member robots clip something the member-only shield
    sweep does not cover, or a precise rollout that ran into contact.  The two
    modes keep different margins on purpose (``d_safe`` for the coarse swept
    vet, zero for precise), but the verdict is the same: a colliding action is
    not an action.  The child is left on the ``children`` slot for inspection;
    ``materialized_actions`` filters on ``legal``, so it stays out of the
    Bellman min and out of every policy over edges.
    """
    branch = node.branches[a]
    if branch.mode == PRECISE:
        next_ms = (
            model.precise_next(node.ms) if branch.pgroup is None
            else model.precise_group_next(node.ms, branch.pgroup)
        )
        child = make_node(model, next_ms)
    else:
        mv = model.coarse_move(node.ms, branch.group)
        branch.candidate = mv.candidate
        if not mv.candidate.safe:
            node.legal[a] = False
            return None
        branch.frames = mv.candidate.frames
        child = make_node(model, mv.next_state)
    node.children[a] = child
    if child.collision:
        node.legal[a] = False
        return None
    return child


def edge_value(node: Node, a: int) -> float:
    """
    Cost of taking the *materialised* edge ``a`` then acting well below it:
    the Bellman estimate capped by the certificate.  A predicted-collision
    child is charged the dominating ``COLLISION_COST``.
    """
    child = node.children[a]
    if child.collision:
        return COLLISION_COST
    est = node.branches[a].step_cost + child.q_hat
    cert = node.branches[a].step_cost + child.U
    return min(est, cert)


def bellman_refresh(node: Node) -> None:
    """
    Recompute ``q_hat`` / ``U`` from the **materialised** children only;
    ``h`` holds the node's value only while nothing is materialised.

    ``h`` must not stay in the min alongside materialised edges: it is an
    estimate, and an estimate that lies *low* would stick as the node's value
    whenever the edge it stands for is (soundly) never bought — e.g. a precise
    edge excluded by the branch-and-bound prune.  Restricting the min to real
    edge values keeps the saturated tree exact; unmaterialised alternatives
    are represented where they belong — in the completed-Q policy, not in the
    Bellman value.
    """
    if node.dead_end:
        # Every edge out of here was refuted or collides: this state is a
        # dead end, priced at the dominating collision cost so the edge into
        # it loses to any alternative.  It is not pruned at the parent — see
        # the module docstring on why dead-endness does not propagate.
        node.q_hat = COLLISION_COST
        node.U = INF
        return
    mat = node.materialized_actions()
    values = [edge_value(node, a) for a in mat]
    node.q_hat = min(values) if values else node.h
    node.U = min(
        (
            node.branches[a].step_cost + node.children[a].U
            for a in mat
            if not node.children[a].collision
        ),
        default=INF,
    )


def completed_q(node: Node, qnorm: QNormalizer) -> np.ndarray:
    """
    Per-edge normalised (higher-is-better) q with the paper's completion.

    Materialised edges use their real (certificate-capped) values.
    Unmaterialised legal edges get the mixed value estimate::

        v_mix = (v_node + N_total · Σ_mat π(a) q(a) / Σ_mat π(a)) / (1 + N_total)

    computed in normalised space (``v_node`` = the node's own effective value,
    ``π`` = softmax of the prior logits over legal edges).  Pruned edges get
    ``-inf`` so no selection rule can pick them.
    """
    k = len(node.branches)
    out = np.full(k, -np.inf, dtype=np.float64)

    legal = node.legal_actions()
    mat = node.materialized_actions()

    raw = {a: edge_value(node, a) for a in mat}
    for v in raw.values():
        qnorm.update(v)
    v_node = node.value()
    qnorm.update(v_node)

    q_mat = {a: qnorm.normalize(raw[a]) for a in mat}
    v_node_n = qnorm.normalize(v_node)

    if mat:
        logits = np.full(k, -np.inf, dtype=np.float64)
        logits[legal] = node.prior_logits[legal]
        pi = _softmax(logits[np.array(legal, dtype=int)])
        pi_by_action = dict(zip(legal, pi))
        w = sum(pi_by_action[a] for a in mat)
        q_bar = (
            sum(pi_by_action[a] * q_mat[a] for a in mat) / w if w > 0.0
            else v_node_n
        )
        n_total = float(node.N.sum())
        v_mix = (v_node_n + n_total * q_bar) / (1.0 + n_total)
    else:
        v_mix = v_node_n

    for a in legal:
        out[a] = q_mat[a] if a in q_mat else v_mix
    return out


def refresh_path(path: list[tuple[Node, int]]) -> None:
    """
    Bellman-refresh every node along ``path`` (root-first), deepest first,
    **without** incrementing visit counts — used after a materialisation was
    refuted: the transition was paid but no leaf was reached and no action was
    actually taken at the deepest node.
    """
    for node, _ in reversed(path):
        bellman_refresh(node)


def backup(path: list[tuple[Node, int]], leaf: Node) -> None:
    """
    Back a completed descent up ``path`` (root-first list of (node, action)):
    increment visits, accumulate returns (diagnostics), Bellman-refresh.
    Certificates propagate inside :func:`bellman_refresh`.
    """
    G = leaf.value()
    for node, a in reversed(path):
        node.N[a] += 1
        G = node.branches[a].step_cost + G
        node.W[a] += G
        bellman_refresh(node)


def simulate(
    root: Node,
    model,
    select_action,
    prior,
    first_action: int | None = None,
) -> None:
    """
    One descent: select down from ``root``, materialise the first unbought
    edge met (at most one transition), back up.

    ``select_action(node, g)`` picks a legal edge given the exact accumulated
    path cost ``g``.  ``first_action`` forces the root edge (Sequential
    Halving).  Descents that reach a terminal / predicted-collision / dead-end
    node buy nothing but still back up their visit so selection statistics
    move on.  A materialisation the shield refutes — or one whose child
    collides — prunes the edge and refreshes the path without a visit (see
    :func:`refresh_path`).
    """
    path: list[tuple[Node, int]] = []
    node = root
    g = 0.0

    while True:
        if node.terminal or node.collision:
            backup(path, node)
            return
        if not node.expanded:
            # Cheap: stubs + one prior forward pass (e.g. a root child that was
            # eagerly materialised but not yet visited).
            expand_node(node, model, prior)
        if node.dead_end:
            # Every edge here has been pruned; there is nothing left to buy.
            bellman_refresh(node)
            backup(path, node)
            return
        a = first_action if node is root and first_action is not None else \
            select_action(node, g)
        first_action = None if node is root else first_action
        child = node.children[a]
        if child is None:
            child = materialize(node, a, model)
            if child is None:                       # shield refuted → pruned
                refresh_path(path + [(node, a)])
                return
            if not (child.terminal or child.collision):
                expand_node(child, model, prior)    # cheap: stubs + prior
            path.append((node, a))
            backup(path, child)
            return
        path.append((node, a))
        g += node.branches[a].step_cost
        node = child
