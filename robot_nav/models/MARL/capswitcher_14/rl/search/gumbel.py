"""
Gumbel AlphaZero switcher over the lazy tree — planning-only (Danihelka et
al., ICLR 2022), 14-robot instantiation.

Division of labour per real decision:

* **Root — eager.**  Every coarse group is vetted by the real shield (22
  single-group vets) and the precise edge rolled out, so the root's legal set
  is exact, the executed action always carries exactly vetted frames, and all
  22 clearance labels are harvested for the feasibility head's dataset.  This
  costs 23 transitions of the budget — charged honestly.
* **Below the root — lazy.**  Nodes expand to stubs + one prior forward pass;
  transitions are bought one per descent (``tree.simulate``).  Selection uses
  the deterministic rule ``argmax π'(a) − N(a)/(1+ΣN)`` with π' built from
  **completed** Q-values (v_mix for unmaterialised edges), the prior's
  *predicted* feasibility mask steering it away from likely-refuted coarse
  edges.  The real vet corrects a wrong prediction on contact by pruning.
* **Root bandit.**  Gumbel-top-m over the legal root edges, Sequential Halving
  by ``gum(a) + logits(a) + σ(q(a))``, act with the survivor.  ``σ(q) =
  (c_visit + max_b N(b)) · c_scale · q`` on min–max-normalised negated costs.
* **Distillation output.**  The improved root policy ``π' = softmax(logits +
  σ(completedQ))`` over legal edges is returned on every decision — the
  root-only distillation target for the learned prior (settled: interior
  nodes are not harvested).
* **Unsolvable states.**  Colliding edges are pruned in both modes
  (``tree.materialize``), so the root's legal set can empty out: every coarse
  group refuted or colliding *and* every precise rollout running into
  contact.  There is then no action to execute, and the search says so —
  ``decision["unsolvable"]`` — rather than executing a known collision.  The
  harnesses end the episode there and score it ``UNSOLVABLE``.
* **Stall break.**  Independently of the search, the switcher watches the
  root value across real decisions.  If it has not dropped by ``δ`` over the
  last ``k`` decisions the formation is cycling rather than converging (the
  arrive/un-arrive churn diagnosed on the timeout episodes), and the decision
  is forced onto the best legal *precise* edge, which is the mode that can
  actually resolve the robots the coarse table keeps trading against each
  other.  ``δ = 100``, ``k = 5`` by default; ``stall_steps=0`` disables it.

Budget = ``model.n_transitions`` (single-group vets + precise rollouts),
including the root's 23 and any vet the shield refutes.

Known property (inherent to the AlphaZero family, sharpened by laziness): the
non-root rule is *exploitative* — with the paper's ``σ`` scaling, π' is
near-argmax, so a subtree whose leaf estimate is pessimistic (overestimates
cost) may never be materialised and its value never corrected downward.
Optimistic errors self-correct on contact; pessimistic ones persist.  Root
arms are protected by Sequential Halving's forced visits.  Full-budget
equivalence with exhaustive minimin therefore holds when leaf estimates are
exact (see the randomized test), not for arbitrary estimates.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from robot_nav.models.MARL.capswitcher.rl.cost import SwitcherCost
from robot_nav.models.MARL.capswitcher_14.rl.forward_model import (
    ForwardModel14,
    build_forward_model,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.common import (
    COLLISION_COST,
    QNormalizer,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.priors import UniformPrior
from robot_nav.models.MARL.capswitcher_14.rl.search.tree import (
    Node,
    _softmax,
    bellman_refresh,
    completed_q,
    edge_value,
    expand_node,
    make_node,
    materialize,
    simulate,
)


def _state_seed(poses: np.ndarray, salt: int) -> int:
    """Stable 32-bit seed from rounded poses (reproducible planning per state)."""
    key = np.ascontiguousarray(np.round(poses, 6)).tobytes()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return (int.from_bytes(digest, "little") ^ (salt * 0x9E3779B1)) & 0xFFFFFFFF


class GumbelAlphaZero14:
    """
    One search instance (fresh per real decision, receding horizon).

    Args:
        prior:        Base policy ``prior(model, ms, branches) -> logits`` over
                      stubs; may expose ``feasibility(...) -> bool array``.
        budget:       Max model transitions (the root's eager 23 count).
        m:            Max root actions sampled without replacement by
                      Gumbel-top-m (clipped to the legal count).
        c_visit, c_scale: The ``σ`` transform constants (paper defaults).
        gumbel_scale: Scale of the root Gumbel noise; 0 = deterministic root
                      (equivalence tests).
        seed:         Mixed with a state hash per decision.
    """

    def __init__(
        self,
        prior,
        budget: int = 100,
        m: int = 16,
        c_visit: float = 50.0,
        c_scale: float = 1.0,
        gumbel_scale: float = 1.0,
        seed: int = 0,
        lazy_root: bool = False,
    ) -> None:
        self.prior = prior
        self.budget = int(budget)
        self.m = int(m)
        self.c_visit = float(c_visit)
        self.c_scale = float(c_scale)
        self.gumbel_scale = float(gumbel_scale)
        self.seed = int(seed)
        # Vet-on-demand root: instead of eagerly materialising (exactly
        # vetting) all 23 root edges, vet only the Gumbel-top-m arms —
        # ordered by prior logits with the predicted-feasibility mask first —
        # replacing any arm the real shield refutes with the next candidate.
        # Executed actions still carry real vetted frames (an arm must
        # survive its vet to enter the bandit), so the safety contract is
        # unchanged; what is lost is the full 22-label harvest per root.
        self.lazy_root = bool(lazy_root)

    def _sigma(self, q: np.ndarray, node: Node) -> np.ndarray:
        return (self.c_visit + float(node.N.max())) * self.c_scale * q

    def _pi_prime(self, node: Node, qnorm: QNormalizer) -> np.ndarray:
        """π' = softmax(logits + σ(completedQ)) over legal edges (0 elsewhere)."""
        cq = completed_q(node, qnorm)
        legal = node.legal_actions()
        scores = node.prior_logits[legal] + self._sigma(cq[legal], node)
        out = np.zeros(len(node.branches), dtype=np.float64)
        out[legal] = _softmax(scores)
        return out

    def run(
        self, model: ForwardModel14, ms, force_precise: bool = False
    ) -> dict:
        root = make_node(model, ms)
        expand_node(root, model, self.prior)

        k = len(root.branches)
        logits = root.prior_logits

        rng = np.random.default_rng(
            (_state_seed(ms.poses, 1) ^ (self.seed * 0x9E3779B1)) & 0xFFFFFFFF
        )
        gum = rng.gumbel(size=k) * self.gumbel_scale

        qnorm = QNormalizer()

        # Predicted-feasibility cache (advisory mask from the prior, if any).
        feas_fn = getattr(self.prior, "feasibility", None)
        feas_cache: dict[int, np.ndarray] = {}

        def predicted_feasible(node: Node) -> np.ndarray | None:
            if feas_fn is None:
                return None
            key = id(node)
            if key not in feas_cache:
                feas_cache[key] = np.asarray(
                    feas_fn(model, node.ms, node.branches), dtype=bool
                )
            return feas_cache[key]

        if not self.lazy_root:
            # Root: eager exact verification of every edge (the safety
            # contract and the label harvest).  Unsafe coarse edges are
            # pruned here, before the bandit ever sees them.
            for a in range(k):
                materialize(root, a, model)
            bellman_refresh(root)
            legal = root.legal_actions()
            if not legal:
                # Every coarse group refuted or colliding and every precise
                # rollout in contact: no executable action exists here.
                return self._unsolvable(root, model)
            active = sorted(
                legal, key=lambda a: gum[a] + logits[a], reverse=True
            )[: min(self.m, len(legal))]
        else:
            # Vet-on-demand root: order arms by Gumbel-perturbed logits with
            # predicted-feasible (and precise) arms first, then vet arms in
            # that order until m survive, the candidates run out, or the
            # budget is spent.  A refuted arm is pruned (its vet was charged)
            # and the next candidate takes its slot.
            order = sorted(range(k), key=lambda a: gum[a] + logits[a],
                           reverse=True)
            feas0 = predicted_feasible(root)
            if feas0 is not None:
                trusted = [a for a in order
                           if root.is_precise(a) or feas0[a]]
                order = trusted + [a for a in order if a not in trusted]
            active = []
            for a in order:
                if (len(active) >= self.m
                        or model.n_transitions >= self.budget):
                    break
                if materialize(root, a, model) is not None:
                    active.append(a)
            bellman_refresh(root)
            if not active:
                # Every arm we could afford to vet was refuted.  With the
                # whole stub set exhausted this root is genuinely dead.
                return self._unsolvable(root, model)

        def select_action(node: Node, g: float) -> int:
            legal_n = node.legal_actions()
            allowed = [
                a for a in legal_n
                if g + node.branches[a].step_cost < root.U
            ] or legal_n
            feas = predicted_feasible(node)
            if feas is not None:
                # Advisory mask: applies only to unmaterialised coarse edges
                # (real vet results and the precise edge always stand).
                trusted = [
                    a for a in allowed
                    if node.children[a] is not None
                    or node.is_precise(a)
                    or feas[a]
                ]
                allowed = trusted or allowed
            pi = self._pi_prime(node, qnorm)
            total = float(node.N.sum())
            gap = pi - node.N / (1.0 + total)
            return max(allowed, key=lambda a: gap[a])

        def root_score(a: int) -> float:
            v = edge_value(root, a)
            qnorm.update(v)
            sig = self._sigma(np.array([qnorm.normalize(v)]), root)[0]
            return gum[a] + logits[a] + sig

        # ---- Sequential Halving over the active root arms ------------------
        m = len(active)
        phases = max(1, math.ceil(math.log2(m)) if m > 1 else 1)
        attempts, max_attempts = 0, 16 * max(self.budget, 1)
        for phase in range(phases):
            if model.n_transitions >= self.budget:
                break
            remaining = max(self.budget - model.n_transitions, 0)
            remaining_phases = phases - phase
            visits = max(1, remaining // (remaining_phases * len(active)))
            for a in active:
                for _ in range(visits):
                    if model.n_transitions >= self.budget or attempts >= max_attempts:
                        break
                    simulate(root, model, select_action, self.prior, first_action=a)
                    attempts += 1
            if len(active) > 1:
                active.sort(key=root_score, reverse=True)
                active = active[: max(1, math.ceil(len(active) / 2))]
        winner = max(active, key=root_score)

        # Stall break (see the module docstring): override the bandit's choice
        # with the cheapest legal precise edge.  The search still ran in full —
        # its root vets are the safety contract and the label harvest, and its
        # value is what the stall window is measured on.
        forced_precise = False
        if force_precise:
            precise_legal = [
                a for a in root.legal_actions() if root.is_precise(a)
            ]
            if self.lazy_root:
                # A lazily-rooted precise edge may still be a stub — the
                # override needs a real (vetted/rolled-out) edge to execute.
                for a in list(precise_legal):
                    if root.children[a] is None:
                        if materialize(root, a, model) is None:
                            precise_legal.remove(a)
                precise_legal = [a for a in precise_legal
                                 if root.children[a] is not None]
            if precise_legal:
                winner = min(precise_legal, key=lambda a: edge_value(root, a))
                forced_precise = True

        pi_prime = self._pi_prime(root, qnorm)

        b = root.branches[winner]
        return {
            "mode": b.mode,
            "group": b.group,
            "pgroup": b.pgroup,
            "frames": b.frames,
            # All 22 coarse candidates (safe and refuted) — clearance labels.
            "candidates": [
                br.candidate for br in root.branches if br.candidate is not None
            ],
            "value": float(edge_value(root, winner)),
            # Distillation targets (root-only, settled).
            "pi_prime": pi_prime,
            "prior_logits": np.asarray(logits, dtype=np.float64),
            "legal": root.legal.copy(),
            "actions": [(br.mode, br.group, br.pgroup) for br in root.branches],
            "n_transitions": model.n_transitions,
            "unsolvable": False,
            "forced_precise": forced_precise,
        }

    @staticmethod
    def _unsolvable(root: Node, model: ForwardModel14) -> dict:
        """
        Decision dict for a root with no legal action.  Carries no ``frames``
        and no ``mode`` — nothing is executable, so nothing is executed; the
        caller checks ``unsolvable`` before stepping the env.  The harvested
        coarse candidates are still returned (they are valid clearance labels
        whatever the outcome), as is the transition count, so an unsolvable
        decision is charged honestly to the budget accounting.
        """
        return {
            "unsolvable": True,
            "forced_precise": False,
            "mode": None,
            "group": None,
            "pgroup": None,
            "frames": None,
            "candidates": [
                br.candidate for br in root.branches if br.candidate is not None
            ],
            "value": COLLISION_COST,
            "pi_prime": np.zeros(len(root.branches), dtype=np.float64),
            "prior_logits": np.asarray(root.prior_logits, dtype=np.float64),
            "legal": root.legal.copy(),
            "actions": [(br.mode, br.group, br.pgroup) for br in root.branches],
            "n_transitions": model.n_transitions,
        }


class GumbelSwitcher14:
    """
    Drop-in switcher running lazy Gumbel AlphaZero planning per real decision.

    Same ``decide(robot_state) -> dict`` contract as the 6-robot switchers, so
    it plugs into ``SwitcherEnv.step(mode, group=..., frames=...)`` and the
    eval harness.  The decision dict additionally carries ``pi_prime`` /
    ``prior_logits`` / ``legal`` / ``actions`` / ``n_transitions`` for
    phase-0+ logging, plus two flags the caller must respect:

    * ``unsolvable`` — the root had no legal action (see
      :meth:`GumbelAlphaZero14._unsolvable`).  ``mode`` and ``frames`` are
      ``None``; **do not step the env** — end the episode.
    * ``forced_precise`` — the stall break fired and overrode the bandit.

    **Stall break.**  The switcher watches a per-decision progress signal
    and, when it has been flat over the last ``stall_steps`` decisions,
    forces the next decision onto the best legal precise edge.  The signal is
    selected by ``stall_signal``:

    * ``"value"`` — root value must drop by ``stall_delta`` (the original
      rule; its δ is in leaf-value units, so it is *not* scale-free across
      leaf evaluators — 2026-09-03 diagnosis measured ~1.7 false fires per
      healthy episode at δ=100/k=5);
    * ``"sumd"`` — Σd over unreached robots must drop by ``stall_eps``
      metres (physical progress; scale-free, and the diagnosis' tournament
      winner at k=12: 13× fewer false fires than the value rule);
    * ``"both"`` — fire only when *both* are flat (conjunction).
    The window is cleared whenever it fires (so the break has a duty cycle
    rather than latching on) and by :meth:`reset`, which callers must invoke
    at every ``env.reset()`` — otherwise one episode's value trace leaks into
    the next and the detector compares across a discontinuity.

    Args mirror ``capswitcher``'s ``MPCSwitcher`` (minus ``depth``/``alpha``),
    plus the search hyper-parameters and the stall-break knobs
    (``stall_delta`` / ``stall_steps``; ``stall_steps=0`` disables).
    """

    def __init__(
        self,
        backbone,
        coarse,
        sim,
        prior=None,
        budget: int = 100,
        m: int = 16,
        c_visit: float = 50.0,
        c_scale: float = 1.0,
        gumbel_scale: float = 1.0,
        seed: int = 0,
        lazy_root: bool = False,
        d_safe: float = 0.3,
        selection_interval: int = 5,
        goal_threshold: float = 0.3,
        cost: SwitcherCost | None = None,
        default_rho: float = 0.2,
        leaf_value=None,
        feature_builder=None,
        feature_builder_v2=None,
        coupling=None,
        precise_groups: list | None = None,
        stall_delta: float = 100.0,
        stall_steps: int = 5,
        stall_signal: str = "value",
        stall_eps: float = 0.15,
    ) -> None:
        if stall_signal not in ("value", "sumd", "both"):
            raise ValueError(f"unknown stall_signal {stall_signal!r}")
        if cost is None:
            raise ValueError(
                "GumbelSwitcher14 requires a SwitcherCost (load "
                "cost_14robots.yaml with SwitcherCost.from_yaml)"
            )
        self.coupling = coupling
        self.precise_groups = precise_groups
        self.backbone = backbone
        self.coarse = coarse
        self.sim = sim
        # Optional GroupFeatureBuilder: when set, every decision dict carries
        # the root's (group_feats, global_feats) — phase-0 shard logging.
        self.feature_builder = feature_builder
        # Optional GroupFeatureBuilderV2: same hook for the split
        # feasibility/policy shards (feas_feats / policy_feats /
        # global_feats_v2).
        self.feature_builder_v2 = feature_builder_v2
        self.d_safe = float(d_safe)
        self.selection_interval = int(selection_interval)
        self.goal_threshold = float(goal_threshold)
        self.cost = cost
        self.default_rho = float(default_rho)
        self.leaf_value = leaf_value
        self.search = GumbelAlphaZero14(
            prior=prior if prior is not None else UniformPrior(),
            budget=budget, m=m, c_visit=c_visit, c_scale=c_scale,
            gumbel_scale=gumbel_scale, seed=seed, lazy_root=lazy_root,
        )
        # Per-decision transition counts (budget accounting for eval), split
        # by mode so evaluated-action effort is comparable with the best-first
        # baselines' coarse-vet / precise-rollout accounting.
        self.decision_transitions: list[int] = []
        self.decision_coarse_vets: list[int] = []
        self.decision_precise_rollouts: list[int] = []
        # Stall break: root values within the current episode, and how often
        # the break fired (reported by the eval harnesses).
        self.stall_delta = float(stall_delta)
        self.stall_steps = int(stall_steps)
        self.stall_signal = stall_signal
        self.stall_eps = float(stall_eps)
        self.value_history: list[float] = []
        self.sumd_history: list[float] = []
        self.n_stall_breaks = 0

    def reset(self) -> None:
        """Clear per-episode state.  Call at every ``env.reset()``."""
        self.value_history.clear()
        self.sumd_history.clear()

    @staticmethod
    def _flat(hist: list[float], k: int, min_drop: float) -> bool:
        if len(hist) <= k:
            return False
        return hist[-k - 1] - hist[-1] < min_drop

    def _stalled(self) -> bool:
        """
        True when the selected progress signal has been flat over the last
        ``stall_steps`` decisions — the formation is paying cost without
        buying progress (the arrive/un-arrive cycling seen on timeouts).
        """
        k = self.stall_steps
        if k <= 0:
            return False
        v_flat = self._flat(self.value_history, k, self.stall_delta)
        d_flat = self._flat(self.sumd_history, k, self.stall_eps)
        if self.stall_signal == "value":
            return v_flat
        if self.stall_signal == "sumd":
            return d_flat
        return v_flat and d_flat

    def _build_model(self, robot_state: np.ndarray) -> ForwardModel14:
        return build_forward_model(
            self.backbone,
            self.coarse,
            self.sim,
            robot_state,
            d_safe=self.d_safe,
            selection_interval=self.selection_interval,
            goal_threshold=self.goal_threshold,
            cost=self.cost,
            default_rho=self.default_rho,
            leaf_value=self.leaf_value,
            coupling=self.coupling,
            precise_groups=self.precise_groups,
        )

    def decide(self, robot_state: np.ndarray) -> dict:
        model = self._build_model(robot_state)
        ms = ForwardModel14.state_from_robot_state(robot_state)
        decision = self.search.run(model, ms, force_precise=self._stalled())
        if decision["forced_precise"]:
            # Restart the windows: the break has a duty cycle, and the traces
            # after a forced precise move are not comparable with the ones
            # that triggered it.
            self.value_history.clear()
            self.sumd_history.clear()
            self.n_stall_breaks += 1
        elif not decision["unsolvable"]:
            self.value_history.append(float(decision["value"]))
            d = model.goal_distances(ms)
            self.sumd_history.append(float(d[d > self.goal_threshold].sum()))
        self.decision_transitions.append(model.n_transitions)
        self.decision_coarse_vets.append(model.n_coarse_vets)
        self.decision_precise_rollouts.append(model.n_precise_expansions)
        if self.feature_builder is not None:
            gf, glf = self.feature_builder(model, ms)
            decision["group_feats"] = gf
            decision["global_feats"] = glf
        if self.feature_builder_v2 is not None:
            f = self.feature_builder_v2(model, ms)
            decision["feas_feats"] = f["feas"]
            decision["policy_feats"] = f["policy"]
            decision["global_feats_v2"] = f["glob"]
        return decision
