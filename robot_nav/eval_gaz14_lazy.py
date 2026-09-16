"""
Evaluation + phase-0 collection harness for the 14-robot **lazy** Gumbel
switcher — GAZ14-L, the budget-matched counterpart of ``eval_gaz14_eager.py``
(GAZ14-E).

Runs ``GumbelSwitcher14`` (22 coarse groups + precise, lazy tree, budget in
model transitions) against the two mode-only baselines over matched seeded
episodes.  Only the root is eagerly vetted; below it, transitions are bought
one edge at a time and unmaterialised edges are priced by completed-Q — the
variable GAZ14-E holds against this one, at the same budget over the same
seeded episodes.

``build_env`` / ``run`` / ``RESULT_ROWS`` / ``_save_pi_targets`` live here and
are imported by ``eval_gaz14_eager.py``, ``iterate_gaz14.py``,
``render_gaz14.py`` and ``eval_feasibility_14.py``, so every 14-robot harness
measures the same thing the same way.

With ``--log-pi-targets`` every real decision is logged as training data for
the learned prior:

    features (root group/global), all 22 shield clearances (exact labels for
    the feasibility head), the legal mask, and the improved root policy π′
    (root-only distillation target, settled).

Phase 0 = ``--episodes N --budgets 100 --log-pi-targets data/pi_targets_14``
with the default uniform prior (high-budget teacher).  Iteration t trains a
prior with ``train_prior.py`` and re-collects with ``--prior-model <ckpt>``.

Usage (run on the GPU box — local irsim step crashes; see project memory):
    python -m robot_nav.eval_gaz14_lazy --episodes 100 --budgets 100  \
        --log-pi-targets data/pi_targets_14
    python -m robot_nav.eval_gaz14_lazy --episodes 100 --budgets 40 100 \
        --prior-model <ckpt> --value-model <ckpt>
    python -m robot_nav.eval_gaz14_lazy --episode 100 --budgets 115 \
        --prior-model runs/gaz14_value/cycle_05_prior/prior_best.pt \
        --value-model robot_nav/models/MARL/capswitcher/checkpoint/value_local/value_geometry.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from robot_nav.SIM_ENV.corridor_layout import CorridorLayout
from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE
from robot_nav.models.MARL.capswitcher.policies.gat_backbone import GATBackbone
from robot_nav.models.MARL.capswitcher.rl.cost import SwitcherCost
from robot_nav.models.MARL.capswitcher.rl.switcher_env import (
    CoarseHeadingNoise,
    SwitcherEnv,
    seed_episode,
)
from robot_nav.models.MARL.capswitcher_14.configs import (
    A_FULL,
    MOVE_GROUPS,
    PRECISE_CONFIGS,
    build_precise_groups,
    make_coarse_steering,
)
from robot_nav.models.MARL.capswitcher_14.policies.coarse_steering import (
    CoarseSteering14,
)
from robot_nav.models.MARL.capswitcher_14.policies.group_rotation import (
    GroupRotation,
)
from robot_nav.models.MARL.capswitcher_14.policies.precise_coupling import (
    PreciseCoupling,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.features import (
    GroupFeatureBuilder,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.features_v2 import (
    GroupFeatureBuilderV2,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.gumbel import GumbelSwitcher14

logger.disable("irsim")

COARSE, PRECISE = 0, 1

# Reported metrics, shared by this harness and the plan→distil loop.
#
# The headline is **cost per successful episode** — `avg cost/ep` mixes the
# solved and unsolved episodes together, so it moves with the success/timeout
# composition as much as with plan quality, and it is not comparable across
# cost tables at all.  The cost split below separates the flat per-group coarse
# constants from the `precise_unit` surcharge, which is the term that changes
# between cost configs.
RESULT_ROWS = [
    ("success rate",         "success_rate",        "{:.1%}"),
    ("collision rate",       "collision_rate",      "{:.1%}"),
    ("timeout rate",         "timeout_rate",        "{:.1%}"),
    ("unsolvable rate",      "unsolvable_rate",     "{:.1%}"),
    ("avg decisions/ep",     "avg_decisions",       "{:.1f}"),
    ("cost/success ep  <<",  "avg_cost_success",    "{:.0f}"),
    ("  coarse cost/ep",     "avg_coarse_cost",     "{:.0f}"),
    ("  precise cost/ep",    "avg_precise_cost",    "{:.0f}"),
    ("  precise share",      "precise_cost_share",  "{:.1%}"),
    ("precise fraction",     "precise_frac",        "{:.1%}"),
    ("coarse fraction",      "coarse_frac",         "{:.1%}"),
    ("safe-coarse avail.",   "safe_avail_frac",     "{:.1%}"),
    ("coarse breaches (!)",  "coarse_breach",       "{:d}"),
    ("precise breaches",     "precise_breach",      "{:d}"),
    ("stall breaks",         "stall_breaks",        "{:d}"),
    ("  driven robot",       "precise_breach_active",    "{:d}"),
    ("  bystander (!)",      "precise_breach_bystander", "{:d}"),
    ("avg transitions/dec",  "avg_transitions",     "{:.1f}"),
]

# Evaluated-action effort, split by mode — same units as the best-first
# baselines' rows (kept out of RESULT_ROWS: eval_gaz14_baselines appends its
# own copies of these keys and must not print them twice).
GAZ_EFFORT_ROWS = [
    ("coarse vets/ep",       "avg_coarse_vets",      "{:.0f}"),
    ("precise rollouts/ep",  "avg_precise_rollouts", "{:.0f}"),
    ("transitions/ep",       "avg_transitions_ep",   "{:.0f}"),
]

DEFAULT_BACKBONE_CKPT = (
    "robot_nav/models/MARL/marlTD3/checkpoint/"
    "Mar.04_obstacle_14robots_partial_inactive/"
    "TD3-MARL-obstacle-14robots-partial-inactive_epoch210"
)


def resolve_device(name: str) -> torch.device:
    """Resolve ``--device``; ``"auto"`` picks cuda when available, else cpu."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


DEFAULT_COST_CONFIG = "robot_nav/models/MARL/capswitcher_14/cost_14robots.yaml"


CORRIDOR_WORLD = "robot_nav/worlds/multi_robot_world_corridor_14robots.yaml"
SCATTERED_WORLD = "robot_nav/worlds/multi_robot_world_obstacle_14robots.yaml"


def add_layout_args(ap: argparse.ArgumentParser) -> None:
    """Add the shared ``--corridor`` knobs, so all three runners agree on them."""
    ap.add_argument("--corridor", action="store_true",
                    help="banded world: starts in one x-band, goals in the "
                         "opposite one, all obstacles in a band across the "
                         "middle (sparse -> dense -> sparse).  Default is the "
                         "scattered randomized-obstacle world.")
    ap.add_argument("--corridor-band", type=float, nargs=2, default=(0.40, 0.60),
                    metavar=("LO", "HI"),
                    help="obstacle band as fractions of the world width")
    ap.add_argument("--corridor-min-gap", type=float, default=1.0,
                    help="minimum free gap between two obstacles (m); 2*(rho+"
                         "d_safe) keeps the gate threadable by a coarse move")
    ap.add_argument("--corridor-bidirectional", action="store_true",
                    help="send half the robots right-to-left, so the swarm has "
                         "to interpenetrate inside the gate")
    ap.add_argument("--corridor-empty", action="store_true",
                    help="sanity corridor: park all obstacles in a strip along "
                         "the top wall, outside the traffic band — the arena "
                         "is effectively obstacle-free while the world keeps "
                         "the 7 obstacles the frozen backbone expects.  "
                         "Implies --corridor.")
    ap.add_argument("--corridor-aligned", action="store_true",
                    help="sanity corridor: robots start on evenly spaced "
                         "y-lanes facing their direction of travel, and each "
                         "goal is pinned to its robot's start y, so the "
                         "nominal solution is 'drive straight'.  Implies "
                         "--corridor.")


def layout_from_args(args: argparse.Namespace) -> CorridorLayout | None:
    """``CorridorLayout`` for :func:`build_env`, or ``None`` for the scattered world."""
    empty = getattr(args, "corridor_empty", False)
    aligned = getattr(args, "corridor_aligned", False)
    if not (getattr(args, "corridor", False) or empty or aligned):
        return None
    return CorridorLayout(
        band=(float(args.corridor_band[0]), float(args.corridor_band[1])),
        min_gap=args.corridor_min_gap,
        bidirectional=args.corridor_bidirectional,
        obstacle_free=empty,
        aligned_goals=aligned,
    )


def add_noise_args(ap: argparse.ArgumentParser) -> None:
    """
    Coarse-heading disturbance knobs (see ``CoarseHeadingNoise``), shared by
    this harness and ``eval_gaz14_baselines`` so both arms of the robustness
    comparison define the noise identically.  CLI takes degrees; the env works
    in radians.
    """
    ap.add_argument("--coarse-heading-bias-deg", type=float, default=0.0,
                    help="σ of the per-episode/per-robot heading bias (deg) "
                         "applied to executed coarse rotations; the bias is "
                         "the accumulating term (lateral drift ≈ sin δ m per "
                         "metre translated).  0 disables it")
    ap.add_argument("--coarse-heading-white-deg", type=float, default=0.0,
                    help="σ of the white heading component (deg), redrawn "
                         "per coarse control; roughening only — it largely "
                         "averages out along a path")
    ap.add_argument("--coarse-heading-walk-deg", type=float, default=0.0,
                    help="σ of a per-coarse-control random-walk step (deg) on "
                         "the bias, clipped to ±--coarse-heading-bias-max-deg")
    ap.add_argument("--coarse-heading-bias-max-deg", type=float, default=15.0,
                    help="clip on the random-walking bias (deg)")
    ap.add_argument("--noise-seed", type=int, default=777,
                    help="seed of the dedicated disturbance RNG (mixed with "
                         "the episode seed, never touching the global "
                         "np.random the world layout is drawn from) — the "
                         "same value gives every arm identical per-episode "
                         "bias vectors")


def noise_from_args(args: argparse.Namespace) -> CoarseHeadingNoise | None:
    """``CoarseHeadingNoise`` for :func:`build_env`, or ``None`` when clean."""
    bias = getattr(args, "coarse_heading_bias_deg", 0.0)
    white = getattr(args, "coarse_heading_white_deg", 0.0)
    walk = getattr(args, "coarse_heading_walk_deg", 0.0)
    if not (bias or white or walk):
        return None
    return CoarseHeadingNoise(
        bias_std=np.radians(bias),
        white_std=np.radians(white),
        walk_std=np.radians(walk),
        bias_max=np.radians(getattr(args, "coarse_heading_bias_max_deg", 15.0)),
        seed=args.noise_seed,
    )


def noise_meta(args: argparse.Namespace) -> dict | None:
    """Shard-JSON provenance of the disturbance (degrees, as given)."""
    if noise_from_args(args) is None:
        return None
    return {
        "bias_deg": args.coarse_heading_bias_deg,
        "white_deg": args.coarse_heading_white_deg,
        "walk_deg": args.coarse_heading_walk_deg,
        "bias_max_deg": args.coarse_heading_bias_max_deg,
        "noise_seed": args.noise_seed,
    }


def _make_coupling(coupled: bool | str | None):
    """Coupling object for a ``--coupled-precise`` value (falsy = legacy)."""
    if not coupled:
        return None
    if coupled is True or coupled == "pinv":
        return PreciseCoupling(A_FULL, ang_max=1.0)
    if coupled == "group":
        return GroupRotation(A_FULL, ang_max=1.0)
    raise ValueError(f"unknown coupling mode {coupled!r}; use 'pinv' or 'group'")


def build_env(
    device: torch.device, cost: SwitcherCost, goal_threshold: float,
    backbone_ckpt: str, disable_plotting: bool = True,
    layout: CorridorLayout | None = None,
    precise_config: str = "all", coupled: bool | str = False,
    heading_noise: CoarseHeadingNoise | None = None,
    max_decisions: int = 120,
) -> tuple[SwitcherEnv, CoarseSteering14, MARL_SIM_OBSTACLE]:
    """
    Construct 14-robot sim + backbone + coarse primitive + switcher env.

    Plotting is off by default so every evaluation path stays headless;
    ``render_gaz14.py`` turns it on to watch a policy live.

    ``layout`` selects the episode distribution: ``None`` is the scattered
    randomized-obstacle world every result so far was measured on; a
    ``CorridorLayout`` is the banded sparse→dense→sparse world, which keeps the
    same 14 robots and 7 obstacles (the frozen backbone's shapes) and only moves
    where they are drawn.

    ``precise_config`` / ``coupled`` (redesign §2–§3): the precise action set
    ("all" = legacy precise-all; "pairs"/"singles" = per-group edges) and the
    coupled-rotation physics fix — ``coupled`` is falsy (legacy independent
    rotation), ``"pinv"``/``True`` (minimum-norm ``pinv(A_S)`` coupling) or
    ``"group"`` (uniform block rotation, ``GroupRotation``).  Defaults keep
    the **legacy** behaviour so existing pipelines are bit-identical until the
    re-baseline is run deliberately; the built env carries ``env.coupling`` /
    ``env.precise_groups`` for switcher construction.
    """
    sim = MARL_SIM_OBSTACLE(
        world_file=CORRIDOR_WORLD if layout is not None else SCATTERED_WORLD,
        disable_plotting=disable_plotting,
        reward_phase=6,
        per_robot_goal_reset=False,
        obstacle_proximity_threshold=1.5,
        num_inactive_robots=0,
        layout=layout,
    )
    assert sim.num_robots == 14, f"expected 14 robots, world has {sim.num_robots}"
    backbone = GATBackbone(
        checkpoint_path=Path(backbone_ckpt),
        num_robots=sim.num_robots,
        num_obstacles=sim.num_obstacles,
        device=device,
        embedding_source="decoder",
    )
    coarse = make_coarse_steering(
        move_distance=cost.move_distances,
        method="nonlinear",
        nonlinear_solver="bfgs_lean",
        step_time=sim.env.step_time,
        ang_max=1.0,
        lin_max=0.5,
    )
    # Fail loudly if the cost YAML's group table drifted from the algebra.
    cost.validate_members(
        {g: [int(i) for i in coarse.members_of(g)]
         for g in coarse.selectable_groups()}
    )
    env = SwitcherEnv(
        sim=sim,
        backbone=backbone,
        coarse_steering=coarse,
        selection_interval=5,
        max_decisions=max_decisions,
        cost=cost,
        goal_threshold=goal_threshold,
        device=device,
        terminate_on_oob=False,
        coupling=_make_coupling(coupled),
        precise_groups=build_precise_groups(precise_config),
        heading_noise=heading_noise,
    )
    return env, coarse, sim


def run(
    env: SwitcherEnv, decide_fn, episodes: int, base_seed: int, policy=None,
    on_step=None, episode_log: list | None = None, on_episode=None,
) -> dict:
    """
    Run ``decide_fn`` for ``episodes`` seeded episodes and collect stats.

    ``policy`` (a ``GumbelSwitcher14``) supplies per-decision transition
    counts — the lazy budget unit; baselines report 0.

    ``on_step(ep, decision, step_cost, info, done)`` — optional observer called
    once per executed decision, after the env step, with the realised cost of
    that decision.  It exists so ``collect_leaf_data.py`` can harvest
    cost-to-go labels from the **same** episode loop the evaluation tables are
    produced by, rather than a parallel one that could drift from it.

    ``episode_log`` — optional list; when given, one plain-python record per
    episode (seed, outcome, decisions, cost split, transitions) is appended
    and a per-episode progress line is printed, so a seeded episode can be
    compared row-for-row across policies and the baseline planners.

    Episode cost is split into its two priced components so a run is readable
    independently of how precision happens to be priced:

    * ``coarse_cost``   — the flat per-group constants (the baseline of moving
      the formation at all);
    * ``precise_cost``  — the ``precise_unit`` surcharge for resolving robots
      one at a time.

    Collisions are attributed to the mode that was executing when the sim
    flagged them (``coarse_breach`` / ``precise_breach``, which sum to the
    collision count), and — for precise — to whether the flagged robot was the
    one being driven or a stationary bystander.
    """
    if policy is not None:
        policy.decision_transitions = []
        # GumbelSwitcher14 splits per-decision effort by mode (the best-first
        # switcher tracks its split through snapshot() instead).
        if hasattr(policy, "decision_coarse_vets"):
            policy.decision_coarse_vets = []
            policy.decision_precise_rollouts = []
    split_effort = policy is not None and hasattr(policy, "decision_coarse_vets")
    n = {"success": 0, "collision": 0, "timeout": 0, "unsolvable": 0}
    coarse_dec = precise_dec = total_dec = 0
    safe_available = coarse_breach = precise_breach = 0
    precise_breach_active = precise_breach_bystander = 0
    costs, lengths, success_costs = [], [], []
    coarse_costs, precise_costs = [], []

    for ep in range(episodes):
        seed = base_seed + ep
        seed_episode(env, seed)
        if on_episode is not None:
            # Episode-boundary hook (e.g. TraceRecorder.set_episode) so plans
            # recorded inside decide_fn carry their episode/seed identity.
            on_episode(ep, seed)

        if policy is not None and hasattr(policy, "reset"):
            policy.reset()          # per-episode switcher state (stall window)
        env.reset()
        done = False
        unsolvable = False
        ep_cost, ep_len = 0.0, 0
        ep_coarse_cost = ep_precise_cost = 0.0
        ep_coarse_dec = ep_precise_dec = 0
        dec_start = len(policy.decision_transitions) if policy is not None else 0
        info: dict = {}
        while not done:
            decision = decide_fn(env)
            cands = decision.get("candidates") or []
            if any(getattr(c, "safe", False) for c in cands):
                safe_available += 1
            if decision.get("unsolvable"):
                # No legal action at this state — every coarse group refuted
                # or colliding and every precise rollout in contact.  Nothing
                # is executed; the episode ends here.
                unsolvable = True
                break

            _, _, done, info = env.step(
                decision["mode"], group=decision["group"],
                frames=decision["frames"], pgroup=decision.get("pgroup"),
            )
            ep_len += 1
            total_dec += 1
            step_cost = float(info["path_cost"])
            ep_cost += step_cost
            if on_step is not None:
                on_step(ep, decision, step_cost, info, done)
            if decision["mode"] == COARSE:
                coarse_dec += 1
                ep_coarse_dec += 1
                ep_coarse_cost += step_cost
                if info["collision"]:
                    coarse_breach += 1
            else:
                precise_dec += 1
                ep_precise_dec += 1
                ep_precise_cost += step_cost
                if info["collision"]:
                    precise_breach += 1
                    # Precise drives exactly one robot at a time; the rest hold
                    # still.  If the flagged robot is the driven one, the frozen
                    # individual policy steered into something.
                    flagged = set(info.get("collision_robots") or [])
                    if info.get("active_robot") in flagged:
                        precise_breach_active += 1
                    else:
                        precise_breach_bystander += 1

        if unsolvable:
            n["unsolvable"] += 1
        else:
            if info.get("all_reached"):
                n["success"] += 1
                success_costs.append(ep_cost)
            if info.get("collision"):
                n["collision"] += 1
            if info.get("timeout"):
                n["timeout"] += 1
        costs.append(ep_cost)
        coarse_costs.append(ep_coarse_cost)
        precise_costs.append(ep_precise_cost)
        lengths.append(ep_len)

        if episode_log is not None:
            outcome = (
                "UNSOLVABLE" if unsolvable
                else "SUCCESS" if info.get("all_reached")
                else "COLLISION" if info.get("collision")
                else "TIMEOUT" if info.get("timeout") else "ENDED"
            )
            ep_transitions = (
                int(sum(policy.decision_transitions[dec_start:]))
                if policy is not None else 0
            )
            rec = {
                "seed": seed,
                "outcome": outcome,
                "decisions": ep_len,
                "cost": round(ep_cost, 3),
                "coarse_cost": round(ep_coarse_cost, 3),
                "precise_cost": round(ep_precise_cost, 3),
                "coarse_dec": ep_coarse_dec,
                "precise_dec": ep_precise_dec,
                "transitions": ep_transitions,
            }
            if split_effort:
                rec["coarse_vets"] = int(
                    sum(policy.decision_coarse_vets[dec_start:])
                )
                rec["precise_rollouts"] = int(
                    sum(policy.decision_precise_rollouts[dec_start:])
                )
            episode_log.append(rec)
            print(
                f"  ep {ep:3d} seed {seed}: {outcome:<9} dec={ep_len:<4} "
                f"cost={ep_cost:8.1f} (coarse {ep_coarse_cost:7.1f} / "
                f"precise {ep_precise_cost:7.1f}) "
                f"transitions={ep_transitions}",
                flush=True,
            )

    avg_cost = float(np.mean(costs))
    avg_coarse_cost = float(np.mean(coarse_costs))
    avg_precise_cost = float(np.mean(precise_costs))
    stats = {
        "episodes": episodes,
        "success_rate": n["success"] / episodes,
        "collision_rate": n["collision"] / episodes,
        "timeout_rate": n["timeout"] / episodes,
        # No legal action existed at some state: every edge collided.
        "unsolvable_rate": n["unsolvable"] / episodes,
        "avg_decisions": float(np.mean(lengths)),
        "avg_cost": avg_cost,
        # Headline: what a solved episode actually costs.  None when no episode
        # succeeded (cycle 1 of a cold run can legitimately have zero).
        "avg_cost_success": (
            float(np.mean(success_costs)) if success_costs else None
        ),
        "avg_coarse_cost": avg_coarse_cost,
        "avg_precise_cost": avg_precise_cost,
        "precise_cost_share": avg_precise_cost / max(avg_cost, 1e-9),
        "coarse_frac": coarse_dec / max(total_dec, 1),
        "precise_frac": precise_dec / max(total_dec, 1),
        "safe_avail_frac": safe_available / max(total_dec, 1),
        "coarse_breach": coarse_breach,
        "precise_breach": precise_breach,
        "precise_breach_active": precise_breach_active,
        "precise_breach_bystander": precise_breach_bystander,
        "avg_transitions": (
            float(np.mean(policy.decision_transitions))
            if policy is not None and policy.decision_transitions
            else 0.0
        ),
        "stall_breaks": int(getattr(policy, "n_stall_breaks", 0) or 0),
    }
    if split_effort:
        # Per-episode evaluated-action effort, split by mode — the same units
        # as the best-first baselines' coarse-vet / precise-rollout rows.
        stats["avg_coarse_vets"] = (
            sum(policy.decision_coarse_vets) / episodes
        )
        stats["avg_precise_rollouts"] = (
            sum(policy.decision_precise_rollouts) / episodes
        )
        stats["avg_transitions_ep"] = (
            sum(policy.decision_transitions) / episodes
        )
    return stats


# ---- decision sources -----------------------------------------------------

def _precise_only(env: SwitcherEnv) -> dict:
    return {"mode": PRECISE, "group": None, "frames": None, "candidates": []}


def _coarse_only(env: SwitcherEnv) -> dict:
    # group=None → SwitcherEnv picks a uniform-random selectable group.
    return {"mode": COARSE, "group": None, "frames": None, "candidates": []}


def _gumbel_decider(policy: GumbelSwitcher14, pi_log: list | None,
                    ep_box: dict | None = None):
    """Gumbel decider, optionally logging phase-0 prior-training samples.

    ``ep_box`` — mutable ``{"seed": int}`` updated by the ``run`` loop's
    ``on_episode`` hook, so every logged decision carries its episode seed
    (the train/val split unit: decisions within an episode are correlated).
    """

    def decide(env: SwitcherEnv) -> dict:
        decision = policy.decide(env._robot_state)
        # An unsolvable root has an empty legal set and an all-zero π′ — a
        # masked KL over it is undefined, so it is not a distillation sample.
        if pi_log is not None and not decision.get("unsolvable"):
            cands = decision.get("candidates") or []
            clearance = np.full(len(MOVE_GROUPS), np.nan, dtype=np.float32)
            safe = np.zeros(len(MOVE_GROUPS), dtype=bool)
            for c in cands:
                clearance[c.group] = c.clearance
                safe[c.group] = c.safe
            rec = {
                "robot_state": np.asarray(env._robot_state, dtype=np.float32),
                "group_feats": decision["group_feats"],
                "global_feats": decision["global_feats"],
                "pi_prime": np.asarray(decision["pi_prime"], dtype=np.float32),
                "prior_logits": np.asarray(
                    decision["prior_logits"], dtype=np.float32
                ),
                "legal": np.asarray(decision["legal"], dtype=bool),
                "clearance": clearance,
                "safe": safe,
                "n_transitions": decision["n_transitions"],
                "episode_seed": np.int64(
                    ep_box["seed"] if ep_box is not None else -1
                ),
            }
            # v2 split feasibility/policy features (feature_builder_v2 hook).
            if "feas_feats" in decision:
                rec["feas_feats"] = decision["feas_feats"]
                rec["policy_feats"] = decision["policy_feats"]
                rec["global_feats_v2"] = decision["global_feats_v2"]
            pi_log.append(rec)
        return decision

    return decide


# Episode outcome codes stored in the pi-target shards (int8).
OUTCOME_CODES = {"SUCCESS": 0, "TIMEOUT": 1, "COLLISION": 2,
                 "UNSOLVABLE": 3, "ENDED": 4}


class _ValueLabelCollector:
    """
    Attach realized cost-to-go + episode outcome to the pi-target records —
    the value residual's labels, harvested from the same episode loop as
    everything else (self-contained: no dependency on ``collect_leaf_data``).

    Alignment invariant: every non-unsolvable decision is logged by
    ``_gumbel_decider`` exactly once *and* executed exactly once, in order,
    so this episode's ``pi_log`` suffix and the step-cost list line up 1:1.
    An unsolvable decision ends the episode without being logged or executed,
    keeping the invariant.

    Wire ``on_episode`` / ``on_step`` into :func:`run`, and call
    :meth:`finalize` once after it returns (closes an episode that ended
    unsolvable, where no ``done`` step ever fires).
    """

    def __init__(self, pi_log: list) -> None:
        self.pi_log = pi_log
        self._start = 0
        self._costs: list[float] = []
        self._open = False

    def on_episode(self, ep: int, seed: int) -> None:
        self.finalize()                      # previous episode, if still open
        self._start = len(self.pi_log)
        self._costs = []
        self._open = True

    def on_step(self, ep, decision, step_cost, info, done) -> None:
        self._costs.append(float(step_cost))
        if done:
            outcome = ("SUCCESS" if info.get("all_reached")
                       else "COLLISION" if info.get("collision")
                       else "TIMEOUT" if info.get("timeout") else "ENDED")
            self.finalize(outcome)

    def finalize(self, outcome: str = "UNSOLVABLE") -> None:
        if not self._open:
            return
        self._open = False
        recs = self.pi_log[self._start:]
        if len(recs) != len(self._costs):
            print(f"WARNING: {len(recs)} logged decisions vs "
                  f"{len(self._costs)} executed steps — cost-to-go labels "
                  "for this episode are dropped")
            for r in recs:
                r["cost_to_go"] = np.float32(np.nan)
                r["outcome"] = np.int8(OUTCOME_CODES["ENDED"])
            return
        suffix = np.cumsum(np.asarray(self._costs, dtype=np.float64)[::-1])[::-1]
        code = np.int8(OUTCOME_CODES[outcome])
        for r, y in zip(recs, suffix):
            r["cost_to_go"] = np.float32(y)
            r["outcome"] = code


def _save_pi_targets(
    pi_log: list, out_dir: Path, name: str, d_safe: float,
    move_distances: np.ndarray | None = None,
    goal_threshold: float | None = None,
    alpha: float | None = None,
    pricing: tuple[float, float] | None = None,
) -> None:
    """Save one fixed-shape .npz shard per eval row."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.npz"
    arrays = dict(
        robot_states=np.stack([d["robot_state"] for d in pi_log]),
        group_feats=np.stack([d["group_feats"] for d in pi_log]),
        global_feats=np.stack([d["global_feats"] for d in pi_log]),
        pi_prime=np.stack([d["pi_prime"] for d in pi_log]),
        prior_logits=np.stack([d["prior_logits"] for d in pi_log]),
        legal=np.stack([d["legal"] for d in pi_log]),
        clearance=np.stack([d["clearance"] for d in pi_log]),
        safe=np.stack([d["safe"] for d in pi_log]),
        n_transitions=np.array([d["n_transitions"] for d in pi_log]),
        episode_seed=np.array([d["episode_seed"] for d in pi_log],
                              dtype=np.int64),
        d_safe=np.float32(d_safe),
    )
    if "feas_feats" in pi_log[0]:
        arrays["feas_feats"] = np.stack([d["feas_feats"] for d in pi_log])
        arrays["policy_feats"] = np.stack([d["policy_feats"] for d in pi_log])
        arrays["global_feats_v2"] = np.stack(
            [d["global_feats_v2"] for d in pi_log])
    if move_distances is not None:
        arrays["move_distances"] = np.asarray(move_distances, dtype=np.float32)
    if goal_threshold is not None:
        arrays["goal_threshold"] = np.float32(goal_threshold)
    if "cost_to_go" in pi_log[0]:
        arrays["cost_to_go"] = np.array(
            [d["cost_to_go"] for d in pi_log], dtype=np.float32)
        arrays["outcome"] = np.array(
            [d["outcome"] for d in pi_log], dtype=np.int8)
    if alpha is not None:
        arrays["alpha"] = np.float32(alpha)
    if pricing is not None:
        arrays["precise_unit"] = np.float32(pricing[0])
        arrays["selection_interval"] = np.float32(pricing[1])
    np.savez_compressed(path, **arrays)
    print(f"Saved {len(pi_log)} prior-training samples to {path}")


# ---------------------------------------------------------------------------
# Worker shards: one JSON per (budget, seed block); merge exactly.
#
# Same trick as ``eval_gaz14_baselines``: episodes are independently seeded
# (episode k = --seed + k), so disjoint seed blocks in separate processes run
# exactly the episodes a single big run would.  The Gumbel policy's noise is
# seeded per decision from a state hash mixed with ``--policy-seed``, so
# workers sharing one policy seed reproduce the single-run decisions exactly.
# Every mean/rate below is recoverable as a sum given its denominator, so the
# merged table equals the single big run's.
# ---------------------------------------------------------------------------

_INT_KEYS = ("coarse_breach", "precise_breach", "precise_breach_active",
             "precise_breach_bystander", "stall_breaks")

# Shard keys that must match across workers for a merged row to make sense.
_CONFIG_KEYS = ("budget", "m", "policy_seed", "prior_model", "lazy_root",
                "feas_policy_dir", "value_model", "value_residual",
                "stall_delta", "stall_steps", "stall_signal", "stall_eps",
                "heading_noise")


def _to_counts(stats: dict) -> dict:
    """Undo the per-shard averaging: means/rates -> sums with denominators."""
    n = stats["episodes"]
    dec = stats["avg_decisions"] * n
    successes = round(stats["success_rate"] * n)
    return {
        "episodes": n,
        "decisions": dec,
        "successes": successes,
        "collisions": round(stats["collision_rate"] * n),
        "timeouts": round(stats["timeout_rate"] * n),
        # .get: shards written before collision pruning have no such key.
        "unsolvables": round(stats.get("unsolvable_rate", 0.0) * n),
        "cost": stats["avg_cost"] * n,
        "coarse_cost": stats["avg_coarse_cost"] * n,
        "precise_cost": stats["avg_precise_cost"] * n,
        "success_cost": (stats["avg_cost_success"] or 0.0) * successes,
        "coarse_dec": stats["coarse_frac"] * dec,
        "precise_dec": stats["precise_frac"] * dec,
        "safe_avail": stats["safe_avail_frac"] * dec,
        "transitions_dec": stats["avg_transitions"] * dec,
        # .get: shards written before the mode-split effort counters have none.
        "coarse_vets": stats.get("avg_coarse_vets", 0.0) * n,
        "precise_rollouts": stats.get("avg_precise_rollouts", 0.0) * n,
        "transitions_ep": stats.get("avg_transitions_ep", 0.0) * n,
        **{k: int(stats.get(k, 0) or 0) for k in _INT_KEYS},
    }


def _from_counts(c: dict) -> dict:
    """Re-average merged sums into a stats dict for :func:`print_table`."""
    n = c["episodes"]
    dec = max(c["decisions"], 1)
    return {
        "episodes": n,
        "success_rate": c["successes"] / n,
        "collision_rate": c["collisions"] / n,
        "timeout_rate": c["timeouts"] / n,
        "unsolvable_rate": c["unsolvables"] / n,
        "avg_decisions": c["decisions"] / n,
        "avg_cost": c["cost"] / n,
        "avg_cost_success": (
            c["success_cost"] / c["successes"] if c["successes"] else None
        ),
        "avg_coarse_cost": c["coarse_cost"] / n,
        "avg_precise_cost": c["precise_cost"] / n,
        "precise_cost_share": c["precise_cost"] / max(c["cost"], 1e-9),
        "coarse_frac": c["coarse_dec"] / dec,
        "precise_frac": c["precise_dec"] / dec,
        "safe_avail_frac": c["safe_avail"] / dec,
        "avg_transitions": c["transitions_dec"] / dec,
        "avg_coarse_vets": c["coarse_vets"] / n,
        "avg_precise_rollouts": c["precise_rollouts"] / n,
        "avg_transitions_ep": c["transitions_ep"] / n,
        **{k: int(round(c[k])) for k in _INT_KEYS},
    }


def _save_shard(out_dir: Path, algo: str, label: str, budget: int,
                args: argparse.Namespace, stats: dict,
                per_episode: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{algo}_s{args.seed}_e{args.episodes}.json"
    path.write_text(json.dumps(
        {
            "algo": algo,
            "label": label,
            "seed": args.seed,
            "episodes": args.episodes,
            "budget": budget,
            "m": args.m,
            "policy_seed": (
                args.policy_seed if args.policy_seed is not None else args.seed
            ),
            "prior_model": args.prior_model,
            "lazy_root": args.lazy_root,
            "feas_policy_dir": args.feas_policy_dir,
            "value_model": args.value_model,
            "value_residual": args.value_residual,
            "stall_delta": args.stall_delta,
            "stall_steps": args.stall_steps,
            "stall_signal": args.stall_signal,
            "stall_eps": args.stall_eps,
            "heading_noise": noise_meta(args),
            "stats": stats,
            "per_episode": per_episode,
        },
        indent=2,
    ))
    print(f"Saved shard: {path}")


def merge_shards(shard_dir: Path) -> dict[str, dict]:
    """
    Combine all ``*.json`` worker shards in ``shard_dir`` into one table, and
    write ``per_episode.csv`` — one row per (policy, seeded episode) — for
    episode-level comparison against the baseline planners.
    """
    shards = [json.loads(p.read_text()) for p in sorted(shard_dir.glob("*.json"))]
    if not shards:
        raise SystemExit(f"no .json shards in {shard_dir}")

    results: dict[str, dict] = {}
    csv_rows: list[dict] = []
    for algo in dict.fromkeys(s["algo"] for s in shards):
        group = sorted((s for s in shards if s["algo"] == algo),
                       key=lambda s: s["seed"])
        for a, b in zip(group, group[1:]):
            if a["seed"] + a["episodes"] > b["seed"]:
                print(f"WARNING: {algo} shards s{a['seed']} and s{b['seed']} "
                      "overlap — episodes double-counted.")
        for key in _CONFIG_KEYS:
            if len({str(s.get(key)) for s in group}) > 1:
                print(f"WARNING: {algo} shards disagree on {key} — "
                      "the merged row mixes configurations.")
        counts = [_to_counts(s["stats"]) for s in group]
        merged = {k: sum(c[k] for c in counts) for k in counts[0]}
        label = group[0]["label"]
        results[label] = _from_counts(merged)
        for s in group:
            for rec in s.get("per_episode", []):
                csv_rows.append({"algo": label, **rec})
        blocks = ", ".join(f"s{s['seed']}+{s['episodes']}" for s in group)
        print(f"{label}: {len(group)} shard(s) [{blocks}] -> "
              f"{merged['episodes']} episodes")

    if csv_rows:
        csv_path = shard_dir / "per_episode.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(sorted(csv_rows, key=lambda r: (r["algo"], r["seed"])))
        print(f"Wrote {len(csv_rows)} episode rows: {csv_path}")
    return results


def print_table(results: dict[str, dict], rows: list | None = None) -> None:
    """Side-by-side comparison of the result dicts."""
    names = list(results)
    header = f"{'metric':<24}" + "".join(f"{nm:>14}" for nm in names)
    print("\n" + header)
    print("-" * len(header))
    for label, key, fmt in (RESULT_ROWS if rows is None else rows):
        line = f"{label:<24}"
        for nm in names:
            v = results[nm].get(key)
            line += f"{'—' if v is None else fmt.format(v):>14}"
        print(line)
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--budgets", type=int, nargs="+", default=[100],
                    help="transition budgets per decision (root's eager 23 count)")
    ap.add_argument("--m", type=int, default=16,
                    help="Gumbel-top-m root actions sampled without replacement")
    ap.add_argument("--gumbel-scale", type=float, default=1.0)
    ap.add_argument("--c-visit", type=float, default=50.0)
    ap.add_argument("--c-scale", type=float, default=1.0)
    ap.add_argument("--d-safe", type=float, default=0.3)
    ap.add_argument("--cost-config", type=str, default=DEFAULT_COST_CONFIG,
                    help="SwitcherCost YAML (per-group move_distance + cost, "
                         "precise_unit)")
    ap.add_argument("--goal-threshold", type=float, default=0.3)
    ap.add_argument("--max-decisions", type=int, default=120,
                    help="episode decision budget before timeout — changing "
                         "it changes the timeout metric; rows are only "
                         "comparable at equal budgets")
    ap.add_argument("--backbone-ckpt", type=str, default=DEFAULT_BACKBONE_CKPT)
    ap.add_argument("--device", type=str, default="auto",
                    help="torch device for the backbone / prior / value nets "
                         "('auto' = cuda when available, else cpu)")
    ap.add_argument("--baselines", action="store_true",
                    help="also run precise-only and coarse-only baselines")
    ap.add_argument("--prior-model", type=str, default=None,
                    help="learned PriorNet checkpoint (train_prior.py); "
                         "default is the uniform phase-0 prior")
    ap.add_argument("--feas-policy-dir", type=str, default=None,
                    help="directory holding feas_best.pt + policy_best.pt "
                         "(train_feas_policy14.py) — the split "
                         "feasibility/scheduling prior; mutually exclusive "
                         "with --prior-model")
    ap.add_argument("--feas-threshold", type=float, default=0.1,
                    help="advisory in-tree mask: coarse edges with predicted "
                         "P(feasible) below this are steered around "
                         "(--feas-policy-dir only)")
    ap.add_argument("--lazy-root", action="store_true",
                    help="vet-on-demand root: vet only the Gumbel-top-m arms "
                         "(predicted-feasible first) instead of all 22+1 — "
                         "frees ~23 transitions of budget per decision; "
                         "executed actions still carry real vetted frames")
    ap.add_argument("--stall-delta", type=float, default=100.0,
                    help="stall break: minimum root-value drop required over "
                         "--stall-steps decisions before the switcher forces a "
                         "precise action")
    ap.add_argument("--stall-steps", type=int, default=5,
                    help="stall break: window in decisions (0 disables)")
    ap.add_argument("--stall-signal", choices=("value", "sumd", "both"),
                    default="value",
                    help="stall break signal: root-value drop (deployed), "
                         "physical Σd drop over unreached robots "
                         "(scale-free), or their conjunction")
    ap.add_argument("--stall-eps", type=float, default=0.15,
                    help="minimum Σd drop (m) counted as progress "
                         "(sumd/both signals)")
    ap.add_argument("--feas-margin", type=float, default=0.0,
                    help="advisory in-tree feasibility threshold (m) on the "
                         "predicted clearance margin")
    ap.add_argument("--value-model", type=str, default=None,
                    help="learned cost-to-go checkpoint (train_value.py)")
    ap.add_argument("--value-residual", type=str, default=None,
                    help="ValueResidualNet checkpoint "
                         "(train_value_residual14.py): leaf = analytic h "
                         "modulated ±25%% from the v2 global context; "
                         "mutually exclusive with --value-model")
    ap.add_argument("--log-pi-targets", type=str, default=None,
                    help="directory for harvests the training data for learning prior")
    ap.add_argument("--precise-config", choices=list(PRECISE_CONFIGS),
                    default="all",
                    help="precise action set (redesign §3): 'all' = legacy "
                         "single precise-all edge; 'pairs'/'singles' = one "
                         "edge per 2-/1-robot precise group")
    ap.add_argument("--coupled-precise", nargs="?", const="pinv",
                    default=None, choices=["pinv", "group"],
                    help="physics fix (redesign §2): precise rotation is "
                         "realised through the actuation matrix.  'pinv' "
                         "(the bare-flag default) = minimum-norm coupling, "
                         "all bystanders side-rotate; 'group' = the driven "
                         "robot's fixed size-7 block rotates uniformly with "
                         "it.  Off = legacy independent rotation")
    ap.add_argument("--policy-seed", type=int, default=None,
                    help="seed for the Gumbel policy's per-decision noise "
                         "(default: --seed).  Give all parallel workers the "
                         "same value so the sharded run reproduces a single "
                         "big run exactly")
    ap.add_argument("--out", type=str, default=None,
                    help="directory for per-(budget, seed-block) JSON shards "
                         "(summary stats + per-episode records) — run "
                         "disjoint seed blocks in parallel workers, then "
                         "combine with --merge")
    ap.add_argument("--merge", type=str, default=None,
                    help="merge the worker shards in this directory into one "
                         "table + per_episode.csv and exit (no episodes run)")
    add_layout_args(ap)
    add_noise_args(ap)
    args = ap.parse_args()

    if args.merge:
        results = merge_shards(Path(args.merge))
        print_table(results, rows=RESULT_ROWS + GAZ_EFFORT_ROWS)
        return

    if args.log_pi_targets and args.precise_config != "all":
        ap.error("--log-pi-targets assumes the fixed 23-edge action set; "
                 "it is only valid with --precise-config all (the new "
                 "trace collector replaces it for pairs/singles)")
    if args.prior_model and args.precise_config != "all":
        ap.error("the current PriorNet emits fixed 22+1 logits; a learned "
                 "prior with --precise-config pairs/singles needs the "
                 "redesigned token net (design doc §5)")
    if args.feas_policy_dir and args.prior_model:
        ap.error("--feas-policy-dir and --prior-model are mutually "
                 "exclusive priors")
    if args.feas_policy_dir and args.precise_config != "all":
        ap.error("the feas/policy prior emits fixed 22+1 logits; "
                 "--precise-config pairs/singles is unsupported")
    if args.value_residual and args.value_model:
        ap.error("--value-residual and --value-model are mutually "
                 "exclusive leaf evaluators")

    device = resolve_device(args.device)
    cost = SwitcherCost.from_yaml(args.cost_config)
    layout = layout_from_args(args)
    heading_noise = noise_from_args(args)
    env, coarse, sim = build_env(
        device, cost=cost, goal_threshold=args.goal_threshold,
        backbone_ckpt=args.backbone_ckpt, layout=layout,
        precise_config=args.precise_config, coupled=args.coupled_precise,
        heading_noise=heading_noise, max_decisions=args.max_decisions,
    )
    world_desc = "scattered" if layout is None else (
        "corridor "
        + ("empty" if layout.obstacle_free else str(layout.band))
        + (" aligned" if layout.aligned_goals else "")
        + (" bidirectional" if layout.bidirectional else "")
    )
    print(
        f"World: {world_desc}\n"
        f"Env: {sim.num_robots} robots, {sim.num_obstacles} obstacles, "
        f"{len(MOVE_GROUPS)} coarse groups, cost_config={args.cost_config}, "
        f"precise_unit={cost.precise_unit}, d_safe={args.d_safe}, "
        f"budgets={args.budgets}, prior={args.prior_model or 'uniform'}, "
        f"device={device}"
    )
    if heading_noise is not None:
        print(f"Coarse heading noise: {noise_meta(args)}")

    feature_builder = GroupFeatureBuilder(MOVE_GROUPS)
    feature_builder_v2 = GroupFeatureBuilderV2(
        MOVE_GROUPS, move_distances=coarse.move_distances,
        goal_threshold=args.goal_threshold,
    )

    leaf_value = None
    if args.value_model:
        from robot_nav.models.MARL.capswitcher.rl.value_net import LearnedCostToGo

        leaf_value = LearnedCostToGo(args.value_model, device=device)
        print(f"Learned leaf: feature={leaf_value.feature}, "
              f"precise_cost={leaf_value.precise_cost}")
    elif args.value_residual:
        from robot_nav.models.MARL.capswitcher_14.rl.search.value_residual import (
            LearnedValueResidual,
        )

        # CPU on purpose: one tiny forward per leaf — GPU round-trips lose.
        leaf_value = LearnedValueResidual(
            args.value_residual, feature_builder_v2, device="cpu")
        print(f"Value residual leaf: {args.value_residual} "
              f"(base={leaf_value.net.base}, band ±{leaf_value.net.band:.0%})")

    prior = None
    if args.prior_model:
        from robot_nav.models.MARL.capswitcher_14.rl.search.prior_net import (
            LearnedPrior,
            PriorNet,
        )

        prior = LearnedPrior(
            PriorNet.load(args.prior_model, map_location=device),
            feature_builder,
            feas_margin=args.feas_margin,
            device=device,
        )
        print(f"Learned prior: {args.prior_model} (feas_margin={args.feas_margin})")
    elif args.feas_policy_dir:
        from robot_nav.models.MARL.capswitcher_14.rl.search.feas_policy_net import (
            FeasNet14,
            GroupPolicyNet14,
            LearnedFeasPolicyPrior,
        )

        fp_dir = Path(args.feas_policy_dir)
        prior = LearnedFeasPolicyPrior(
            FeasNet14.load(fp_dir / "feas_best.pt", map_location=device),
            GroupPolicyNet14.load(fp_dir / "policy_best.pt",
                                  map_location=device),
            feature_builder_v2,
            feas_threshold=args.feas_threshold,
            device=device,
        )
        print(f"Feas/policy prior: {fp_dir} "
              f"(feas_threshold={args.feas_threshold})")

    results: dict[str, dict] = {}

    if args.baselines:
        print(f"\nRunning precise-only for {args.episodes} episodes ...")
        results["precise"] = run(env, _precise_only, args.episodes, args.seed)
        print(f"Running coarse-only for {args.episodes} episodes ...")
        results["coarse"] = run(env, _coarse_only, args.episodes, args.seed)

    for b in args.budgets:
        policy = GumbelSwitcher14(
            backbone=env.backbone,
            coarse=coarse,
            sim=sim,
            prior=prior,
            budget=b,
            m=args.m,
            c_visit=args.c_visit,
            c_scale=args.c_scale,
            gumbel_scale=args.gumbel_scale,
            lazy_root=args.lazy_root,
            seed=args.policy_seed if args.policy_seed is not None else args.seed,
            d_safe=args.d_safe,
            selection_interval=env.selection_interval,
            goal_threshold=args.goal_threshold,
            cost=env.cost,
            leaf_value=leaf_value,
            feature_builder=feature_builder,
            feature_builder_v2=(
                feature_builder_v2 if args.log_pi_targets else None
            ),
            coupling=env.coupling,
            precise_groups=env.precise_groups,
            stall_delta=args.stall_delta,
            stall_steps=args.stall_steps,
            stall_signal=args.stall_signal,
            stall_eps=args.stall_eps,
        )
        tag = ("+p" if args.prior_model
               else "+fp" if args.feas_policy_dir else "")
        if args.lazy_root:
            tag += "+lr"
        name = f"GAZ14-b{b}{tag}"
        pi_log = [] if args.log_pi_targets else None
        ep_box = {"seed": -1}
        ep_log: list[dict] = []
        labeler = _ValueLabelCollector(pi_log) if pi_log is not None else None

        def on_episode(ep, seed, _labeler=labeler):
            ep_box.update(seed=seed)
            if _labeler is not None:
                _labeler.on_episode(ep, seed)

        print(f"\nRunning {name} for {args.episodes} episodes ...")
        results[name] = run(env, _gumbel_decider(policy, pi_log, ep_box),
                            args.episodes, args.seed, policy=policy,
                            episode_log=ep_log, on_episode=on_episode,
                            on_step=labeler.on_step if labeler else None)
        if pi_log:
            labeler.finalize()      # an unsolvable last episode is still open
            move_dist = np.array(
                [coarse.move_distances[k] for k in range(len(MOVE_GROUPS))],
                dtype=np.float32,
            )
            alpha = cost.precise_unit / (coarse.lin_max * coarse.step_time)
            _save_pi_targets(pi_log, Path(args.log_pi_targets), name,
                             args.d_safe, move_distances=move_dist,
                             goal_threshold=args.goal_threshold, alpha=alpha,
                             pricing=(cost.precise_unit,
                                      env.selection_interval))
        # Save (and print) each budget's row the moment it finishes — the sweep
        # takes hours per budget, and a killed run must not lose finished rows.
        print_table({name: results[name]}, rows=RESULT_ROWS + GAZ_EFFORT_ROWS)
        if args.out:
            _save_shard(Path(args.out), f"gaz-b{b}", name, b, args,
                        results[name], ep_log)

    print_table(results, rows=RESULT_ROWS + GAZ_EFFORT_ROWS)
    if any(r["coarse_breach"] > 0 for r in results.values()):
        print("WARNING: coarse breaches > 0 — raise --d-safe or check geometry.\n")


if __name__ == "__main__":
    main()
