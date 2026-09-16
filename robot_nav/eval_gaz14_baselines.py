"""
Classical plan-to-goal baselines for the 14-robot switcher: A*, LevinTS,
PHS_h, PHS* — the comparison set for GAZ14-L.

Each baseline is the canonical best-first algorithm (Orseau & Lelis, AAAI
2021 — see ``capswitcher_14/rl/search/best_first.py`` for the exact φ's and
the domain mapping): it plans from the episode's start state to an in-model
``all_reached`` node over the same :class:`ForwardModel14`, same
:class:`SwitcherCost` step costs, same 22-groups+precise action set, then
replays the plan through the env (re-planning only when the plan runs out
before the episode ends — model–sim drift).  Guidance comes from the same
learned functions GAZ uses:

* h  = ``--value-model`` (:class:`LearnedCostToGo`, per-robot value summed
  over unreached robots, priced by the live cost table); analytic
  α·Σdist fallback when absent.  Used by A*, PHS_h, PHS*.
* π  = ``--prior-model`` (:class:`LearnedPrior` logits over the 23 stubs);
  uniform fallback.  Used by LevinTS, PHS_h, PHS*.

Search effort is reported in **materialised edges** — the model's own
``n_coarse_vets`` (refuted vets included) and ``n_precise_expansions``
counters, per episode — rather than a per-decision budget: plan-to-goal
concentrates its transitions in one call, so GAZ's per-decision budget knob
does not apply.  ``--max-transitions`` caps a single planning call; a cap-hit
executes the best generated goal plan (still exact in-model) or the best
partial path by g + h, and is counted in the table.

Episodes, seeds, metrics and the result table are shared with
``eval_gaz14_lazy`` (same ``build_env`` / ``run`` / ``print_table``), so rows
are comparable across the two scripts' tables at matched ``--seed``.

Parallel workers
----------------
Same trick as the data collectors: episodes are independently seeded
(episode k = ``--seed + k``), so disjoint seed blocks in separate processes
run exactly the episodes a single big run would, and ``--out`` /
``--merge`` recombine them **exactly** (means episode-weighted, counts
summed).  Workers must share ``--max-transitions`` / prior / value / world
flags — the merge warns when shards disagree or seed ranges overlap.

Usage (run on the GPU box — local irsim step crashes; see project memory):
    python -m robot_nav.eval_gaz14_baselines --episodes 100 \
        --algos astar levints phs phs-star \
        --prior-model runs/gaz14_value/cycle_05_prior/prior_best.pt \
        --value-model robot_nav/models/MARL/capswitcher/checkpoint/value_local/value_geometry.pt

    # 4 workers over the same 100 episodes (disjoint seed blocks), then merge.
    # Only --seed differs between workers; every planner flag (here
    # --max-transitions, --prior-model, --value-model — plus any world flags)
    # must be identical, or the merged rows mix configurations.
    mkdir -p runs/baselines14
    for S in 1000 1025 1050 1075; do
        python -m robot_nav.eval_gaz14_baselines --algos astar --episodes 25 \
            --seed $S --max-transitions 5000 \
            --prior-model runs/gaz14_value/cycle_05_prior/prior_best.pt \
            --value-model robot_nav/models/MARL/capswitcher/checkpoint/value_local/value_geometry.pt \
            --out runs/baselines14 > runs/baselines14/astart_$S.log 2>&1 &
    done; wait
    
    for S in 1000 1025 1050 1075; do
        python -m robot_nav.eval_gaz14_baselines --algos phs --episodes 25 \
            --seed $S --max-transitions 5000 \
            --prior-model runs/gaz14_value/cycle_05_prior/prior_best.pt \
            --value-model robot_nav/models/MARL/capswitcher/checkpoint/value_local/value_geometry.pt \
            --out runs/baselines14 > runs/baselines14/phs_$S.log 2>&1 &
    done; wait
    
    for S in 1000 1025 1050 1075; do
        python -m robot_nav.eval_gaz14_baselines --algos phs-star --episodes 25 \
            --seed $S --max-transitions 5000 \
            --prior-model runs/gaz14_value/cycle_05_prior/prior_best.pt \
            --value-model robot_nav/models/MARL/capswitcher/checkpoint/value_local/value_geometry.pt \
            --out runs/baselines14 > runs/baselines14/phs-star_$S.log 2>&1 &
    done; wait
    python -m robot_nav.eval_gaz14_baselines --merge runs/baselines14
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger

from robot_nav.eval_gaz14_lazy import (
    DEFAULT_BACKBONE_CKPT,
    DEFAULT_COST_CONFIG,
    RESULT_ROWS,
    add_layout_args,
    add_noise_args,
    build_env,
    layout_from_args,
    noise_from_args,
    noise_meta,
    print_table,
    resolve_device,
    run,
)
from robot_nav.models.MARL.capswitcher.rl.cost import SwitcherCost
from robot_nav.models.MARL.capswitcher_14.configs import MOVE_GROUPS
from robot_nav.models.MARL.capswitcher_14.rl.search.best_first import (
    EVALUATIONS,
    PlanToGoalSwitcher14,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.features import (
    GroupFeatureBuilder,
)

logger.disable("irsim")

# Pretty table names for the CLI algo keys.
ALGO_LABELS = {
    "astar": "A*",
    "levints": "LevinTS",
    "levints-depth": "LevinTS-d",
    "phs": "PHS",
    "phs-star": "PHS*",
}

# Search-effort rows appended below the shared outcome metrics.  Effort is in
# materialised edges (model transition counters), averaged per episode.
BASELINE_ROWS = RESULT_ROWS + [
    ("plans/ep",             "avg_plans",            "{:.2f}"),
    ("fallback decisions",   "fallbacks",            "{:d}"),
    ("coarse vets/ep",       "avg_coarse_vets",      "{:.0f}"),
    ("precise rollouts/ep",  "avg_precise_rollouts", "{:.0f}"),
    ("transitions/ep",       "avg_transitions_ep",   "{:.0f}"),
    ("expansions/ep",        "avg_expansions",       "{:.0f}"),
]


def _episode_tracker(policy: PlanToGoalSwitcher14):
    """
    ``on_step`` observer diffing the policy's cumulative counters at each
    episode end — per-episode search effort from the same loop that produces
    the outcome metrics.  Prints one progress line per finished episode:
    plan-to-goal runs are long, and a silent 12-hour loop is undebuggable.
    """
    per_ep: list[dict] = []
    prev = policy.snapshot()

    def on_step(ep, decision, step_cost, info, done) -> None:
        nonlocal prev
        if done:
            policy.reset_plan()      # never replay a stale suffix next episode
            cur = policy.snapshot()
            d = {k: cur[k] - prev[k] for k in cur}
            per_ep.append(d)
            prev = cur
            outcome = (
                "SUCCESS" if info.get("all_reached")
                else "COLLISION" if info.get("collision")
                else "TIMEOUT" if info.get("timeout") else "ENDED"
            )
            print(
                f"  ep {ep:3d}: {outcome:<9} plans={d['plans']:<3} "
                f"solved={d['solved_plans']:<3} cap_hits={d['cap_hits']:<3} "
                f"transitions={d['coarse_vets'] + d['precise_expansions']}",
                flush=True,
            )

    return on_step, per_ep


def _effort_stats(per_ep: list[dict]) -> dict:
    plans = sum(e["plans"] for e in per_ep)
    return {
        "avg_plans": float(np.mean([e["plans"] for e in per_ep])),
        "plan_solved_rate": (
            sum(e["solved_plans"] for e in per_ep) / plans if plans else 0.0
        ),
        "cap_hits": sum(e["cap_hits"] for e in per_ep),
        "fallbacks": sum(e["fallbacks"] for e in per_ep),
        "avg_coarse_vets": float(np.mean([e["coarse_vets"] for e in per_ep])),
        "avg_precise_rollouts": float(
            np.mean([e["precise_expansions"] for e in per_ep])
        ),
        "avg_transitions_ep": float(
            np.mean([e["coarse_vets"] + e["precise_expansions"] for e in per_ep])
        ),
        "avg_expansions": float(np.mean([e["expansions"] for e in per_ep])),
    }


# ---------------------------------------------------------------------------
# Worker shards: save one JSON per (algo, seed block); merge exactly.
#
# Every mean/rate in the stats dict is recoverable as a sum given its
# denominator (episodes, total decisions, or total plans), so shards from
# disjoint seed blocks recombine into exactly the numbers a single big run
# would have printed.
# ---------------------------------------------------------------------------

_INT_KEYS = ("coarse_breach", "precise_breach", "precise_breach_active",
             "precise_breach_bystander", "cap_hits", "fallbacks")


def _to_counts(stats: dict) -> dict:
    """Undo the per-shard averaging: means/rates -> sums with denominators."""
    n = stats["episodes"]
    dec = stats["avg_decisions"] * n
    successes = round(stats["success_rate"] * n)
    plans = stats["avg_plans"] * n
    return {
        "episodes": n,
        "decisions": dec,
        "successes": successes,
        "collisions": round(stats["collision_rate"] * n),
        "timeouts": round(stats["timeout_rate"] * n),
        "cost": stats["avg_cost"] * n,
        "coarse_cost": stats["avg_coarse_cost"] * n,
        "precise_cost": stats["avg_precise_cost"] * n,
        "success_cost": (
            (stats["avg_cost_success"] or 0.0) * successes
        ),
        "coarse_dec": stats["coarse_frac"] * dec,
        "precise_dec": stats["precise_frac"] * dec,
        "safe_avail": stats["safe_avail_frac"] * dec,
        "transitions_dec": stats["avg_transitions"] * dec,
        "plans": plans,
        "solved_plans": stats["plan_solved_rate"] * plans,
        "coarse_vets": stats["avg_coarse_vets"] * n,
        "precise_rollouts": stats["avg_precise_rollouts"] * n,
        "transitions_ep": stats["avg_transitions_ep"] * n,
        "expansions": stats["avg_expansions"] * n,
        **{k: stats[k] for k in _INT_KEYS},
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
        "avg_plans": c["plans"] / n,
        "plan_solved_rate": c["solved_plans"] / max(c["plans"], 1e-9),
        "avg_coarse_vets": c["coarse_vets"] / n,
        "avg_precise_rollouts": c["precise_rollouts"] / n,
        "avg_transitions_ep": c["transitions_ep"] / n,
        "avg_expansions": c["expansions"] / n,
        **{k: int(round(c[k])) for k in _INT_KEYS},
    }


def _save_shard(out_dir: Path, algo: str, args: argparse.Namespace,
                stats: dict, per_episode: list[dict] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{algo}_s{args.seed}_e{args.episodes}.json"
    path.write_text(json.dumps(
        {
            "algo": algo,
            "label": ALGO_LABELS[algo],
            "seed": args.seed,
            "episodes": args.episodes,
            "max_transitions": args.max_transitions,
            "prior_model": args.prior_model,
            "value_model": args.value_model,
            "analytic_alpha_scale": getattr(args, "analytic_alpha_scale", 1.0),
            "heading_noise": noise_meta(args),
            "stats": stats,
            "per_episode": per_episode or [],
        },
        indent=2,
    ))
    print(f"Saved shard: {path}")


def merge_shards(shard_dir: Path) -> dict[str, dict]:
    """Combine all ``*.json`` worker shards in ``shard_dir`` into one table."""
    shards = [json.loads(p.read_text()) for p in sorted(shard_dir.glob("*.json"))]
    if not shards:
        raise SystemExit(f"no .json shards in {shard_dir}")

    results: dict[str, dict] = {}
    for algo in dict.fromkeys(s["algo"] for s in shards):   # keep CLI order
        group = sorted((s for s in shards if s["algo"] == algo),
                       key=lambda s: s["seed"])
        # Sanity: disjoint seed blocks, identical planner configuration.
        for a, b in zip(group, group[1:]):
            if a["seed"] + a["episodes"] > b["seed"]:
                print(f"WARNING: {algo} shards s{a['seed']} and s{b['seed']} "
                      "overlap — episodes double-counted.")
        for key in ("max_transitions", "prior_model", "value_model",
                    "analytic_alpha_scale", "heading_noise"):
            if len({str(s.get(key)) for s in group}) > 1:
                print(f"WARNING: {algo} shards disagree on {key} — "
                      "the merged row mixes configurations.")
        counts = [_to_counts(s["stats"]) for s in group]
        merged = {k: sum(c[k] for c in counts) for k in counts[0]}
        label = group[0]["label"]
        results[label] = _from_counts(merged)
        blocks = ", ".join(f"s{s['seed']}+{s['episodes']}" for s in group)
        print(f"{label}: {len(group)} shard(s) [{blocks}] -> "
              f"{merged['episodes']} episodes")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--algos", type=str, nargs="+",
                    default=["astar", "levints", "phs", "phs-star"],
                    choices=sorted(EVALUATIONS),
                    help="best-first baselines to run (see best_first.py)")
    ap.add_argument("--max-transitions", type=int, default=20000,
                    help="cap on model transitions (coarse vets + precise "
                         "rollouts) per planning call; a cap-hit falls back "
                         "to the best plan found so far")
    ap.add_argument("--d-safe", type=float, default=0.3)
    ap.add_argument("--cost-config", type=str, default=DEFAULT_COST_CONFIG)
    ap.add_argument("--goal-threshold", type=float, default=0.3)
    ap.add_argument("--backbone-ckpt", type=str, default=DEFAULT_BACKBONE_CKPT)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--replan-drift", type=float, default=None,
                    help="drift-triggered replanning: drop the plan suffix "
                         "when live poses deviate from the plan's predicted "
                         "poses by more than this (m).  The heading-noise "
                         "robustness mechanism; default off = open-loop "
                         "replay")
    ap.add_argument("--receding", action="store_true",
                    help="receding horizon: replan every decision and execute "
                         "only the first action (default replays the plan "
                         "until exhaustion/deviation)")
    ap.add_argument("--max-decisions", type=int, default=120,
                    help="episode decision budget before timeout — match the "
                         "run being compared against (xeval_coupled used 150)")
    ap.add_argument("--feas-policy-dir", type=str, default=None,
                    help="split feas/policy prior directory "
                         "(train_feas_policy14.py) — the π for PHS; mutually "
                         "exclusive with --prior-model")
    ap.add_argument("--feas-threshold", type=float, default=0.1)
    ap.add_argument("--value-residual", type=str, default=None,
                    help="ValueResidualNet leaf (train_value_residual14.py); "
                         "mutually exclusive with --value-model")
    ap.add_argument("--prior-model", type=str, default=None,
                    help="learned PriorNet checkpoint -> π for LevinTS/PHS/"
                         "PHS*; default uniform")
    ap.add_argument("--value-model", type=str, default=None,
                    help="learned cost-to-go checkpoint -> h for A*/PHS/PHS*; "
                         "default analytic α·Σdist")
    ap.add_argument("--analytic-alpha-scale", type=float, default=1.0,
                    help="scale on the analytic leaf's α (default α ≈ 77.1 "
                         "per robot-metre prices everything at precise rates "
                         "— strongly inadmissible; smaller scales broaden "
                         "the search toward cheaper plans).  Label-quality "
                         "probe of the value-corpus pilot; incompatible "
                         "with --value-model")
    ap.add_argument("--precise-config", choices=["all", "pairs", "singles"],
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
    ap.add_argument("--log-traces", type=str, default=None,
                    help="directory for search-trace shards (redesign §6): "
                         "every expansion + plan outcome of every planning "
                         "call, one npz per plan, raw snapshots — the "
                         "training-data source for the token value/prior "
                         "nets.  Shards land in "
                         "<dir>/<algo>_<config>_s<seed>/")
    ap.add_argument("--out", type=str, default=None,
                    help="directory for per-(algo, seed-block) JSON shards — "
                         "run disjoint seed blocks in parallel workers, then "
                         "combine with --merge")
    ap.add_argument("--merge", type=str, default=None,
                    help="merge the worker shards in this directory into one "
                         "table and exit (no episodes are run)")
    add_layout_args(ap)
    add_noise_args(ap)
    args = ap.parse_args()

    if args.merge:
        results = merge_shards(Path(args.merge))
        print_table(results, rows=BASELINE_ROWS)
        return

    device = resolve_device(args.device)
    cost = SwitcherCost.from_yaml(args.cost_config)
    layout = layout_from_args(args)
    if args.prior_model and args.precise_config != "all":
        ap.error("the current PriorNet emits fixed 22+1 logits; a learned "
                 "prior with --precise-config pairs/singles needs the "
                 "redesigned token net (design doc §5)")
    heading_noise = noise_from_args(args)
    env, coarse, sim = build_env(
        device, cost=cost, goal_threshold=args.goal_threshold,
        backbone_ckpt=args.backbone_ckpt, layout=layout,
        precise_config=args.precise_config, coupled=args.coupled_precise,
        heading_noise=heading_noise, max_decisions=args.max_decisions,
    )
    print(
        f"World: {'corridor ' + str(layout.band) if layout else 'scattered'}\n"
        f"Env: {sim.num_robots} robots, {sim.num_obstacles} obstacles, "
        f"{len(MOVE_GROUPS)} coarse groups, cost_config={args.cost_config}, "
        f"d_safe={args.d_safe}, max_transitions={args.max_transitions}, "
        f"prior={args.prior_model or 'uniform'}, "
        f"value={args.value_model or 'analytic'}, device={device}"
    )
    if heading_noise is not None:
        print(f"Coarse heading noise: {noise_meta(args)}")

    if args.value_model and args.analytic_alpha_scale != 1.0:
        ap.error("--analytic-alpha-scale scales the analytic leaf; it has no "
                 "meaning together with --value-model")

    if args.value_model and args.value_residual:
        ap.error("--value-model and --value-residual are mutually exclusive")
    if args.prior_model and args.feas_policy_dir:
        ap.error("--prior-model and --feas-policy-dir are mutually exclusive")

    leaf_value = None
    if args.value_model:
        from robot_nav.models.MARL.capswitcher.rl.value_net import LearnedCostToGo

        leaf_value = LearnedCostToGo(args.value_model, device=device)
        print(f"Learned leaf: feature={leaf_value.feature}")
    elif args.value_residual:
        from robot_nav.models.MARL.capswitcher_14.rl.search.features_v2 import (
            GroupFeatureBuilderV2,
        )
        from robot_nav.models.MARL.capswitcher_14.rl.search.value_residual import (
            LearnedValueResidual,
        )

        leaf_value = LearnedValueResidual(
            args.value_residual,
            GroupFeatureBuilderV2(MOVE_GROUPS,
                                  move_distances=coarse.move_distances,
                                  goal_threshold=args.goal_threshold),
            device="cpu",
        )
        print(f"Value residual leaf: {args.value_residual} "
              f"(base={leaf_value.net.base})")
    elif args.analytic_alpha_scale != 1.0:
        from robot_nav.models.MARL.capswitcher_14.rl.forward_model import (
            analytic_cost_to_go,
        )

        scale = float(args.analytic_alpha_scale)
        leaf_value = lambda m, ms: analytic_cost_to_go(  # noqa: E731
            m.goal_distances(ms), m.goal_threshold, scale * m.analytic_alpha()
        )
        print(f"Analytic leaf: alpha scaled x{scale}")

    prior = None
    if args.prior_model:
        from robot_nav.models.MARL.capswitcher_14.rl.search.prior_net import (
            LearnedPrior,
            PriorNet,
        )

        prior = LearnedPrior(
            PriorNet.load(args.prior_model, map_location=device),
            GroupFeatureBuilder(MOVE_GROUPS),
            device=device,
        )
        print(f"Learned prior: {args.prior_model}")
    elif args.feas_policy_dir:
        from robot_nav.models.MARL.capswitcher_14.rl.search.feas_policy_net import (
            FeasNet14,
            GroupPolicyNet14,
            LearnedFeasPolicyPrior,
        )
        from robot_nav.models.MARL.capswitcher_14.rl.search.features_v2 import (
            GroupFeatureBuilderV2,
        )

        fp_dir = Path(args.feas_policy_dir)
        prior = LearnedFeasPolicyPrior(
            FeasNet14.load(fp_dir / "feas_best.pt", map_location=device),
            GroupPolicyNet14.load(fp_dir / "policy_best.pt",
                                  map_location=device),
            GroupFeatureBuilderV2(MOVE_GROUPS,
                                  move_distances=coarse.move_distances,
                                  goal_threshold=args.goal_threshold),
            feas_threshold=args.feas_threshold,
            device=device,
        )
        print(f"Feas/policy prior: {fp_dir}")

    results: dict[str, dict] = {}
    for algo in args.algos:
        recorder = None
        if args.log_traces:
            from robot_nav.models.MARL.capswitcher_14.configs import (
                A_FULL,
                build_precise_groups,
            )
            from robot_nav.models.MARL.capswitcher_14.rl.search.trace import (
                TraceRecorder,
            )

            recorder = TraceRecorder(
                Path(args.log_traces)
                / f"{algo}_{args.precise_config}_s{args.seed}",
                meta={
                    "algo": algo,
                    "precise_config": args.precise_config,
                    "coupled": args.coupled_precise or False,
                    "A_full": A_FULL,
                    "move_groups": [
                        [int(i) for i in coarse.members_of(g)]
                        for g in coarse.selectable_groups()
                    ],
                    "precise_groups": build_precise_groups(args.precise_config),
                    "precise_unit": env.cost.precise_unit,
                    "move_distances": env.cost.move_distances,
                    "selection_interval": env.selection_interval,
                    "d_safe": args.d_safe,
                    "goal_threshold": args.goal_threshold,
                    "max_transitions": args.max_transitions,
                    "value_model": args.value_model,
                    "analytic_alpha_scale": args.analytic_alpha_scale,
                    "prior_model": args.prior_model,
                    "base_seed": args.seed,
                    # Provenance for regenerating/auditing child states: the
                    # precise children are functions of the GAT weights.
                    "backbone_ckpt": args.backbone_ckpt,
                    "device": str(device),
                    "torch_version": __import__("torch").__version__,
                    "trace_schema": 3,
                    # Precise rollouts truncate at the first colliding
                    # sub-step; shards collected without this flag explored
                    # branches this search prunes — do not mix corpora.
                    "substep_collision": True,
                    # Offline model reconstruction (collect_sibling_values):
                    "step_time": getattr(coarse, "step_time", 0.3),
                    "lin_max": getattr(coarse, "lin_max", 0.5),
                    "ang_max": getattr(coarse, "ang_max", 1.0),
                    "cost_config": args.cost_config,
                    "group_costs": {
                        int(g): env.cost.coarse_cost(g)
                        for g in coarse.selectable_groups()
                    },
                    "episodes": args.episodes,
                    "world": "corridor" if layout is not None else "scattered",
                },
            )
        policy = PlanToGoalSwitcher14(
            backbone=env.backbone,
            coarse=coarse,
            sim=sim,
            evaluate=EVALUATIONS[algo],
            prior=prior,
            max_transitions=args.max_transitions,
            d_safe=args.d_safe,
            selection_interval=env.selection_interval,
            goal_threshold=args.goal_threshold,
            cost=env.cost,
            leaf_value=leaf_value,
            coupling=env.coupling,
            precise_groups=env.precise_groups,
            recorder=recorder,
            replan_drift=args.replan_drift,
        )
        name = ALGO_LABELS[algo]
        print(f"\nRunning {name} for {args.episodes} episodes ...")
        on_step, per_ep = _episode_tracker(policy)
        if args.receding:
            # Receding horizon: execute only the first action of each plan —
            # clearing the plan forces a fresh (cap-limited) solve every
            # decision, like the GAZ switchers' per-decision planning.
            def decide(env_, p=policy):
                d = p.decide(env_._robot_state)
                p.reset_plan()
                return d
        else:
            decide = lambda env_, p=policy: p.decide(env_._robot_state)  # noqa: E731
        on_episode = (
            (lambda ep, seed, r=recorder: r.set_episode(ep, seed))
            if recorder is not None else None
        )
        ep_log: list[dict] = []
        stats = run(env, decide, args.episodes, args.seed,
                    policy=policy, on_step=on_step, episode_log=ep_log,
                    on_episode=on_episode)
        stats.update(_effort_stats(per_ep))
        results[name] = stats
        # Each episode's outcome/cost record plus its search-effort diff — the
        # per-seed row the cross-policy comparison joins on.
        per_episode = [
            {**rec, "plans": d["plans"],
             "coarse_vets": d["coarse_vets"],
             "precise_rollouts": d["precise_expansions"]}
            for rec, d in zip(ep_log, per_ep)
        ]
        # Print each algorithm's rows the moment it finishes — these runs take
        # hours per algorithm, and a killed run must not lose finished results.
        print_table({name: stats}, rows=BASELINE_ROWS)
        if args.out:
            _save_shard(Path(args.out), algo, args, stats, per_episode)

    if len(results) > 1:
        print_table(results, rows=BASELINE_ROWS)


if __name__ == "__main__":
    main()
