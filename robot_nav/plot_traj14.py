"""
Paper-ready 14-robot trajectory figures for GAZ14-solved episodes.

Re-runs seeded episodes (episode seed = the same handle the eval tables use,
via ``seed_episode``) with the deployed GAZ14 switcher, records every robot's
position at **sim sub-step** resolution through the ``_apply_substep`` hook,
and renders one figure per episode: obstacle discs, per-robot path, start
disc, goal star, final disc.

Recording and rendering are decoupled — each episode is cached as an ``.npz``
next to the figures, and ``--replot`` restyles from the cache without touching
the sim.  Defaults reproduce the cross-eval GAZ arm the paper tables come from
(``runs/xeval_coupled/uncoupled_c9``: b200, m16, policy-seed 77, uncoupled
cycle-9 nets, pinv-coupled world, 150-decision budget, CPU).  Run with
``OMP_NUM_THREADS=2`` to match the eval workers bit-for-bit.

Usage (repo root, DRL_nav env):
    # one episode, to check the design
    OMP_NUM_THREADS=2 PYTHONPATH=. python -m robot_nav.plot_traj14 \
        --seeds 50010 --out runs/paper_figs/traj

    # the candidate set for the paper
    OMP_NUM_THREADS=2 PYTHONPATH=. python -m robot_nav.plot_traj14 \
        --seeds 50000 50001 50005 50007 50010 50013 50023 50031 50050 50053 \
        --out runs/paper_figs/traj

    # restyle without re-running episodes
    PYTHONPATH=. python -m robot_nav.plot_traj14 --seeds 50010 \
        --out runs/paper_figs/traj --replot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from robot_nav.models.MARL.capswitcher.rl.cost import SwitcherCost
from robot_nav.models.MARL.capswitcher.rl.shield import ShieldGeometry
from robot_nav.models.MARL.capswitcher.rl.switcher_env import seed_episode
from robot_nav.models.MARL.capswitcher_14.configs import MOVE_GROUPS
from robot_nav.models.MARL.capswitcher_14.rl.search.features import (
    GroupFeatureBuilder,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.features_v2 import (
    GroupFeatureBuilderV2,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.gumbel import GumbelSwitcher14

logger.disable("irsim")

# ---------------------------------------------------------------------------
# Deployed GAZ14 configuration — the cross-eval arm (91% @ seeds 50000-50099).
# ---------------------------------------------------------------------------
DEFAULT_FEAS_POLICY_DIR = "runs/az6_nostall/cycle8/nets"
DEFAULT_VALUE_RESIDUAL = "runs/az6_nostall/cycle8/value/value_best.pt"

# 14 fixed, print-strong hues (identity palette; assignment robot i -> color i
# never changes between figures).  Start/goal/final markers share the robot's
# hue, so identity is carried by the connected start-path-goal chain, not by
# color alone.
ROBOT_COLORS = [
    "#e6194B", "#f58231", "#b8860b", "#808000", "#3cb44b", "#469990",
    "#17becf", "#4363d8", "#000075", "#911eb4", "#f032e6", "#9A6324",
    "#800000", "#e377c2",
]

OBSTACLE_FACE = "#d4d4d8"
OBSTACLE_EDGE = "#71717a"


# ---------------------------------------------------------------------------
# Episode capture
# ---------------------------------------------------------------------------

def _xy(poses) -> np.ndarray:
    """(N, 2) positions from the env's per-robot pose list."""
    return np.array(
        [np.asarray(p, dtype=np.float64).ravel()[:2] for p in poses],
        dtype=np.float32,
    )


def record_episode(env, decide_fn, policy, seed: int, algo: str,
                   coarse) -> dict:
    """Run one seeded episode; return arrays for the figure + npz cache."""
    seed_episode(env, seed)
    if hasattr(policy, "reset"):
        policy.reset()
    env.reset()

    positions = [_xy(env._poses)]          # sub-step resolution path
    modes = [-1]                           # mode executing at each sub-step
    sizes = [0]                            # primitive size (|group|; 1=precise)
    current = {"mode": -1, "size": 0}

    orig_substep = env._apply_substep

    def hooked_substep(actions, info):
        done = orig_substep(actions, info)
        positions.append(_xy(env._poses))
        modes.append(current["mode"])
        sizes.append(current["size"])
        return done

    env._apply_substep = hooked_substep
    ep_cost, decisions = 0.0, 0
    dec_modes: list[int] = []      # per executed decision, for statistics
    dec_groups: list[int] = []
    dec_sizes: list[int] = []
    outcome = "ENDED"
    try:
        done = False
        info: dict = {}
        while not done:
            decision = decide_fn()
            if decision.get("unsolvable"):
                outcome = "UNSOLVABLE"
                break
            current["mode"] = int(decision["mode"])
            current["size"] = (
                1 if decision["mode"] == 1
                else len(coarse.members_of(decision["group"]))
                if decision.get("group") is not None else 0
            )
            dec_modes.append(current["mode"])
            dec_groups.append(
                -1 if decision.get("group") is None
                else int(decision["group"]))
            dec_sizes.append(current["size"])
            _, _, done, info = env.step(
                decision["mode"], group=decision["group"],
                frames=decision["frames"], pgroup=decision.get("pgroup"),
            )
            decisions += 1
            ep_cost += float(info["path_cost"])
        else:
            outcome = ("SUCCESS" if info.get("all_reached")
                       else "COLLISION" if info.get("collision")
                       else "TIMEOUT" if info.get("timeout") else "ENDED")
    finally:
        env._apply_substep = orig_substep

    geom = ShieldGeometry.from_sim(env.sim)
    return {
        "algo": np.str_(algo),
        "seed": np.int64(seed),
        "positions": np.stack(positions),                  # (T, N, 2)
        "modes": np.array(modes, dtype=np.int8),           # (T,)
        "sizes": np.array(sizes, dtype=np.int8),           # (T,)
        "dec_modes": np.array(dec_modes, dtype=np.int8),   # (D,)
        "dec_groups": np.array(dec_groups, dtype=np.int16),
        "dec_sizes": np.array(dec_sizes, dtype=np.int8),
        "goals": np.array(
            [np.asarray(g, dtype=np.float64).ravel()[:2]
             for g in env._goal_positions], dtype=np.float32),
        "obstacle_xy": np.asarray(geom.obstacle_xy, dtype=np.float32),
        "obstacle_r": np.asarray(geom.obstacle_r, dtype=np.float32),
        "rho": np.float32(geom.rho),
        "x_range": np.asarray(env.sim.x_range, dtype=np.float32),
        "y_range": np.asarray(env.sim.y_range, dtype=np.float32),
        "outcome": np.str_(outcome),
        "decisions": np.int64(decisions),
        "cost": np.float32(ep_cost),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

# Dash pattern for precise-mode path segments (scaled for lw ~1.3).
PRECISE_DASH = (0, (2.2, 1.4))

# --color-by size: path color encodes the executing primitive's size —
# the driven coarse group's cardinality (7 / 4 / 3), or 1 for a precise
# per-robot move (drawn dashed, neutral).  Robot identity is dropped:
# every marker goes neutral dark so the group color is the only signal.
SIZE_COLORS = {7: "#4363d8", 4: "#f58231", 3: "#3cb44b", 1: "#4a4a4a"}
NEUTRAL = "#1f2937"


def _mode_runs(modes: np.ndarray):
    """
    Contiguous runs of equal mode over the path's segments.

    Segment t (connecting sub-step t-1 to t) carries ``modes[t]``.  Yields
    ``(mode, a, b)`` with the run covering segments a..b inclusive, so the
    polyline to draw is ``pos[a-1 : b+1]``.
    """
    seg = modes[1:]
    if seg.size == 0:
        return
    starts = np.flatnonzero(np.diff(seg) != 0) + 1
    bounds = np.concatenate(([0], starts, [seg.size]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        yield int(seg[a]), int(a) + 1, int(b)


def render_figure(rec: dict, out_stem: Path, title: bool = True,
                  labels: bool = True,
                  formats: tuple = ("pdf", "png"),
                  color_by: str = "robot") -> None:
    """
    One paper-ready trajectory figure from a recorded episode.

    Path style encodes the executing mode: solid = coarse group control,
    dashed = precise per-robot resolution.  (During precise decisions only
    the driven robot moves, so other robots' dashed runs are zero-length
    and invisible.)

    ``color_by``: ``"robot"`` (default) gives each robot its identity hue;
    ``"size"`` colors each path segment by the executing primitive's size
    (coarse |G| = 7/4/3, precise = 1, see ``SIZE_COLORS``) with all robots'
    markers neutral — requires an npz recorded with the ``sizes`` array.
    """
    if color_by == "size" and "sizes" not in rec:
        raise SystemExit(
            f"{out_stem.name}: npz has no 'sizes' array — re-record this "
            "seed (delete the cached npz) to use --color-by size")
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType (camera-ready)
        "font.size": 8,
        "axes.linewidth": 0.8,
    })
    pos = np.asarray(rec["positions"])            # (T, N, 2)
    goals = np.asarray(rec["goals"])
    rho = float(rec["rho"])
    n_robots = pos.shape[1]

    fig, ax = plt.subplots(figsize=(4.4, 4.4), constrained_layout=True)
    ax.set_aspect("equal")
    xr, yr = np.asarray(rec["x_range"]), np.asarray(rec["y_range"])
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    ax.set_xticks(np.arange(xr[0], xr[1] + 1e-6, 5))
    ax.set_yticks(np.arange(yr[0], yr[1] + 1e-6, 5))
    ax.tick_params(length=2.5, labelsize=7, colors="#374151")
    for s in ax.spines.values():
        s.set_color("#374151")
    ax.set_xlabel("x (m)", fontsize=8, color="#111827")
    ax.set_ylabel("y (m)", fontsize=8, color="#111827")

    # Obstacles.
    for (ox, oy), orad in zip(rec["obstacle_xy"], rec["obstacle_r"]):
        ax.add_patch(Circle((ox, oy), float(orad), facecolor=OBSTACLE_FACE,
                            edgecolor=OBSTACLE_EDGE, linewidth=0.8, zorder=1))

    modes = np.asarray(rec["modes"], dtype=np.int64)
    if color_by == "size":
        # Runs of constant (mode, size); decode below.
        codes = modes * 100 + np.asarray(rec["sizes"], dtype=np.int64)
    else:
        codes = modes

    for i in range(n_robots):
        c = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        # Path — solid while a coarse control executes, dashed while precise.
        for code, a, b in _mode_runs(codes):
            mode = code // 100 if color_by == "size" else code
            if color_by == "size":
                seg_color = SIZE_COLORS.get(code % 100, NEUTRAL)
            else:
                seg_color = c
            ax.plot(pos[a - 1:b + 1, i, 0], pos[a - 1:b + 1, i, 1],
                    color=seg_color, linewidth=1.3, alpha=0.85,
                    linestyle=PRECISE_DASH if mode == 1 else "-",
                    solid_capstyle="round", zorder=3)
        mk = NEUTRAL if color_by == "size" else c
        # Start: robot disc, hollow.
        ax.add_patch(Circle(tuple(pos[0, i]), rho, facecolor="none",
                            edgecolor=mk, linewidth=1.2, zorder=4))
        # Final: robot disc, filled.
        ax.add_patch(Circle(tuple(pos[-1, i]), rho, facecolor=mk,
                            edgecolor="white", linewidth=0.6, alpha=0.9,
                            zorder=5))
        # Goal star.
        ax.plot(goals[i, 0], goals[i, 1], marker="*", markersize=9,
                markerfacecolor=mk, markeredgecolor="white",
                markeredgewidth=0.5, linestyle="none", zorder=6)
        if labels:
            ax.annotate(str(i), xy=tuple(pos[0, i]),
                        xytext=(pos[0, i, 0] + 1.6 * rho,
                                pos[0, i, 1] + 1.6 * rho),
                        fontsize=5.5, color=mk, ha="center", va="center",
                        zorder=7)

    # Symbol key (gray = not an identity claim).
    if color_by == "size":
        mode_key = [
            Line2D([], [], color=SIZE_COLORS[7], linewidth=1.3,
                   linestyle="-", label="coarse |G|=7"),
            Line2D([], [], color=SIZE_COLORS[4], linewidth=1.3,
                   linestyle="-", label="coarse |G|=4"),
            Line2D([], [], color=SIZE_COLORS[3], linewidth=1.3,
                   linestyle="-", label="coarse |G|=3"),
            Line2D([], [], color=SIZE_COLORS[1], linewidth=1.3,
                   linestyle=PRECISE_DASH, label="precise"),
        ]
    else:
        mode_key = [
            Line2D([], [], color="#4b5563", linewidth=1.3, linestyle="-",
                   label="coarse"),
            Line2D([], [], color="#4b5563", linewidth=1.3,
                   linestyle=PRECISE_DASH, label="precise"),
        ]
    key = mode_key + [
        Line2D([], [], marker="o", markerfacecolor="none",
               markeredgecolor="#4b5563", markersize=6, linestyle="none",
               label="start"),
        Line2D([], [], marker="*", markerfacecolor="#4b5563",
               markeredgecolor="white", markeredgewidth=0.4, markersize=9,
               linestyle="none", label="goal"),
        Line2D([], [], marker="o", markerfacecolor="#4b5563",
               markeredgecolor="white", markersize=6, linestyle="none",
               label="final"),
        Line2D([], [], marker="o", markerfacecolor=OBSTACLE_FACE,
               markeredgecolor=OBSTACLE_EDGE, markersize=7, linestyle="none",
               label="obstacle"),
    ]
    ax.legend(handles=key, loc="upper left", bbox_to_anchor=(0.0, 1.0),
              ncol=4 if color_by == "size" else 3, fontsize=6.5,
              frameon=True, framealpha=0.9,
              edgecolor="#d1d5db", borderpad=0.4, handletextpad=0.3,
              columnspacing=0.9, bbox_transform=ax.transAxes)

    if title:
        ax.set_title(
            f"seed {int(rec['seed'])}  ·  {str(rec['outcome'])}  ·  "
            f"{int(rec['decisions'])} decisions  ·  cost {float(rec['cost']):.0f}",
            fontsize=7.5, color="#374151", pad=4,
        )

    for ext in formats:
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  wrote {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_policy(args, env, coarse, sim, device):
    """
    The deployed decider: ``--algo gaz`` builds the Gumbel switcher; the
    best-first algos (``astar``/``phs``/...) build ``PlanToGoalSwitcher14``
    in receding-horizon mode (replan every decision, ``--max-transitions``
    cap per plan — the paper baselines' configuration).  Both share the same
    cycle-8 guidance: feas/policy prior as pi, value residual as h.

    Returns ``(policy, decide_fn)``.
    """
    from robot_nav.models.MARL.capswitcher_14.rl.search.feas_policy_net import (
        FeasNet14,
        GroupPolicyNet14,
        LearnedFeasPolicyPrior,
    )
    from robot_nav.models.MARL.capswitcher_14.rl.search.value_residual import (
        LearnedValueResidual,
    )

    feature_builder = GroupFeatureBuilder(MOVE_GROUPS)
    feature_builder_v2 = GroupFeatureBuilderV2(
        MOVE_GROUPS, move_distances=coarse.move_distances,
        goal_threshold=args.goal_threshold,
    )
    fp_dir = Path(args.feas_policy_dir)
    prior = LearnedFeasPolicyPrior(
        FeasNet14.load(fp_dir / "feas_best.pt", map_location=device),
        GroupPolicyNet14.load(fp_dir / "policy_best.pt", map_location=device),
        feature_builder_v2,
        feas_threshold=args.feas_threshold,
        device=device,
    )
    leaf_value = LearnedValueResidual(
        args.value_residual, feature_builder_v2, device="cpu")

    if args.algo == "gaz":
        policy = GumbelSwitcher14(
            backbone=env.backbone, coarse=coarse, sim=sim, prior=prior,
            budget=args.budget, m=args.m, seed=args.policy_seed,
            d_safe=args.d_safe, selection_interval=env.selection_interval,
            goal_threshold=args.goal_threshold, cost=env.cost,
            leaf_value=leaf_value, feature_builder=feature_builder,
            coupling=env.coupling, precise_groups=env.precise_groups,
            stall_steps=args.stall_steps,
        )
        return policy, lambda: policy.decide(env._robot_state)

    from robot_nav.models.MARL.capswitcher_14.rl.search.best_first import (
        EVALUATIONS,
        PlanToGoalSwitcher14,
    )

    policy = PlanToGoalSwitcher14(
        backbone=env.backbone, coarse=coarse, sim=sim,
        evaluate=EVALUATIONS[args.algo], prior=prior,
        max_transitions=args.max_transitions, d_safe=args.d_safe,
        selection_interval=env.selection_interval,
        goal_threshold=args.goal_threshold, cost=env.cost,
        leaf_value=leaf_value, coupling=env.coupling,
        precise_groups=env.precise_groups,
    )

    def decide():
        d = policy.decide(env._robot_state)
        policy.reset_plan()          # receding horizon: replan every decision
        return d

    return policy, decide


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True,
                    help="episode seeds (the eval tables' handles, e.g. 50000)")
    ap.add_argument("--out", type=str, default="runs/paper_figs/traj")
    ap.add_argument("--replot", action="store_true",
                    help="render from cached .npz only; never run the sim")
    ap.add_argument("--no-title", action="store_true")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--color-by", choices=["robot", "size"], default="robot",
                    help="path color: per-robot identity hue, or the "
                         "executing primitive's size (|G|=7/4/3, precise=1; "
                         "markers go neutral)")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--algo", type=str, default="gaz",
                    choices=["gaz", "astar", "levints", "levints-depth",
                             "phs", "phs-star"],
                    help="decider: 'gaz' = Gumbel switcher (b200); others = "
                         "guided best-first planners in receding-horizon "
                         "mode (paper baseline config)")
    ap.add_argument("--max-transitions", type=int, default=5000,
                    help="per-plan transition cap for the best-first algos")
    # GAZ config — defaults are the paper arm (cycle-8 nets, b200).
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--policy-seed", type=int, default=77)
    ap.add_argument("--stall-steps", type=int, default=0)
    ap.add_argument("--max-decisions", type=int, default=150)
    ap.add_argument("--d-safe", type=float, default=0.3)
    ap.add_argument("--feas-threshold", type=float, default=0.1)
    ap.add_argument("--goal-threshold", type=float, default=0.3)
    ap.add_argument("--feas-policy-dir", type=str,
                    default=DEFAULT_FEAS_POLICY_DIR)
    ap.add_argument("--value-residual", type=str,
                    default=DEFAULT_VALUE_RESIDUAL)
    ap.add_argument("--coupled-precise", type=str, default="pinv",
                    choices=["pinv", "group", "off"])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [s for s in args.seeds
            if not (out_dir / f"traj_s{s}.npz").exists()]
    if todo and args.replot:
        raise SystemExit(f"--replot but no cache for seeds {todo}")

    if todo:
        from robot_nav.eval_gaz14_lazy import DEFAULT_BACKBONE_CKPT, \
            DEFAULT_COST_CONFIG, build_env

        device = torch.device("cpu")   # matches the eval workers (determinism)
        cost = SwitcherCost.from_yaml(DEFAULT_COST_CONFIG)
        coupled = False if args.coupled_precise == "off" else args.coupled_precise
        env, coarse, sim = build_env(
            device, cost=cost, goal_threshold=args.goal_threshold,
            backbone_ckpt=DEFAULT_BACKBONE_CKPT, coupled=coupled,
            max_decisions=args.max_decisions,
        )
        policy, decide_fn = build_policy(args, env, coarse, sim, device)
        for seed in todo:
            print(f"Recording seed {seed} ...", flush=True)
            rec = record_episode(env, decide_fn, policy, seed, args.algo,
                                 coarse)
            np.savez_compressed(out_dir / f"traj_s{seed}.npz", **rec)
            print(f"  {rec['outcome']} in {int(rec['decisions'])} decisions, "
                  f"cost {float(rec['cost']):.0f}", flush=True)

    for seed in args.seeds:
        with np.load(out_dir / f"traj_s{seed}.npz") as z:
            rec = {k: z[k] for k in z.files}
        stem = (f"traj_s{seed}" if args.color_by == "robot"
                else f"traj_s{seed}_bysize")
        render_figure(rec, out_dir / stem,
                      title=not args.no_title, labels=not args.no_labels,
                      formats=tuple(args.formats), color_by=args.color_by)


if __name__ == "__main__":
    main()
