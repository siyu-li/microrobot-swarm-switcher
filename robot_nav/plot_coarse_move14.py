"""
Anatomy of ONE coarse group move — the rank-deficiency figure.

No trajectories: a seeded world's start state, one coarse group primitive
(default: a size-7 block) executed through the env, and the before/after
poses of the driven robots.  Each driven robot shows its heading arrow and
a dashed line to its goal (the desired heading); after the move the arrows
still disagree with the goal lines — the group rotation cannot orient
robots individually (rank-deficient actuation), so the group translates
toward the goals but no robot tracks its own desired angle.

Usage (repo root, DRL_nav env):
    PYTHONPATH=. python -m robot_nav.plot_coarse_move14 --seed 50009 \
        --out runs/paper_figs/coarse_move
    # explicit group id (0-based; 18-21 are the size-7 blocks)
    PYTHONPATH=. python -m robot_nav.plot_coarse_move14 --seed 50009 --group 19
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
from matplotlib.patches import Circle, FancyArrow

from robot_nav.models.MARL.capswitcher.rl.cost import SwitcherCost
from robot_nav.models.MARL.capswitcher.rl.shield import ShieldGeometry
from robot_nav.models.MARL.capswitcher.rl.switcher_env import seed_episode
import colorsys

from matplotlib.colors import to_hex, to_rgb

from robot_nav.plot_traj14 import (
    NEUTRAL,
    OBSTACLE_EDGE,
    OBSTACLE_FACE,
    ROBOT_COLORS,
)

logger.disable("irsim")

DESIRED_BLUE = "#2563eb"         # the desired-action color
ARROW_LEN = 0.85                 # heading arrow length (m)


def _tint(color: str, f: float = 0.55) -> str:
    """
    Lighter version of the same hue (the next-state color): mix with white.
    A pure saturation cut keeps lightness fixed, so dark hues go muddy
    instead of pale — tinting is the right operation for "lighter".
    """
    r, g, b = to_rgb(color)
    return to_hex((r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f))


def _pose3(poses) -> np.ndarray:
    """(N, 3) [x, y, theta] from the env's per-robot pose list."""
    return np.array(
        [np.asarray(p, dtype=np.float64).ravel()[:3] for p in poses],
        dtype=np.float64,
    )


def _heading_arrow(ax, x, y, th, color, zorder=6, alpha=1.0):
    ax.add_patch(FancyArrow(
        x, y, ARROW_LEN * np.cos(th), ARROW_LEN * np.sin(th),
        width=0.045, head_width=0.24, head_length=0.2,
        length_includes_head=True, color=color, linewidth=0,
        zorder=zorder, alpha=alpha,
    ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=50009)
    ap.add_argument("--group", type=int, default=None,
                    help="0-based coarse group id (default: first size-7 "
                         "group, id 18)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="apply the same group primitive this many times "
                         "(a longer move makes the deviation more visible)")
    ap.add_argument("--out", type=str, default="runs/paper_figs/coarse_move")
    ap.add_argument("--no-title", action="store_true")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--coupled-precise", type=str, default="pinv",
                    choices=["pinv", "group", "off"])
    args = ap.parse_args()

    from robot_nav.eval_gaz14_lazy import (
        DEFAULT_BACKBONE_CKPT,
        DEFAULT_COST_CONFIG,
        build_env,
    )

    device = torch.device("cpu")
    cost = SwitcherCost.from_yaml(DEFAULT_COST_CONFIG)
    coupled = False if args.coupled_precise == "off" else args.coupled_precise
    env, coarse, sim = build_env(
        device, cost=cost, goal_threshold=0.3,
        backbone_ckpt=DEFAULT_BACKBONE_CKPT, coupled=coupled,
    )

    group = args.group
    if group is None:
        group = next(g for g in coarse.selectable_groups()
                     if len(coarse.members_of(g)) == 7)
    members = np.asarray(coarse.members_of(group), dtype=int)

    seed_episode(env, args.seed)
    env.reset()
    before = _pose3(env._poses)
    goals = np.array(
        [np.asarray(g, dtype=np.float64).ravel()[:2]
         for g in env._goal_positions])
    geom = ShieldGeometry.from_sim(sim)

    for _ in range(args.repeats):
        env.step(0, group=int(group))
    after = _pose3(env._poses)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"coarse_move_s{args.seed}_g{group}.npz",
        before=before, after=after, goals=goals, members=members,
        group=group, seed=args.seed, rho=np.float32(geom.rho),
        obstacle_xy=geom.obstacle_xy, obstacle_r=geom.obstacle_r,
        x_range=np.asarray(sim.x_range), y_range=np.asarray(sim.y_range),
    )

    # ---- figure ----------------------------------------------------------
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.size": 8, "axes.linewidth": 0.8})
    rho = float(geom.rho)
    fig, ax = plt.subplots(figsize=(4.4, 4.4), constrained_layout=True)
    ax.set_aspect("equal")
    ax.set_xlim(*sim.x_range)
    ax.set_ylim(*sim.y_range)
    ax.set_xticks(np.arange(sim.x_range[0], sim.x_range[1] + 1e-6, 5))
    ax.set_yticks(np.arange(sim.y_range[0], sim.y_range[1] + 1e-6, 5))
    ax.tick_params(length=2.5, labelsize=7, colors="#374151")
    for s in ax.spines.values():
        s.set_color("#374151")
    ax.set_xlabel("x (m)", fontsize=8, color="#111827")
    ax.set_ylabel("y (m)", fontsize=8, color="#111827")

    for (ox, oy), orad in zip(geom.obstacle_xy, geom.obstacle_r):
        ax.add_patch(Circle((ox, oy), float(orad), facecolor=OBSTACLE_FACE,
                            edgecolor=OBSTACLE_EDGE, linewidth=0.8, zorder=1))

    member_set = set(members.tolist())
    for i in range(before.shape[0]):
        c = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        cp = _tint(c)
        bx, by, bth = before[i]
        axp, ayp, ath = after[i]

        if i in member_set:
            gx, gy = goals[i]
            # Desired action: dashed line in the robot's identity color,
            # robot (before) -> goal.
            ax.plot([bx, gx], [by, gy], color=c, linewidth=0.9,
                    linestyle=(0, (3, 2)), alpha=0.6, zorder=3)
            # Executed displacement (dotted connector, not a trajectory).
            ax.plot([bx, axp], [by, ayp], color="#6b7280", linewidth=0.9,
                    linestyle=(0, (1, 1.6)), alpha=0.8, zorder=4)
            # Before: vivid disc + vivid heading arrow.
            ax.add_patch(Circle((bx, by), rho, facecolor=c,
                                edgecolor="white", linewidth=0.6, zorder=5))
            _heading_arrow(ax, bx, by, bth, color=c, zorder=6)
            # After: same hue at 40% saturation, drawn exactly like `before`.
            ax.add_patch(Circle((axp, ayp), rho, facecolor=cp,
                                edgecolor="white", linewidth=0.6, zorder=6))
            _heading_arrow(ax, axp, ayp, ath, color=cp, zorder=7)
            # Goal in the robot's color.
            ax.plot(gx, gy, marker="*", markersize=10, markerfacecolor=c,
                    markeredgecolor="white", markeredgewidth=0.5,
                    linestyle="none", zorder=7)
            ax.annotate(str(i), xy=(bx, by),
                        xytext=(bx + 1.7 * rho, by + 1.7 * rho),
                        fontsize=6, color=c, ha="center", va="center",
                        zorder=8)
        else:
            # Bystander: stays in place but the coupled actuation rotates it —
            # one vivid disc, vivid before-arrow, desaturated after-arrow.
            ax.add_patch(Circle((bx, by), rho, facecolor=c,
                                edgecolor="white", linewidth=0.5, alpha=0.9,
                                zorder=4))
            _heading_arrow(ax, bx, by, bth, color=c, zorder=5)
            _heading_arrow(ax, axp, ayp, ath, color=cp, zorder=6)

    # Legend exemplar: robot 12's color throughout, chips at equal size.
    c12 = ROBOT_COLORS[12]
    key = [
        Line2D([], [], marker="o", markerfacecolor=c12,
               markeredgecolor="none", markersize=6.5, linestyle="none",
               label="before"),
        Line2D([], [], marker="o", markerfacecolor=_tint(c12),
               markeredgecolor="none", markersize=6.5, linestyle="none",
               label="after"),
        Line2D([], [], marker=r"$\rightarrow$", markersize=8,
               color=c12, markeredgewidth=0.0, linestyle="none",
               label="heading"),
        Line2D([], [], color=c12, linewidth=0.9,
               linestyle=(0, (3, 2)), label="desired (to goal)"),
        Line2D([], [], color="#6b7280", linewidth=0.9,
               linestyle=(0, (1, 1.6)), label="executed move"),
        Line2D([], [], marker="*", markerfacecolor=c12,
               markeredgecolor="white", markersize=9, linestyle="none",
               label="goal"),
    ]
    ax.legend(handles=key, loc="upper left", ncol=3, fontsize=6.5,
              frameon=True, framealpha=0.9, edgecolor="#d1d5db",
              borderpad=0.4, handletextpad=0.3, columnspacing=0.9)

    if not args.no_title:
        ax.set_title(
            f"seed {args.seed}  ·  coarse group {group} "
            f"(|G|={len(members)})  ·  {args.repeats} move(s)",
            fontsize=7.5, color="#374151", pad=4)

    for ext in args.formats:
        p = out_dir / f"coarse_move_s{args.seed}_g{group}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
