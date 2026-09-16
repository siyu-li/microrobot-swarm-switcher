"""
Learned per-robot precise cost-to-go for the MPC switcher leaf (value step 1).

A small regressor ``v_psi(feature_i) -> scalar`` predicts, per robot, the number
of remaining precise decisions until that robot reaches its goal (trained by
Monte-Carlo supervised learning on rollouts of the frozen precise GAT policy —
see ``robot_nav/collect_value_data.py`` / ``robot_nav/train_value.py``).  The
leaf evaluator sums the per-robot predictions over unreached robots and scales
by the per-robot decision price from the live cost table:

    h_precise(x) = precise_unit * selection_interval
                   * sum_i  v_psi(feature_i),   i unreached

which plugs into :meth:`ForwardModel.cost_to_go` as a selectable replacement for
the crude analytic heuristic ``alpha * sum_i ||p_i - goal_i||``.

Two feature variants share the same architecture and checkpoint format:

  * ``"embedding"`` — the frozen GAT per-robot 512-d embedding
    (``GATBackbone.get_embedding_and_actions``), which carries real
    robot-neighbor/obstacle attention;
  * ``"geometry"``  — the raw 11-col per-robot state row (px, py, cos, sin,
    dist, cos_err, sin_err, lin, ang, gx, gy), congestion-blind baseline.

The label unit is per-robot: labels treat the neighbors seen during collection
as FIXED (3 static robots), so this is a partial-congestion cost-to-go, not a
moving-agent one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

FEATURES = ("embedding", "geometry")


class PerRobotValue(nn.Module):
    """
    Per-robot cost-to-go regressor: feature vector -> remaining precise
    decisions (scalar, in decision units — multiply by ``precise_cost`` for the
    planner's cost units).

    Plain MLP following the DeepSetsHead phi pattern (Linear-ReLU stack with a
    final linear scalar head); no pooling here — pooling over robots happens in
    :class:`LearnedCostToGo` (and learned pooling is a later step).
    """

    def __init__(self, in_dim: int = 512, hidden: Sequence[int] = (256, 128),
                 dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(..., in_dim) -> (...,) predicted remaining decisions."""
        return self.net(x).squeeze(-1)


def save_value_checkpoint(
    path: str | Path,
    net: PerRobotValue,
    feature: str,
    in_dim: int,
    hidden: Sequence[int],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    precise_cost: float,
    dropout: float = 0.0,
    meta: dict | None = None,
) -> None:
    """Save the regressor with everything needed to run it standalone."""
    if feature not in FEATURES:
        raise ValueError(f"feature must be one of {FEATURES}, got {feature!r}")
    torch.save(
        {
            "state_dict": net.state_dict(),
            "feature": feature,
            "in_dim": int(in_dim),
            "hidden": list(hidden),
            "dropout": float(dropout),
            "x_mean": np.asarray(x_mean, dtype=np.float32),
            "x_std": np.asarray(x_std, dtype=np.float32),
            "precise_cost": float(precise_cost),
            "meta": meta or {},
        },
        Path(path),
    )


def load_value_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[PerRobotValue, dict]:
    """Load a checkpoint saved by :func:`save_value_checkpoint`."""
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    # Infer dropout from state_dict key indices when not explicitly stored.
    # Without dropout each hidden block occupies 2 indices (Linear + ReLU);
    # with dropout it occupies 3 (Linear + ReLU + Dropout).  The last Linear
    # sits at index 2*N (no dropout) or 3*N (with dropout) where N=len(hidden).
    dropout = ckpt.get("dropout", None)
    if dropout is None:
        max_idx = max(
            int(k.split(".")[1])
            for k in ckpt["state_dict"]
            if k.split(".")[1].isdigit()
        )
        n_hidden = len(ckpt["hidden"])
        dropout = 0.1 if max_idx == n_hidden * 3 else 0.0
    net = PerRobotValue(in_dim=ckpt["in_dim"], hidden=ckpt["hidden"], dropout=dropout)
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, ckpt


class LearnedCostToGo:
    """
    Selectable leaf evaluator for :meth:`ForwardModel.cost_to_go`:

        h_precise(x) = (precise_unit * selection_interval)
                       * sum_{i unreached}  v_psi(feature_i)

    ``v_psi`` predicts remaining precise *decisions* per robot; one decision
    drives one robot for ``selection_interval`` sub-steps at ``precise_unit``
    cost per sub-step, so the per-decision price comes from the **model's
    live cost table** (``model.cost.precise_unit * model.selection_interval``)
    — not from the checkpoint, which stores only the legacy ``precise_cost``
    metadata of its training run.

    Robots already within ``goal_threshold`` contribute 0 (cost-to-go 0 at
    goal); predictions are clamped at 0.  The sum pool is the project's chosen
    combination of the per-robot values (learned pooling is a later step).

    Callable as ``leaf_value(model, ms) -> float`` — the contract
    ``ForwardModel`` dispatches to when configured with a learned leaf.
    """

    def __init__(self, checkpoint_path: str | Path, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.net, ckpt = load_value_checkpoint(checkpoint_path, device=self.device)
        self.feature: str = ckpt["feature"]
        # Legacy metadata (training-time constant); pricing now comes from the
        # model's SwitcherCost at call time.
        self.precise_cost: float = float(ckpt["precise_cost"])
        self.x_mean = torch.as_tensor(ckpt["x_mean"], dtype=torch.float32, device=self.device)
        self.x_std = torch.as_tensor(ckpt["x_std"], dtype=torch.float32, device=self.device)

    def _features(self, model, ms) -> torch.Tensor:
        """Per-robot feature matrix (N, in_dim) for the leaf state."""
        rs = model.robot_state(ms)
        if self.feature == "embedding":
            _, h = model.backbone.get_embedding_and_actions(rs, model.obstacle_states)
            return h.to(self.device)
        return torch.as_tensor(rs, dtype=torch.float32, device=self.device)

    def __call__(self, model, ms) -> float:
        dist = model.goal_distances(ms)
        unreached = dist > model.goal_threshold
        if not unreached.any():
            return 0.0
        x = (self._features(model, ms) - self.x_mean) / self.x_std
        with torch.no_grad():
            v = self.net(x).clamp_min(0.0)             # (N,) decisions
        mask = torch.as_tensor(unreached, device=self.device)
        per_decision = model.cost.precise_unit * model.selection_interval
        return float(per_decision * v[mask].sum().item())
