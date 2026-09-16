"""
Bounded multiplicative value residual on a selectable base level.

    V(s) = B(s) * (1 + band * tanh(MLP([B(s)/h_scale, G0..G3])))

where the base ``B(s)`` is either

* ``"analytic"`` — ``α · Σ_{i unreached} d_i`` (the model's own analytic
  cost-to-go), or
* ``"geometry"`` — the per-robot learned precise cost-to-go
  (``capswitcher.rl.value_net.LearnedCostToGo``, feature="geometry"): its
  level is regressed on realized rollouts, so labels sit far closer to it
  than to the analytic α level (whose realized/level ratio measured ≈3.5–3.9
  — outside any modest band), at the price of one robot-state rebuild + net
  forward per leaf.  Its known blindness — per-robot independence, trained
  with neighbours frozen — is exactly what the G0..G3 scheduling globals
  (``features_v2.py``) are there to modulate.

The correction is hard-capped at ±``band``: a bad net can misrank leaves by
at most that factor.  The residual is signed (tanh), per the inadmissibility
analysis.  Labels are realized suffix costs of the executing switcher's own
episodes (harvested by ``eval_gaz14_lazy --log-pi-targets``), so V estimates
the deployed policy's cost — reference-policy cost, not an admissible bound.

:class:`LearnedValueResidual` satisfies the ``leaf_value(model, ms) ->
float`` contract of ``ForwardModel14.cost_to_go``.  One CPU forward per leaf
(plus the base evaluation).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from robot_nav.models.MARL.capswitcher_14.rl.search.feas_policy_net import (
    _Saveable,
)
from robot_nav.models.MARL.capswitcher_14.rl.search.features_v2 import (
    GLOBAL_FEAT_DIM,
    GroupFeatureBuilderV2,
)

BASES = ("analytic", "geometry")


class ValueResidualNet(_Saveable):
    """tanh-bounded modulation factor ``m ∈ (−1, 1)``; ``V = B(1 + band·m)``.

    The checkpoint freezes everything needed to reproduce the base at deploy:
    ``base`` (which evaluator), ``base_ckpt`` (the geometry checkpoint path,
    if any), ``h_scale`` (median training base level) and ``alpha`` (the
    analytic α of the label run, for a deploy-time mismatch warning).
    """

    def __init__(
        self,
        global_dim: int = GLOBAL_FEAT_DIM,
        hidden: int = 32,
        band: float = 0.5,
        h_scale: float = 5000.0,
        alpha: float | None = None,
        base: str = "analytic",
        base_ckpt: str | None = None,
    ) -> None:
        super().__init__()
        if base not in BASES:
            raise ValueError(f"base must be one of {BASES}, got {base!r}")
        self.config = dict(global_dim=global_dim, hidden=hidden, band=band,
                           h_scale=h_scale, alpha=alpha, base=base,
                           base_ckpt=base_ckpt)
        self.band = float(band)
        self.h_scale = float(h_scale)
        self.alpha = alpha
        self.base = base
        self.base_ckpt = base_ckpt
        self.mlp = nn.Sequential(
            nn.Linear(1 + global_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
        """h (B,), glob (B, global_dim) → modulation m (B,) in (−1, 1)."""
        x = torch.cat([(h / self.h_scale)[:, None], glob], dim=-1)
        return torch.tanh(self.mlp(x)).squeeze(-1)

    def value(self, h: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
        return h * (1.0 + self.band * self.forward(h, glob))


class LearnedValueResidual:
    """
    Leaf evaluator: ``leaf_value(model, ms) -> float``.

    The base comes from the checkpoint's ``base`` field: analytic uses the
    **model's own** α (level always matches the live cost table; warns once
    if it differs from the training α); geometry loads the frozen
    :class:`LearnedCostToGo` from ``base_ckpt`` (override with
    ``base_ckpt=`` when the stored path moved).
    """

    def __init__(
        self,
        ckpt_path: str,
        features: GroupFeatureBuilderV2,
        device: str = "cpu",
        base_ckpt: str | None = None,
    ) -> None:
        self.net = ValueResidualNet.load(ckpt_path,
                                         map_location=device).to(device)
        self.features = features
        self.device = device
        self._alpha_checked = False
        self.base_value = None
        if self.net.base == "geometry":
            from robot_nav.models.MARL.capswitcher.rl.value_net import (
                LearnedCostToGo,
            )

            path = base_ckpt or self.net.base_ckpt
            if not path:
                raise ValueError("geometry-based residual checkpoint lacks "
                                 "base_ckpt — pass base_ckpt= explicitly")
            self.base_value = LearnedCostToGo(path, device=device)

    def _base(self, model, ms, unreached, d) -> float:
        if self.base_value is not None:
            return float(self.base_value(model, ms))
        alpha = model.analytic_alpha()
        if not self._alpha_checked:
            self._alpha_checked = True
            trained = self.net.alpha
            if trained is not None and abs(trained - alpha) > 1e-6 * alpha:
                print(f"WARNING: value residual trained at alpha={trained:.3f}"
                      f" but model uses alpha={alpha:.3f} — the modulation "
                      "band was fit to a different ratio")
        return float(alpha * d[unreached].sum())

    def __call__(self, model, ms) -> float:
        d = model.goal_distances(ms)
        unreached = d > model.goal_threshold
        if not unreached.any():
            return 0.0
        h = self._base(model, ms, unreached, d)
        if h <= 0.0:
            return h
        glob = self.features.globals_only(ms.poses, model.goals)
        with torch.no_grad():
            v = self.net.value(
                torch.tensor([h], dtype=torch.float32, device=self.device),
                torch.as_tensor(glob, device=self.device)[None],
            )
        return float(v[0])
