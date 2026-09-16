"""
Split feasibility / group-scheduling prior for the lazy Gumbel search.

Two nets over the v2 features (``features_v2.py``), replacing the joint
``PriorNet`` of ``prior_net.py``:

* :class:`FeasNet14` — shared per-group MLP over the 11 feasibility features
  → P(shield vet passes).  Trained with BCE on the root vet outcomes
  (``safe``) harvested by ``eval_gaz14_lazy --log-pi-targets``; the
  probability is both an input to the policy and, thresholded, the advisory
  in-tree legality mask.
* :class:`GroupPolicyNet14` — shared per-group MLP over
  ``[12 policy features ‖ feas probability ‖ 4 global context]`` → one coarse
  logit per group; a separate head maps the global context alone to the
  precise logit (softmax makes the comparison relative, so "precise wins
  when every coarse row scores badly" needs no explicit coupling).  Trained
  with masked KL against the search's improved root policy π′.

Weight sharing across the 22 rows keeps both nets permutation-consistent and
group-table-agnostic (nothing is indexed by group id — scaling past 14 robots
changes only the feature builder's table).

:class:`LearnedFeasPolicyPrior` adapts the pair to the search contract
``prior(model, ms, branches) -> logits`` + advisory
``feasibility(model, ms, branches)`` (feas probability ≥ ``feas_threshold``;
the precise edge is always feasible).  One cached forward per node serves
both lookups.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from robot_nav.models.MARL.capswitcher_14.rl.search.features_v2 import (
    FEAS_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    GroupFeatureBuilderV2,
    POLICY_BASE_DIM,
)


class _Saveable(nn.Module):
    """save/load with the constructor config embedded in the checkpoint."""

    config: dict

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        torch.save({"state_dict": self.state_dict(), "config": self.config,
                    "extra": extra or {}}, str(path))

    @classmethod
    def load(cls, path: str | Path, map_location="cpu"):
        ckpt = torch.load(str(path), map_location=map_location,
                          weights_only=False)
        net = cls(**ckpt["config"])
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        return net


class FeasNet14(_Saveable):
    """Per-group feasibility logit; ``sigmoid`` gives P(vet passes)."""

    def __init__(self, feat_dim: int = FEAS_FEAT_DIM, hidden: int = 64) -> None:
        super().__init__()
        self.config = dict(feat_dim=feat_dim, hidden=hidden)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        """rows (..., K, feat_dim) → logits (..., K)."""
        return self.mlp(rows).squeeze(-1)


class GroupPolicyNet14(_Saveable):
    """Group-scheduling policy: 22 coarse logits + 1 precise logit."""

    def __init__(
        self,
        base_dim: int = POLICY_BASE_DIM,
        global_dim: int = GLOBAL_FEAT_DIM,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.config = dict(base_dim=base_dim, global_dim=global_dim,
                           hidden=hidden)
        self.global_dim = global_dim
        self.trunk = nn.Sequential(
            nn.Linear(base_dim + 1 + global_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.precise_head = nn.Sequential(
            nn.Linear(global_dim, hidden // 4), nn.ReLU(),
            nn.Linear(hidden // 4, 1),
        )

    def forward(
        self,
        base_rows: torch.Tensor,
        feas_prob: torch.Tensor,
        glob: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            base_rows: (B, K, base_dim) policy features 0–11.
            feas_prob: (B, K) feasibility probabilities (feature 12).
            glob:      (B, global_dim) global context.

        Returns:
            logits (B, K + 1) — 22 coarse + precise (last).
        """
        B, K, _ = base_rows.shape
        ctx = glob[:, None, :].expand(B, K, self.global_dim)
        rows = torch.cat([base_rows, feas_prob[..., None], ctx], dim=-1)
        coarse = self.trunk(rows).squeeze(-1)                  # (B, K)
        precise = self.precise_head(glob)                      # (B, 1)
        return torch.cat([coarse, precise], dim=1)


class LearnedFeasPolicyPrior:
    """
    Search-facing wrapper: v2 features → feas forward → policy forward.

    Cached on the node state's ``id`` so the search's separate ``__call__`` /
    ``feasibility`` lookups at one node cost a single evaluation.

    Args:
        feas_net:       trained :class:`FeasNet14`.
        policy_net:     trained :class:`GroupPolicyNet14`.
        features:       :class:`GroupFeatureBuilderV2` (same move-group order
                        as the branch stubs).
        feas_threshold: advisory in-tree mask — coarse edges with predicted
                        P(feasible) below this are steered around (hard mask
                        below ~0.1 per the design; the root and every
                        materialised edge still use the real shield).
        device:         torch device for the forward passes.
    """

    def __init__(
        self,
        feas_net: FeasNet14,
        policy_net: GroupPolicyNet14,
        features: GroupFeatureBuilderV2,
        feas_threshold: float = 0.1,
        device: str = "cpu",
    ) -> None:
        self.feas_net = feas_net.to(device).eval()
        self.policy_net = policy_net.to(device).eval()
        self.features = features
        self.feas_threshold = float(feas_threshold)
        self.device = device
        self._cache_key: int | None = None
        self._cache: tuple[np.ndarray, np.ndarray] | None = None

    def _evaluate(self, model, ms) -> tuple[np.ndarray, np.ndarray]:
        key = id(ms)
        if key != self._cache_key:
            f = self.features(model, ms)
            with torch.no_grad():
                feas_rows = torch.as_tensor(
                    f["feas"], device=self.device)[None]
                base_rows = torch.as_tensor(
                    f["policy"], device=self.device)[None]
                glob = torch.as_tensor(f["glob"], device=self.device)[None]
                prob = torch.sigmoid(self.feas_net(feas_rows))
                logits = self.policy_net(base_rows, prob, glob)
            self._cache_key = key
            self._cache = (
                logits[0].cpu().numpy().astype(np.float64),
                prob[0].cpu().numpy().astype(np.float64),
            )
        return self._cache

    def __call__(self, model, ms, branches) -> np.ndarray:
        logits, _ = self._evaluate(model, ms)
        assert len(branches) == logits.shape[0], (
            f"branch/logit mismatch: {len(branches)} vs {logits.shape[0]}"
        )
        return logits

    def feasibility(self, model, ms, branches) -> np.ndarray:
        _, prob = self._evaluate(model, ms)
        feas = np.ones(len(branches), dtype=bool)
        feas[: prob.shape[0]] = prob >= self.feas_threshold
        return feas
