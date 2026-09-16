"""
Learned prior for the lazy Gumbel search: one tiny forward pass ranks all 23
edges — the cheap substitute for the ~23 transitions an eager expansion pays.

Architecture (settled: small graph-derived features only, never the 512-d GAT
embeddings):

* shared per-group trunk MLP over ``[group_feats ⊕ global_feats]`` (weight
  sharing across the 22 groups = permutation-consistent), with two heads:
  - **policy logit** (distilled from the search's root π′, masked KL), and
  - **clearance margin** (regression on the shield's exact ``clearance −
    d_safe`` labels harvested at every root — richer than a binary
    safe/unsafe, and thresholdable at deploy time without retraining);
* a separate small head maps the global context to the **precise** edge logit.

``LearnedPrior`` adapts the net to the search's prior contract:
``prior(model, ms, branches) -> logits`` plus the advisory
``feasibility(model, ms, branches)`` mask (predicted margin ≥ ``feas_margin``;
the precise edge is always feasible).  Predictions steer selection below the
root only — the root and every materialised edge use the real shield.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from robot_nav.models.MARL.capswitcher_14.rl.search.features import (
    GLOBAL_FEAT_DIM,
    GROUP_FEAT_DIM,
    GroupFeatureBuilder,
)


class PriorNet(nn.Module):
    """
    Args:
        group_feat_dim / global_feat_dim: feature sizes (defaults from
            ``features.py``).
        hidden: trunk width.
    """

    def __init__(
        self,
        group_feat_dim: int = GROUP_FEAT_DIM,
        global_feat_dim: int = GLOBAL_FEAT_DIM,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.group_feat_dim = group_feat_dim
        self.global_feat_dim = global_feat_dim
        self.trunk = nn.Sequential(
            nn.Linear(group_feat_dim + global_feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, 1)
        self.clearance_head = nn.Linear(hidden, 1)
        self.precise_head = nn.Sequential(
            nn.Linear(global_feat_dim, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self, group_feats: torch.Tensor, global_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            group_feats:  (B, K, group_feat_dim)
            global_feats: (B, global_feat_dim)

        Returns:
            logits:            (B, K + 1) — 22 coarse + precise (last)
            clearance_margin:  (B, K) — predicted ``clearance − d_safe`` (m)
        """
        B, K, _ = group_feats.shape
        ctx = global_feats[:, None, :].expand(B, K, self.global_feat_dim)
        z = self.trunk(torch.cat([group_feats, ctx], dim=-1))    # (B, K, H)
        coarse_logits = self.policy_head(z).squeeze(-1)          # (B, K)
        margin = self.clearance_head(z).squeeze(-1)              # (B, K)
        precise_logit = self.precise_head(global_feats)          # (B, 1)
        logits = torch.cat([coarse_logits, precise_logit], dim=1)
        return logits, margin

    # -- checkpointing --------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "group_feat_dim": self.group_feat_dim,
                "global_feat_dim": self.global_feat_dim,
            },
            str(path),
        )

    @classmethod
    def load(cls, path: str | Path, map_location="cpu") -> "PriorNet":
        ckpt = torch.load(str(path), map_location=map_location)
        net = cls(
            group_feat_dim=ckpt["group_feat_dim"],
            global_feat_dim=ckpt["global_feat_dim"],
        )
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        return net


class LearnedPrior:
    """
    Search-facing wrapper: features → one net forward → logits + feasibility.

    The per-node result is cached on the state object's ``id`` so the search's
    separate ``__call__`` / ``feasibility`` lookups at the same node cost one
    forward pass total.

    Args:
        net:          trained :class:`PriorNet` (eval mode).
        features:     :class:`GroupFeatureBuilder` (same move-group order as
                      the branch stubs).
        feas_margin:  advisory threshold (m) on the predicted clearance margin;
                      raise it for a more conservative in-tree mask.
        device:       torch device for the forward pass.
    """

    def __init__(
        self,
        net: PriorNet,
        features: GroupFeatureBuilder,
        feas_margin: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.net = net.to(device).eval()
        self.features = features
        self.feas_margin = float(feas_margin)
        self.device = device
        self._cache_key: int | None = None
        self._cache: tuple[np.ndarray, np.ndarray] | None = None

    def _evaluate(self, model, ms) -> tuple[np.ndarray, np.ndarray]:
        key = id(ms)
        if key != self._cache_key:
            gf, glf = self.features(model, ms)
            with torch.no_grad():
                logits, margin = self.net(
                    torch.as_tensor(gf, device=self.device)[None],
                    torch.as_tensor(glf, device=self.device)[None],
                )
            self._cache_key = key
            self._cache = (
                logits[0].cpu().numpy().astype(np.float64),
                margin[0].cpu().numpy().astype(np.float64),
            )
        return self._cache

    def __call__(self, model, ms, branches) -> np.ndarray:
        logits, _ = self._evaluate(model, ms)
        assert len(branches) == logits.shape[0], (
            f"branch/logit mismatch: {len(branches)} vs {logits.shape[0]}"
        )
        return logits

    def feasibility(self, model, ms, branches) -> np.ndarray:
        _, margin = self._evaluate(model, ms)
        feas = np.ones(len(branches), dtype=bool)
        feas[: margin.shape[0]] = margin >= self.feas_margin
        return feas
