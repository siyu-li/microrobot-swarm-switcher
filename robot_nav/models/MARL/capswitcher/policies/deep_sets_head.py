"""
DeepSetsHead: permutation-invariant readout over per-robot embeddings.

Motivation
----------
The frozen GAT backbone was trained on a *per-robot* navigation objective, so
its per-robot embeddings are individually meaningful but were never pushed into
a space where a *fixed* pooling operator (plain mean/max on the raw 512-dim
vectors) yields a meaningful swarm-level summary.  A Deep Sets readout solves
this by applying a *learned* per-robot transform φ before aggregation:

    out = ρ( aggregate_i φ(h_i) )

φ maps each robot embedding into a space where aggregation is meaningful; the
aggregation is permutation-invariant; ρ maps the swarm summary to the output
(Q-values for the switcher, or a scalar for the deferred probe experiment).

This module is deliberately task-neutral (configurable ``out_dim``) so the same
architecture can back both the DQN switcher (``out_dim=2``) and a supervised
swarm-target probe (``out_dim=1``).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


_AGGREGATIONS = ("mean", "sum", "max", "sum_max")


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: Optional[int] = None) -> nn.Sequential:
    """Build an MLP: Linear→ReLU per hidden width, optional final Linear (no act)."""
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    if out_dim is not None:
        layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class DeepSetsHead(nn.Module):
    """
    Deep Sets readout over a set of per-robot embeddings.

    Args:
        embed_dim:   Per-robot input embedding dimension. Default 512.
        phi_dims:    Hidden widths of the per-robot encoder φ. The last entry
                     is φ's output width. Default (256, 128).
        rho_dims:    Hidden widths of the post-aggregation network ρ.
                     Default (128,).
        out_dim:     Output dimension (2 for the binary switcher Q-values,
                     1 for a scalar probe target). Default 2.
        aggregation: One of {"mean", "sum", "max", "sum_max"}.  "sum_max"
                     concatenates sum- and max-pooled features (ρ input is
                     2×φ_out): sum captures density/count, max captures the
                     worst-case (most-constrained) robot. Default "sum_max".
    """

    def __init__(
        self,
        embed_dim: int = 512,
        phi_dims: Sequence[int] = (256, 128),
        rho_dims: Sequence[int] = (128,),
        out_dim: int = 2,
        aggregation: str = "sum_max",
    ) -> None:
        super().__init__()
        if aggregation not in _AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {_AGGREGATIONS}, got {aggregation!r}"
            )
        self.aggregation = aggregation

        self.phi = _mlp(embed_dim, phi_dims)            # (B, N, phi_out)
        phi_out = phi_dims[-1]
        agg_dim = phi_out * (2 if aggregation == "sum_max" else 1)
        self.rho = _mlp(agg_dim, rho_dims, out_dim)     # (B, out_dim)

    def _aggregate(
        self, z: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Permutation-invariant pool over the robot dimension (dim=1).

        Args:
            z:    (B, N, phi_out) per-robot features.
            mask: Optional (B, N) float/bool mask; 1 = active robot. Masked
                  robots are excluded from the aggregation.

        Returns:
            (B, agg_dim) pooled features.
        """
        if mask is not None:
            m = mask.unsqueeze(-1).to(z.dtype)          # (B, N, 1)

        def _sum() -> torch.Tensor:
            return (z * m).sum(dim=1) if mask is not None else z.sum(dim=1)

        def _mean() -> torch.Tensor:
            if mask is not None:
                denom = m.sum(dim=1).clamp_min(1.0)
                return (z * m).sum(dim=1) / denom
            return z.mean(dim=1)

        def _max() -> torch.Tensor:
            if mask is not None:
                neg_inf = torch.finfo(z.dtype).min
                z_masked = z.masked_fill(m == 0, neg_inf)
                return z_masked.max(dim=1).values
            return z.max(dim=1).values

        if self.aggregation == "sum":
            return _sum()
        if self.aggregation == "mean":
            return _mean()
        if self.aggregation == "max":
            return _max()
        # sum_max
        return torch.cat([_sum(), _max()], dim=-1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, N, embed_dim) per-robot embeddings.
            mask: Optional (B, N) active-robot mask.

        Returns:
            (B, out_dim) output (Q-values / probe prediction).
        """
        z = self.phi(x)                  # (B, N, phi_out)
        pooled = self._aggregate(z, mask)  # (B, agg_dim)
        return self.rho(pooled)          # (B, out_dim)
