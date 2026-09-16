"""
Shared helpers for extracting pre-decoder embeddings and attention weights
from a frozen GAT attention module.

All switcher-related code (RL env, supervised test, oracle data collection)
should call these instead of duplicating the forward-pass + reshape logic.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def _prepare_obstacle_tensor(
    obstacle_obs: Union[np.ndarray, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Convert obstacle observations to a batched tensor on *device*.

    Accepts either a numpy array (CPU) or an existing torch.Tensor
    (possibly already on the target device).  Always returns a 3-D
    tensor ``(1, N_obs, obs_dim)``.
    """
    if isinstance(obstacle_obs, torch.Tensor):
        t = obstacle_obs.to(device=device)
    else:
        t = torch.as_tensor(
            np.asarray(obstacle_obs), dtype=torch.float32, device=device,
        )
    if t.dim() == 2:
        t = t.unsqueeze(0)
    return t


def extract_embeddings(
    attention_module: nn.Module,
    robot_obs: np.ndarray,
    obstacle_obs: Union[np.ndarray, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the attention module and return **pre-decoder** embeddings.

    The pre-decoder embedding ``[self_embed || attn_aggregation]`` retains
    richer relational structure than the post-decoder output.

    Args:
        attention_module: The GAT ``AttentionObstacleOptimized`` module
            (e.g. ``policy.actor.attention``).
        robot_obs: Robot observations, shape ``(N, state_dim)``.
        obstacle_obs: Obstacle observations — numpy ``(N_obs, obs_dim)``
            **or** a pre-built ``torch.Tensor`` (to skip redundant copies).
        device: Torch device for computation.

    Returns:
        h: Pre-decoder per-robot embeddings ``(N, embed_dim)`` (detached).
        attn_rr: Robot-robot hard attention ``(N, N)``.
        attn_ro: Robot-obstacle hard attention ``(N, N_obs)``.
    """
    robot_t = torch.as_tensor(
        robot_obs, dtype=torch.float32, device=device,
    )
    if robot_t.dim() == 2:
        robot_t = robot_t.unsqueeze(0)

    obstacle_t = _prepare_obstacle_tensor(obstacle_obs, device)

    with torch.no_grad():
        (
            _H, _, _, _, _, _,
            hard_weights_rr, hard_weights_ro, _,
        ) = attention_module(robot_t, obstacle_t)

    N = robot_t.shape[1]
    pre_dec = attention_module._pre_decoder_embedding          # (B*N, 512)
    embed_dim = pre_dec.shape[-1]
    h = pre_dec.view(-1, N, embed_dim).squeeze(0)              # (N, embed_dim)
    attn_rr = hard_weights_rr.squeeze(0)                       # (N, N)
    attn_ro = hard_weights_ro.squeeze(0)                       # (N, N_obs)
    return h, attn_rr, attn_ro


def extract_embeddings_and_actions(
    actor: nn.Module,
    robot_obs: np.ndarray,
    obstacle_obs: Union[np.ndarray, torch.Tensor],
    device: torch.device,
    embedding_source: str = "decoder",
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single forward pass → raw actions **and** per-robot embeddings.

    Avoids running the attention module twice (once for actions, once for
    observation building).

    Args:
        actor: The ``ActorObstacle`` module (``policy.actor``).
        robot_obs: Robot observations ``(N, state_dim)``.
        obstacle_obs: Obstacle observations — numpy or torch.Tensor.
        device: Torch device for computation.
        embedding_source: Which per-robot embedding to return:

            * ``"decoder"`` (default) — the post-decoder embedding ``H`` that
              feeds the policy head (``attn_out``). Encodes "what the navigation
              policy wants to do here"; used by the CAPSwitcher.
            * ``"pre_decoder"`` — the earlier ``_pre_decoder_embedding``
              (``self_embed ⊕ attn_aggregation``), retained for ablation.

    Returns:
        raw_actions: ``(N, 2)`` numpy array on CPU.
        h: Per-robot embeddings ``(N, embed_dim)`` (detached) from the chosen
           source.
        attn_rr: Robot-robot hard attention ``(N, N)``.
        attn_ro: Robot-obstacle hard attention ``(N, N_obs)``.
    """
    if embedding_source not in ("decoder", "pre_decoder"):
        raise ValueError(
            f"embedding_source must be 'decoder' or 'pre_decoder', "
            f"got {embedding_source!r}"
        )

    robot_t = torch.as_tensor(
        robot_obs, dtype=torch.float32, device=device,
    ).unsqueeze(0)

    obstacle_t = _prepare_obstacle_tensor(obstacle_obs, device)

    with torch.no_grad():
        (
            H, _, _, _, _, _,
            hard_weights_rr, hard_weights_ro, _,
        ) = actor.attention(robot_t, obstacle_t)

        action = actor.policy_head(H)

    N = robot_t.shape[1]
    if embedding_source == "decoder":
        # H is the decoder output (attn_out) fed to the policy head: (B*N, 512).
        src = H
    else:
        src = actor.attention._pre_decoder_embedding               # (B*N, 512)
    embed_dim = src.shape[-1]
    h = src.view(-1, N, embed_dim).squeeze(0)                      # (N, embed_dim)
    attn_rr = hard_weights_rr.squeeze(0)                           # (N, N)
    attn_ro = hard_weights_ro.squeeze(0)                           # (N, N_obs)

    raw_actions = action.cpu().data.numpy().reshape(-1, 2)
    return raw_actions, h, attn_rr, attn_ro
