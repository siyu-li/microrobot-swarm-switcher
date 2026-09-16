"""
GATBackbone: frozen GAT backbone wrapper for CAPSwitcher.

Loads a pre-trained TD3Obstacle actor, freezes all parameters, and exposes
two public methods:

  - prepare_state()              — constructs 11-dim per-robot state vectors
                                   (delegates to TD3Obstacle.prepare_state)
  - get_embedding_and_actions()  — single frozen forward pass returning both
                                   the precise GAT actions and the pre-decoder
                                   embeddings (N, 512)
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import TD3Obstacle
from robot_nav.models.MARL.capswitcher.embedding_utils import (
    extract_embeddings_and_actions,
)


class GATBackbone:
    """
    Frozen GAT backbone for the CAPSwitcher.

    Wraps a TD3Obstacle instance loaded in ``inference_only=True`` mode
    (no optimizers, no TensorBoard writer).  All actor and actor_target
    parameters are frozen (``requires_grad=False``) and set to eval mode
    immediately after loading.

    Args:
        checkpoint_path: Path prefix for saved TD3Obstacle weights, e.g.
            ``Path("robot_nav/models/MARL/marlTD3/checkpoint/"
                   "obstacle_6robots_v4/TD3-MARL-obstacle-6robots-reward6")``.
            The loader appends ``_actor.pth``, ``_actor_target.pth``, etc.
        num_robots:    Number of robot agents.
        num_obstacles: Number of obstacle nodes in the graph.
        device:        Torch device for all tensors and models.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        num_robots: int,
        num_obstacles: int,
        device: torch.device,
        embedding_source: str = "decoder",
    ) -> None:
        self.device = device
        self.num_robots = num_robots
        self.embedding_source = embedding_source

        checkpoint_path = Path(checkpoint_path)
        directory = checkpoint_path.parent
        filename = checkpoint_path.name

        # Load in inference-only mode: skips optimizer + SummaryWriter creation.
        self._td3 = TD3Obstacle(
            state_dim=11,
            action_dim=2,
            max_action=1,
            num_robots=num_robots,
            num_obstacles=num_obstacles,
            obstacle_state_dim=4,
            device=device,
            load_model=True,
            load_model_name=filename,
            load_directory=directory,
            inference_only=True,
        )

        # Freeze every parameter in both actor and actor_target.
        for model in (self._td3.actor, self._td3.actor_target):
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()

        print(
            f"[GATBackbone] Loaded and frozen: {checkpoint_path}\n"
            f"  Robots={num_robots}, Obstacles={num_obstacles}, Device={device}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_state(self, poses, distance, cos, sin, collision, action, goal_positions):
        """
        Build per-robot 11-dim state vectors from raw simulator outputs.

        Delegates directly to ``TD3Obstacle.prepare_state`` — same signature
        and return values.

        Returns:
            states:   List of N state vectors (length 11 each).
            terminal: List of N bool collision flags.
        """
        return self._td3.prepare_state(
            poses, distance, cos, sin, collision, action, goal_positions
        )

    def get_embedding_and_actions(
        self,
        robot_state: np.ndarray,
        obstacle_states: np.ndarray,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """
        Single frozen forward pass returning precise actions and embeddings.

        Runs ``actor.attention`` once (no_grad) to populate
        ``_pre_decoder_embedding``, then reads it back.  No second forward
        pass is needed.

        Args:
            robot_state:     (N, 11) per-robot state array.
            obstacle_states: (M, 4) obstacle observation array.

        Returns:
            raw_actions: (N, 2) numpy array in actor output space (Tanh,
                         values ∈ [−1, 1]).  Convert to sim-input via
                         ``lin_vel = (raw[0]+1)/4, ang_vel = raw[1]``.
            h:           (N, 512) per-robot embeddings as a CPU tensor
                         (detached, float32).  Source set by
                         ``self.embedding_source`` ("decoder" by default).
        """
        raw_actions, h, _, _ = extract_embeddings_and_actions(
            actor=self._td3.actor,
            robot_obs=np.asarray(robot_state, dtype=np.float32),
            obstacle_obs=obstacle_states,
            device=self.device,
            embedding_source=self.embedding_source,
        )
        return raw_actions, h.cpu()

    # ------------------------------------------------------------------
    # Direct access for diagnostics
    # ------------------------------------------------------------------

    @property
    def actor(self) -> nn.Module:
        """The frozen ``ActorObstacle`` (for diagnostics / unit tests)."""
        return self._td3.actor
