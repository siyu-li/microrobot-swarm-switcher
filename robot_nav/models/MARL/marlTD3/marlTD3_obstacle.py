"""
TD3 Multi-Agent RL with Obstacle Graph Nodes.

This module extends marlTD3.py to include static obstacle nodes in the graph
attention network. Obstacles are represented as graph nodes that only send
messages - they do not have actors or produce actions.

Key differences from marlTD3.py:
- Actor/Critic use AttentionObstacle instead of Attention
- State includes obstacle information
- Only robot nodes produce actions
- Hard attention supervision includes robot-obstacle edges
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

# from robot_nav.models.MARL.Attention.iga_obstacle import AttentionObstacle
from robot_nav.models.MARL.Attention.iga_obstacle_optimized import AttentionObstacleOptimized as AttentionObstacle

class ActorObstacle(nn.Module):
    """
    Policy network for multi-agent control with obstacle-aware attention encoder.

    The actor encodes inter-agent and robot-obstacle context via AttentionObstacle
    and maps the attended embedding to continuous actions for robots only.

    Args:
        action_dim (int): Number of action dimensions per robot.
        embedding_dim (int): Dimensionality of the attention embedding.

    Attributes:
        attention (AttentionObstacle): Attention encoder with obstacle nodes.
        policy_head (nn.Sequential): MLP mapping attended embeddings to actions.
    """

    def __init__(self, action_dim, embedding_dim):
        super().__init__()
        self.attention = AttentionObstacle(embedding_dim)
        self.policy_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 300),
            nn.LeakyReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh(),
        )

    def forward(self, robot_obs, obstacle_obs, detach_attn=False):
        """
        Run the actor to produce actions and attention diagnostics.

        Args:
            robot_obs (Tensor): Robot observations of shape (B, N_robots, robot_state_dim).
            obstacle_obs (Tensor): Obstacle observations of shape (B, N_obs, obstacle_state_dim).
            detach_attn (bool, optional): If True, detaches attention features before
                the policy head. Defaults to False.

        Returns:
            tuple:
                action (Tensor): Predicted actions, shape (B*N_robots, action_dim).
                hard_logits_rr (Tensor): Robot-robot hard attention logits.
                hard_logits_ro (Tensor): Robot-obstacle hard attention logits.
                dist_rr (Tensor): Unnormalized robot-robot distances.
                dist_ro (Tensor): Unnormalized robot-obstacle distances.
                mean_entropy (Tensor): Mean soft-attention entropy.
                hard_weights_rr (Tensor): Binary robot-robot hard mask.
                hard_weights_ro (Tensor): Binary robot-obstacle hard mask.
                combined_weights (Tensor): Soft weights for visualization.
        """
        (
            attn_out,
            hard_logits_rr,
            hard_logits_ro,
            dist_rr,
            dist_ro,
            mean_entropy,
            hard_weights_rr,
            hard_weights_ro,
            combined_weights,
        ) = self.attention(robot_obs, obstacle_obs)

        if detach_attn:
            attn_out = attn_out.detach()

        action = self.policy_head(attn_out)
        return (
            action,
            hard_logits_rr,
            hard_logits_ro,
            dist_rr,
            dist_ro,
            mean_entropy,
            hard_weights_rr,
            hard_weights_ro,
            combined_weights,
        )


class CriticObstacle(nn.Module):
    """
    Twin Q-value critic with obstacle-aware attention encoding.

    Computes two independent Q estimates from attended embeddings and
    concatenated actions, following the TD3 design.

    Args:
        action_dim (int): Number of action dimensions per robot.
        embedding_dim (int): Dimensionality of the attention embedding.
    """

    def __init__(self, action_dim, embedding_dim):
        super(CriticObstacle, self).__init__()
        self.embedding_dim = embedding_dim
        self.attention = AttentionObstacle(embedding_dim)

        # Q1 network
        self.layer_1 = nn.Linear(self.embedding_dim * 2, 400)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2_s = nn.Linear(400, 300)
        nn.init.kaiming_uniform_(self.layer_2_s.weight, nonlinearity="leaky_relu")
        self.layer_2_a = nn.Linear(action_dim, 300)
        nn.init.kaiming_uniform_(self.layer_2_a.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, 1)
        nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="leaky_relu")

        # Q2 network
        self.layer_4 = nn.Linear(self.embedding_dim * 2, 400)
        nn.init.kaiming_uniform_(self.layer_4.weight, nonlinearity="leaky_relu")
        self.layer_5_s = nn.Linear(400, 300)
        nn.init.kaiming_uniform_(self.layer_5_s.weight, nonlinearity="leaky_relu")
        self.layer_5_a = nn.Linear(action_dim, 300)
        nn.init.kaiming_uniform_(self.layer_5_a.weight, nonlinearity="leaky_relu")
        self.layer_6 = nn.Linear(300, 1)
        nn.init.kaiming_uniform_(self.layer_6.weight, nonlinearity="leaky_relu")

    def forward(self, robot_obs, obstacle_obs, action):
        """
        Compute twin Q values from attended embeddings and actions.

        Args:
            robot_obs (Tensor): Robot observations of shape (B, N_robots, robot_state_dim).
            obstacle_obs (Tensor): Obstacle observations of shape (B, N_obs, obstacle_state_dim).
            action (Tensor): Actions of shape (B*N_robots, action_dim).

        Returns:
            tuple:
                Q1 (Tensor): First Q-value estimate, shape (B*N_robots, 1).
                Q2 (Tensor): Second Q-value estimate, shape (B*N_robots, 1).
                mean_entropy (Tensor): Mean soft-attention entropy.
                hard_logits_rr (Tensor): Robot-robot hard attention logits.
                hard_logits_ro (Tensor): Robot-obstacle hard attention logits.
                dist_rr (Tensor): Unnormalized robot-robot distances.
                dist_ro (Tensor): Unnormalized robot-obstacle distances.
                hard_weights_rr (Tensor): Binary robot-robot hard mask.
                hard_weights_ro (Tensor): Binary robot-obstacle hard mask.
        """
        (
            embedding_with_attention,
            hard_logits_rr,
            hard_logits_ro,
            dist_rr,
            dist_ro,
            mean_entropy,
            hard_weights_rr,
            hard_weights_ro,
            _,
        ) = self.attention(robot_obs, obstacle_obs)

        # Q1
        s1 = F.leaky_relu(self.layer_1(embedding_with_attention))
        s1 = F.leaky_relu(self.layer_2_s(s1) + self.layer_2_a(action))
        q1 = self.layer_3(s1)

        # Q2
        s2 = F.leaky_relu(self.layer_4(embedding_with_attention))
        s2 = F.leaky_relu(self.layer_5_s(s2) + self.layer_5_a(action))
        q2 = self.layer_6(s2)

        return (
            q1,
            q2,
            mean_entropy,
            hard_logits_rr,
            hard_logits_ro,
            dist_rr,
            dist_ro,
            hard_weights_rr,
            hard_weights_ro,
        )


class TD3Obstacle:
    """
    Twin Delayed DDPG (TD3) agent with obstacle graph nodes.

    Extends TD3 to handle obstacle nodes in the graph attention network.
    Obstacles are static nodes that only send messages to robot nodes.

    Args:
        state_dim (int): Per-robot state dimension.
        action_dim (int): Per-robot action dimension.
        max_action (float): Action clip magnitude.
        device (torch.device): Target device for models and tensors.
        num_robots (int): Number of robot agents.
        num_obstacles (int): Number of obstacle nodes.
        obstacle_state_dim (int): Per-obstacle state dimension. Default 4.
        lr_actor (float, optional): Actor learning rate. Defaults to 1e-4.
        lr_critic (float, optional): Critic learning rate. Defaults to 3e-4.
        save_every (int, optional): Save frequency in training iterations. Defaults to 0.
        load_model (bool, optional): If True, loads weights on init. Defaults to False.
        save_directory (Path, optional): Directory for saving checkpoints.
        model_name (str, optional): Base filename for checkpoints.
        load_model_name (str or None, optional): Filename base to load.
        load_directory (Path, optional): Directory to load checkpoints from.
        inference_only (bool, optional): If True, skips SummaryWriter and optimizer
            creation. Use for inference-only contexts (e.g. worker processes,
            evaluation scripts). Defaults to False.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        num_robots,
        num_obstacles,
        obstacle_state_dim=4,
        lr_actor=1e-4,
        lr_critic=3e-4,
        save_every=0,
        load_model=False,
        save_directory=Path("robot_nav/models/MARL/marlTD3/checkpoint"),
        model_name="marlTD3_obstacle",
        load_model_name=None,
        load_directory=Path("robot_nav/models/MARL/marlTD3/checkpoint"),
        inference_only=False,
    ):
        self.num_robots = num_robots
        self.num_obstacles = num_obstacles
        self.device = device
        self.state_dim = state_dim
        self.obstacle_state_dim = obstacle_state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # Initialize Actor networks
        self.actor = ActorObstacle(action_dim, embedding_dim=256).to(self.device)
        self.actor_target = ActorObstacle(action_dim, embedding_dim=256).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        if not inference_only:
            self.attn_params = list(self.actor.attention.parameters())
            self.policy_params = list(self.actor.policy_head.parameters())
            self.actor_optimizer = torch.optim.Adam(
                self.policy_params + self.attn_params, lr=lr_actor
            )

        # Initialize Critic networks
        self.critic = CriticObstacle(action_dim, embedding_dim=256).to(self.device)
        self.critic_target = CriticObstacle(action_dim, embedding_dim=256).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        if not inference_only:
            self.critic_optimizer = torch.optim.Adam(
                params=self.critic.parameters(), lr=lr_critic
            )

        # Logging and saving
        self.writer = None if inference_only else SummaryWriter(comment=model_name)
        self.iter_count = 0
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory

        if load_model_name is None:
            load_model_name = model_name
        if load_model:
            self.load(filename=load_model_name, directory=load_directory)

    def get_action(self, robot_obs, obstacle_obs, add_noise):
        """
        Compute an action for the given observations, with optional exploration noise.

        Args:
            robot_obs (np.ndarray): Robot observations of shape (N_robots, state_dim).
            obstacle_obs (np.ndarray): Obstacle observations of shape (N_obs, obstacle_state_dim).
            add_noise (bool): If True, adds Gaussian exploration noise.

        Returns:
            tuple:
                action (np.ndarray): Actions reshaped to (N_robots, action_dim).
                combined_weights (Tensor): Soft attention weights for visualization.
        """
        action, combined_weights = self.act(robot_obs, obstacle_obs)
        if add_noise:
            noise = np.random.normal(0, 0.5, size=action.shape)
            noise = [n / 4 if i % 2 else n for i, n in enumerate(noise)]
            action = (action + noise).clip(-self.max_action, self.max_action)
        return action.reshape(-1, 2), combined_weights

    def act(self, robot_obs, obstacle_obs):
        """
        Compute the deterministic action from the current policy.

        Args:
            robot_obs (np.ndarray): Robot observations.
            obstacle_obs (np.ndarray): Obstacle observations.

        Returns:
            tuple:
                action (np.ndarray): Flattened action vector.
                combined_weights (Tensor): Soft attention weights.
        """
        robot_state = torch.Tensor(robot_obs).to(self.device)
        obstacle_state = torch.Tensor(obstacle_obs).to(self.device)

        action, _, _, _, _, _, _, _, combined_weights = self.actor(
            robot_state, obstacle_state
        )
        return action.cpu().data.numpy().flatten(), combined_weights

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        bce_weight=0.01,
        entropy_weight=1,
        connection_proximity_threshold_rr=4.0,
        connection_proximity_threshold_ro=2.0,
        max_grad_norm=7.0,
    ):
        """
        Run a TD3 training loop over sampled mini-batches.

        Args:
            replay_buffer: Buffer supporting sample_batch() -> (robot_states, obstacle_states, ...).
            iterations (int): Number of gradient steps.
            batch_size (int): Mini-batch size.
            discount (float, optional): Discount factor. Defaults to 0.99.
            tau (float, optional): Target network update rate. Defaults to 0.005.
            policy_noise (float, optional): Std of target policy noise. Defaults to 0.2.
            noise_clip (float, optional): Clipping range for target noise. Defaults to 0.5.
            policy_freq (int, optional): Actor update period. Defaults to 2.
            bce_weight (float, optional): Weight for hard-connection BCE loss. Defaults to 0.01.
            entropy_weight (float, optional): Weight for attention entropy bonus. Defaults to 1.
            connection_proximity_threshold_rr (float, optional): Distance threshold for
                robot-robot BCE supervision. Defaults to 4.0.
            connection_proximity_threshold_ro (float, optional): Distance threshold for
                robot-obstacle BCE supervision. Defaults to 2.0.
            max_grad_norm (float, optional): Gradient clipping norm. Defaults to 7.0.
        """
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        av_critic_loss = 0
        av_critic_entropy = []
        av_actor_entropy = []
        av_actor_loss = 0
        av_critic_bce_loss_rr = []
        av_critic_bce_loss_ro = []
        av_actor_bce_loss_rr = []
        av_actor_bce_loss_ro = []

        for it in range(iterations):
            # Sample batch
            (
                batch_robot_states,
                batch_obstacle_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_robot_states,
                batch_next_obstacle_states,
                batch_active_masks,
            ) = replay_buffer.sample_batch(batch_size)

            robot_state = (
                torch.Tensor(batch_robot_states)
                .to(self.device)
                .view(batch_size, self.num_robots, self.state_dim)
            )
            obstacle_state = (
                torch.Tensor(batch_obstacle_states)
                .to(self.device)
                .view(batch_size, self.num_obstacles, self.obstacle_state_dim)
            )
            next_robot_state = (
                torch.Tensor(batch_next_robot_states)
                .to(self.device)
                .view(batch_size, self.num_robots, self.state_dim)
            )
            next_obstacle_state = (
                torch.Tensor(batch_next_obstacle_states)
                .to(self.device)
                .view(batch_size, self.num_obstacles, self.obstacle_state_dim)
            )
            action = (
                torch.Tensor(batch_actions)
                .to(self.device)
                .view(batch_size * self.num_robots, self.action_dim)
            )
            reward = (
                torch.Tensor(batch_rewards)
                .to(self.device)
                .view(batch_size * self.num_robots, 1)
            )
            done = (
                torch.Tensor(batch_dones)
                .to(self.device)
                .view(batch_size * self.num_robots, 1)
            )
            active_mask = (
                torch.Tensor(batch_active_masks.astype(float))
                .to(self.device)
                .view(batch_size * self.num_robots, 1)
            )

            # Target action
            with torch.no_grad():
                next_action, _, _, _, _, _, _, _, _ = self.actor_target(
                    next_robot_state, next_obstacle_state, detach_attn=True
                )

            # Target smoothing
            noise = (
                torch.Tensor(batch_actions)
                .data.normal_(0, policy_noise)
                .to(self.device)
            ).reshape(-1, 2)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Target Q values
            with torch.no_grad():
                target_Q1, target_Q2, _, _, _, _, _, _, _ = self.critic_target(
                    next_robot_state, next_obstacle_state, next_action
                )
                target_Q = torch.min(target_Q1, target_Q2)
                av_Q += target_Q.mean().item()
                max_Q = max(max_Q, target_Q.max().item())
                target_Q = reward + (1 - done) * discount * target_Q

            # Critic update
            (
                current_Q1,
                current_Q2,
                mean_entropy,
                hard_logits_rr,
                hard_logits_ro,
                dist_rr,
                dist_ro,
                _,
                _,
            ) = self.critic(robot_state, obstacle_state, action)

            # Apply active_mask to critic loss (only train on active robots)
            critic_loss_per_robot_1 = F.mse_loss(current_Q1, target_Q, reduction='none')
            critic_loss_per_robot_2 = F.mse_loss(current_Q2, target_Q, reduction='none')
            critic_loss = (
                (critic_loss_per_robot_1 * active_mask).sum() / active_mask.sum() +
                (critic_loss_per_robot_2 * active_mask).sum() / active_mask.sum()
            )

            # BCE losses for hard attention
            targets_rr = (
                dist_rr.flatten() < connection_proximity_threshold_rr
            ).float()
            bce_loss_rr = F.binary_cross_entropy_with_logits(
                hard_logits_rr.flatten(), targets_rr
            )

            targets_ro = (
                dist_ro.flatten() < connection_proximity_threshold_ro
            ).float()
            bce_loss_ro = F.binary_cross_entropy_with_logits(
                hard_logits_ro.flatten(), targets_ro
            )

            av_critic_bce_loss_rr.append(bce_loss_rr.item())
            av_critic_bce_loss_ro.append(bce_loss_ro.item())

            total_loss = (
                critic_loss
                - entropy_weight * mean_entropy
                + bce_weight * (bce_loss_rr + bce_loss_ro)
            )
            av_critic_entropy.append(mean_entropy.item())

            self.critic_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
            self.critic_optimizer.step()

            av_loss += total_loss.item()
            av_critic_loss += critic_loss.item()

            # Actor update
            if it % policy_freq == 0:
                (
                    action_pred,
                    hard_logits_rr,
                    hard_logits_ro,
                    dist_rr,
                    dist_ro,
                    mean_entropy,
                    _,
                    _,
                    _,
                ) = self.actor(robot_state, obstacle_state, detach_attn=False)

                targets_rr = (
                    dist_rr.flatten() < connection_proximity_threshold_rr
                ).float()
                bce_loss_rr = F.binary_cross_entropy_with_logits(
                    hard_logits_rr.flatten(), targets_rr
                )

                targets_ro = (
                    dist_ro.flatten() < connection_proximity_threshold_ro
                ).float()
                bce_loss_ro = F.binary_cross_entropy_with_logits(
                    hard_logits_ro.flatten(), targets_ro
                )

                av_actor_bce_loss_rr.append(bce_loss_rr.item())
                av_actor_bce_loss_ro.append(bce_loss_ro.item())

                actor_Q, _, _, _, _, _, _, _, _ = self.critic(
                    robot_state, obstacle_state, action_pred
                )
                # Apply active_mask to actor loss (only train on active robots)
                actor_loss_per_robot = -actor_Q
                actor_loss = (actor_loss_per_robot * active_mask).sum() / active_mask.sum()
                total_loss = (
                    actor_loss
                    - entropy_weight * mean_entropy
                    + bce_weight * (bce_loss_rr + bce_loss_ro)
                )
                av_actor_entropy.append(mean_entropy.item())

                self.actor_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_params, max_grad_norm)
                self.actor_optimizer.step()

                av_actor_loss += total_loss.item()

                # Soft update target networks
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

        # Logging
        self.iter_count += 1
        self.writer.add_scalar("train/loss_total", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/critic_loss", av_critic_loss / iterations, self.iter_count)
        self.writer.add_scalar(
            "train/av_critic_entropy",
            sum(av_critic_entropy) / len(av_critic_entropy),
            self.iter_count,
        )
        self.writer.add_scalar(
            "train/av_actor_entropy",
            sum(av_actor_entropy) / len(av_actor_entropy) if av_actor_entropy else 0,
            self.iter_count,
        )
        self.writer.add_scalar(
            "train/av_critic_bce_loss_rr",
            sum(av_critic_bce_loss_rr) / len(av_critic_bce_loss_rr),
            self.iter_count,
        )
        self.writer.add_scalar(
            "train/av_critic_bce_loss_ro",
            sum(av_critic_bce_loss_ro) / len(av_critic_bce_loss_ro),
            self.iter_count,
        )
        self.writer.add_scalar(
            "train/av_actor_bce_loss_rr",
            sum(av_actor_bce_loss_rr) / len(av_actor_bce_loss_rr) if av_actor_bce_loss_rr else 0,
            self.iter_count,
        )
        self.writer.add_scalar(
            "train/av_actor_bce_loss_ro",
            sum(av_actor_bce_loss_ro) / len(av_actor_bce_loss_ro) if av_actor_bce_loss_ro else 0,
            self.iter_count,
        )
        self.writer.add_scalar("train/avg_Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("train/max_Q", max_Q, self.iter_count)
        self.writer.add_scalar(
            "train/actor_loss",
            av_actor_loss / (iterations // policy_freq) if iterations >= policy_freq else 0,
            self.iter_count,
        )

        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)

    def save(self, filename, directory):
        """Save the current model parameters."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.actor_target.state_dict(), f"{directory}/{filename}_actor_target.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
        torch.save(self.critic_target.state_dict(), f"{directory}/{filename}_critic_target.pth")

    def load(self, filename, directory):
        """Load model parameters from files."""
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth", map_location=self.device), strict=False)
        self.actor_target.load_state_dict(torch.load(f"{directory}/{filename}_actor_target.pth", map_location=self.device), strict=False)
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth", map_location=self.device), strict=False)
        self.critic_target.load_state_dict(torch.load(f"{directory}/{filename}_critic_target.pth", map_location=self.device), strict=False)
        print(f"Loaded weights from: {directory}")

    def prepare_state(
        self, poses, distance, cos, sin, collision, action, goal_positions
    ):
        """
        Convert raw environment outputs into per-robot state vectors.

        Args:
            poses (list): Per-robot poses [[x, y, theta], ...].
            distance (list): Per-robot distances to goal.
            cos (list): Per-robot cos(heading error to goal).
            sin (list): Per-robot sin(heading error to goal).
            collision (list): Per-robot collision flags.
            action (list): Per-robot last actions [[lin_vel, ang_vel], ...].
            goal_positions (list): Per-robot goals [[gx, gy], ...].

        Returns:
            tuple:
                states (list): Per-robot state vectors (length == state_dim).
                terminal (list): Terminal flags (collision), same length as states.
        """
        states = []
        terminal = []

        for i in range(self.num_robots):
            pose = poses[i]
            goal_pos = goal_positions[i]
            act = action[i]

            px, py, theta = pose
            gx, gy = goal_pos

            heading_cos = np.cos(theta)
            heading_sin = np.sin(theta)
            lin_vel = act[0] * 2
            ang_vel = (act[1] + 1) / 2

            state = [
                px,
                py,
                heading_cos,
                heading_sin,
                distance[i] / 17,
                cos[i],
                sin[i],
                lin_vel,
                ang_vel,
                gx,
                gy,
            ]

            assert len(state) == self.state_dim, (
                f"State length mismatch: expected {self.state_dim}, got {len(state)}"
            )
            states.append(state)
            terminal.append(collision[i])

        return states, terminal
