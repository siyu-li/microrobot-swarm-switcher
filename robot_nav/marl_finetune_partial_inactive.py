"""
Fine-tuning script: Annealed Partial-Inactive Training for 14 Robots.

Fine-tunes a pretrained all-robots-active TD3 policy so that the policy
learns to handle the group-switching setting where some robots are frozen.

Key idea:
    At each selection interval, ``num_controlled`` robots are randomly chosen
    to be "controlled" — they always receive their normal policy actions.
    The remaining ``num_robots - num_controlled`` robots are "uncontrolled":
    each one independently has probability ``p_inactive`` of being frozen
    ([0, 0]) and ``1 - p_inactive`` of receiving its normal policy action.

    ``p_inactive`` is annealed from 0% → 70% over ``anneal_epochs`` epochs
    so the policy gradually adapts to having some neighbors frozen.

    The policy is fine-tuned with a reduced learning rate (0.3× of original)
    to prevent catastrophic forgetting of the pretrained navigation skills.

Fine-tunes from:
    checkpoint/Feb.27_obstacle_14robot/TD3-MARL-obstacle-14robots

Saves to:
    checkpoint/Mar.04_obstacle_14robots_partial_inactive/

Usage:
    python -m robot_nav.marl_finetune_partial_inactive
"""

from pathlib import Path
import random

import torch
import numpy as np

from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import TD3Obstacle
from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE
from robot_nav.models.MARL.marlTD3.replay_buffer_obstacle import ReplayBufferObstacle

# Suppress IRSim warnings
from loguru import logger
logger.disable("irsim")


def outside_of_bounds(poses, sim):
    """Check if any robot is outside world boundaries."""
    for pose in poses:
        if pose[0] < sim.x_range[0] or pose[0] > sim.x_range[1]:
            return True
        if pose[1] < sim.y_range[0] or pose[1] > sim.y_range[1]:
            return True
    return False


def select_controlled_robots(num_robots: int, num_controlled: int) -> set:
    """
    Randomly select which robots are "controlled" this interval.

    Args:
        num_robots: Total number of robots.
        num_controlled: How many robots are guaranteed to receive their action.

    Returns:
        Set of robot indices that are controlled.
    """
    return set(random.sample(range(num_robots), num_controlled))


def apply_partial_inactive_actions(
    raw_actions: np.ndarray,
    controlled_set: set,
    num_robots: int,
    p_inactive: float,
) -> tuple:
    """
    Apply stochastic partial-inactive logic to actions.

    Controlled robots: always receive their normal scaled action.
    Uncontrolled robots (not in controlled_set):
        - With probability p_inactive: get [0, 0] (frozen)
        - With probability (1 - p_inactive): get their normal scaled action

    Args:
        raw_actions: Raw policy output, shape (num_robots, 2), values in [-1, 1].
        controlled_set: Set of robot indices that are controlled.
        num_robots: Total number of robots.
        p_inactive: Probability of freezing each uncontrolled robot.

    Returns:
        Tuple of:
        - env_actions: List of [v, w] actions for all robots (scaled for env).
        - active_mask: np.ndarray of shape (num_robots,), dtype bool.
            True for robots whose policy action was actually executed,
            False for robots that were frozen ([0, 0]).
    """
    env_actions = []
    active_mask = np.ones(num_robots, dtype=bool)

    for i in range(num_robots):
        # Scale linear velocity: [-1, 1] → [0, 0.5]
        v_scaled = (raw_actions[i][0] + 1) / 4
        w = raw_actions[i][1]

        if i in controlled_set:
            # Controlled robot: always gets its action
            env_actions.append([v_scaled, w])
            # active_mask[i] stays True
        else:
            # Uncontrolled robot: stochastic inactive
            if random.random() < p_inactive:
                env_actions.append([0.0, 0.0])
                active_mask[i] = False  # Frozen — exclude from training
            else:
                env_actions.append([v_scaled, w])
                # active_mask[i] stays True

    return env_actions, active_mask


def main():
    """Main fine-tuning function."""
    # =====================================================================
    # Configuration
    # =====================================================================

    # Source model (pretrained all-robots-active)
    load_model_name = "TD3-MARL-obstacle-14robots"
    load_directory = Path("robot_nav/models/MARL/marlTD3/checkpoint/Feb.27_obstacle_14robot")

    # Save directory for fine-tuned model
    save_model_name = "TD3-MARL-obstacle-14robots-partial-inactive"
    save_directory = Path("robot_nav/models/MARL/marlTD3/checkpoint/Mar.04_obstacle_14robots_partial_inactive")

    # Architecture
    action_dim = 2
    max_action = 1
    state_dim = 11
    obstacle_state_dim = 4
    num_robots = 14
    num_obstacles = 7

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fine-tuning hyperparameters (reduced LR for fine-tuning)
    lr_actor = 3e-5      # 0.3× of original 1e-4
    lr_critic = 1e-4     # 0.33× of original 3e-4

    # Training schedule
    max_epochs = 500           # 500 epochs of fine-tuning
    checkpoint_every = 30      # Save every 30 epochs
    train_every_n = 10
    training_iterations = 80
    batch_size = 32
    max_steps = 300
    buffer_size = 50000

    # Partial-inactive configuration
    num_controlled = 3         # Half the robots are always controlled
    selection_interval = 10    # Re-randomize controlled set every N steps

    # Annealing schedule for p_inactive (applied to uncontrolled robots)
    # Linearly anneal from 0% → 70% over anneal_epochs, then hold at 70%
    p_inactive_start = 0.0
    p_inactive_end = 0.7
    anneal_epochs = 200        # Reach 70% at epoch 200, then hold

    # Environment hyperparameters
    per_robot_goal_reset = True
    obstacle_proximity_threshold = 1.5
    goal_dwell_min = 0
    goal_respawn_prob = 1.0
    station_keeping_reward = 5.0

    # =====================================================================
    # Environment
    # =====================================================================
    sim = MARL_SIM_OBSTACLE(
        world_file="robot_nav/worlds/multi_robot_world_obstacle_14robots.yaml",
        disable_plotting=True,
        reward_phase=8,
        per_robot_goal_reset=per_robot_goal_reset,
        obstacle_proximity_threshold=obstacle_proximity_threshold,
        num_inactive_robots=0,
        goal_dwell_min=goal_dwell_min,
        goal_respawn_prob=goal_respawn_prob,
        station_keeping_reward=station_keeping_reward,
    )

    print(f"\n{'='*60}")
    print(f"FINE-TUNING — Annealed Partial-Inactive (14 Robots)")
    print(f"{'='*60}")
    print(f"  Source model: {load_directory / load_model_name}")
    print(f"  Save to:      {save_directory / save_model_name}")
    print(f"  Robots: {sim.num_robots}, Obstacles: {sim.num_obstacles}")
    print(f"  Controlled robots: {num_controlled} / {num_robots}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Checkpoint every: {checkpoint_every} epochs")
    print(f"  LR actor: {lr_actor}, LR critic: {lr_critic}")
    print(f"  p_inactive (uncontrolled): {p_inactive_start} → {p_inactive_end} over {anneal_epochs} epochs")
    print(f"  Selection interval: {selection_interval} steps")
    print(f"{'='*60}\n")

    # =====================================================================
    # Model — load pretrained weights
    # =====================================================================
    model = TD3Obstacle(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        num_robots=sim.num_robots,
        num_obstacles=sim.num_obstacles,
        obstacle_state_dim=obstacle_state_dim,
        device=device,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        save_every=0,  # We handle saving manually
        load_model=True,
        load_model_name=load_model_name,
        load_directory=load_directory,
        model_name=save_model_name,
        save_directory=save_directory,
    )

    # =====================================================================
    # Replay buffer
    # =====================================================================
    replay_buffer = ReplayBufferObstacle(buffer_size=buffer_size)

    # =====================================================================
    # Initial environment step
    # =====================================================================
    (
        poses, distance, cos, sin, collision, goal, a, reward,
        positions, goal_positions, obstacle_states
    ) = sim.step([[0, 0] for _ in range(sim.num_robots)], None)

    running_goals = 0
    running_collisions = 0
    running_timesteps = 0
    epoch = 1
    episode = 0
    steps = 0

    print(f"Starting fine-tuning...\n")

    # =====================================================================
    # Main fine-tuning loop
    # =====================================================================
    controlled_set = None

    while epoch <= max_epochs:
        # ------------------------------------------------------------------
        # Compute current p_inactive (annealed)
        # ------------------------------------------------------------------
        if epoch <= anneal_epochs:
            p_inactive = p_inactive_start + (p_inactive_end - p_inactive_start) * (epoch / anneal_epochs)
        else:
            p_inactive = p_inactive_end

        # ------------------------------------------------------------------
        # Re-randomize which robots are controlled every N steps
        # ------------------------------------------------------------------
        if steps % selection_interval == 0 or controlled_set is None:
            controlled_set = select_controlled_robots(num_robots, num_controlled)

        # ------------------------------------------------------------------
        # Prepare robot state and get raw policy action
        # ------------------------------------------------------------------
        robot_state, terminal = model.prepare_state(
            poses, distance, cos, sin, collision, a, goal_positions
        )

        # Get raw action from policy (with exploration noise)
        action, combined_weights = model.get_action(
            np.array(robot_state), obstacle_states, add_noise=True
        )
        # action shape: (num_robots, 2), values in [-1, 1]

        # ------------------------------------------------------------------
        # Apply partial-inactive logic
        # ------------------------------------------------------------------
        a_in, active_mask = apply_partial_inactive_actions(
            raw_actions=action,
            controlled_set=controlled_set,
            num_robots=sim.num_robots,
            p_inactive=p_inactive,
        )

        # ------------------------------------------------------------------
        # Step environment
        # ------------------------------------------------------------------
        (
            poses, distance, cos, sin, collision, goal, a, reward,
            positions, goal_positions, next_obstacle_states
        ) = sim.step(a_in, None, combined_weights)

        running_goals += sum(goal)
        running_collisions += sum(collision)
        running_timesteps += 1

        # ------------------------------------------------------------------
        # Prepare next state and store transition
        # ------------------------------------------------------------------
        next_robot_state, terminal = model.prepare_state(
            poses, distance, cos, sin, collision, a, goal_positions
        )

        # Store in replay buffer
        # Store the ACTUALLY EXECUTED action and mask out frozen robots.
        # Frozen robots got [0,0] in the env, so their (s, a_raw, r, s') tuple
        # is inconsistent — the reward/next-state came from [0,0], not a_raw.
        # active_mask ensures frozen robots are excluded from critic/actor loss.
        executed_action = np.array(action, copy=True)
        for i in range(sim.num_robots):
            if not active_mask[i]:
                executed_action[i] = [0.0, 0.0]

        replay_buffer.add(
            robot_state,
            obstacle_states,
            executed_action,
            reward,
            terminal,
            next_robot_state,
            next_obstacle_states,
            active_mask=active_mask,
        )

        obstacle_states = next_obstacle_states
        steps += 1
        episode += 1

        # ------------------------------------------------------------------
        # Episode termination
        # ------------------------------------------------------------------
        if (
            any(collision)
            or steps >= max_steps
            or outside_of_bounds(poses, sim)
        ):
            (
                poses, distance, cos, sin, collision, goal, a, reward,
                positions, goal_positions, obstacle_states
            ) = sim.reset(random_obstacles=True)

            steps = 0
            controlled_set = None  # Re-select on next step
            epoch += 1

            # ----------------------------------------------------------
            # Training
            # ----------------------------------------------------------
            if episode >= train_every_n and replay_buffer.size() >= batch_size:
                avg_goal_rate = running_goals / max(running_timesteps, 1)
                avg_collision_rate = running_collisions / max(running_timesteps, 1)

                model.writer.add_scalar("run/avg_goal", avg_goal_rate, model.iter_count)
                model.writer.add_scalar("run/avg_collision", avg_collision_rate, model.iter_count)
                model.writer.add_scalar("run/buffer_size", replay_buffer.size(), model.iter_count)
                model.writer.add_scalar("run/p_inactive", p_inactive, model.iter_count)

                # Log dwell statistics
                num_dwelling = sum(1 for c in sim.dwell_counters if c >= 0)
                model.writer.add_scalar("run/num_dwelling", num_dwelling, model.iter_count)

                running_goals = 0
                running_collisions = 0
                running_timesteps = 0

                model.train(
                    replay_buffer,
                    training_iterations,
                    batch_size,
                    connection_proximity_threshold_rr=5.0,
                    connection_proximity_threshold_ro=2.5,
                )
                episode = 0

                # ----------------------------------------------------------
                # Checkpoint saving (every 50 epochs)
                # ----------------------------------------------------------
                if epoch % checkpoint_every == 0:
                    checkpoint_name = f"{model.model_name}_epoch{epoch}"
                    model.save(filename=checkpoint_name, directory=model.save_directory)
                    print(f"✅ Checkpoint saved: {checkpoint_name} (p_inactive={p_inactive:.2f})")

                # Console logging
                if epoch % 10 == 0:
                    print(
                        f"Epoch {epoch}/{max_epochs} | "
                        f"p_inactive={p_inactive:.2f} | "
                        f"controlled={num_controlled}/{num_robots} | "
                        f"Buffer: {replay_buffer.size()} | "
                        f"Goals: {avg_goal_rate*100:.1f}% | "
                        f"Collisions: {avg_collision_rate*100:.1f}%"
                    )

    # =====================================================================
    # Save final model
    # =====================================================================
    print(f"\n{'='*60}")
    print("Fine-tuning complete!")
    print(f"{'='*60}")
    model.save(filename=model.model_name, directory=model.save_directory)
    print(f"Final model saved to: {model.save_directory / model.model_name}")
    print(f"Checkpoints saved every {checkpoint_every} epochs at:")
    for e in range(checkpoint_every, max_epochs + 1, checkpoint_every):
        print(f"  {model.model_name}_epoch{e}")


if __name__ == "__main__":
    main()
