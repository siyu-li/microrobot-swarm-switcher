"""
Training script for MARL TD3 with 14 Robots and Obstacle Graph Nodes (FROM SCRATCH).

This script trains a 14-robot multi-agent TD3 policy

Key features:
- 14 robots with 7 obstacles in 25x25 world
- Training from randomly initialized weights
- Adjusted hyperparameters for larger scale (14 vs 6 robots)
- Longer episodes and larger replay buffer
"""

from pathlib import Path
import pickle

import torch
import numpy as np
import logging

from robot_nav.models.MARL.marlTD3.marlTD3_obstacle import TD3Obstacle
from robot_nav.SIM_ENV.marl_obstacle_sim import MARL_SIM_OBSTACLE
from robot_nav.models.MARL.marlTD3.replay_buffer_obstacle import ReplayBufferObstacle

# Suppress IRSim warnings - irsim uses loguru, not standard logging
from loguru import logger
logger.disable("irsim")


def outside_of_bounds(poses, sim):
    """
    Check if any robot is outside the defined world boundaries.

    Args:
        poses (list): List of [x, y, theta] poses for each robot.
        sim: Simulation environment with x_range and y_range.

    Returns:
        bool: True if any robot is outside world boundaries.
    """
    for pose in poses:
        if pose[0] < sim.x_range[0] or pose[0] > sim.x_range[1]:
            return True
        if pose[1] < sim.y_range[0] or pose[1] > sim.y_range[1]:
            return True
    return False


def main(args=None):
    """Main training function for 14-robot obstacle-aware MARL."""

    # ---- Hyperparameters ----
    action_dim = 2
    max_action = 1
    state_dim = 11  # Robot state dimension
    obstacle_state_dim = 4  # Obstacle state: [x, y, cos_h, sin_h]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Training hyperparameters (adjusted for 14 robots)
    max_epochs = 5000  # More epochs needed for 14 robots
    epoch = 1
    episode = 0
    train_every_n = 10  # 15
    training_iterations = 80  # 100
    batch_size = 32  # 24
    max_steps = 300  # 400
    steps = 0
    save_every = 5
    buffer_size = 100000  # 100000

    # Environment hyperparameters
    per_robot_goal_reset = True
    obstacle_proximity_threshold = 1.5  # For reward penalty
    num_inactive_robots = 0  # Number of robots to be inactive each episode (treated as obstacles)
    goal_dwell_min = 0  # Robot stays at goal for at least 30 steps
    goal_respawn_prob = 1.0  # Respawn immediately after dwell period ends
    station_keeping_reward = 5.0  # Small reward for holding position at goal

    # ---- Instantiate environment ----
    sim = MARL_SIM_OBSTACLE(
        world_file="robot_nav/worlds/multi_robot_world_obstacle_14robots.yaml",
        disable_plotting=True,
        reward_phase=8,
        per_robot_goal_reset=per_robot_goal_reset,
        obstacle_proximity_threshold=obstacle_proximity_threshold,
        num_inactive_robots=num_inactive_robots,
        goal_dwell_min=goal_dwell_min,
        goal_respawn_prob=goal_respawn_prob,
        station_keeping_reward=station_keeping_reward,
    )

    print(f"\n{'='*60}")
    print(f"TRAINING - 14 ROBOTS")
    print(f"{'='*60}")
    print(f"Environment initialized:")
    print(f"  - Number of robots: {sim.num_robots}")
    print(f"  - Number of obstacles: {sim.num_obstacles}")
    print(f"  - World bounds: x={sim.x_range}, y={sim.y_range}")
    print(f"  - Max epochs: {max_epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Buffer size: {buffer_size}")
    print(f"{'='*60}\n")

    # ---- Instantiate model----
    model = TD3Obstacle(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        num_robots=sim.num_robots,
        num_obstacles=sim.num_obstacles,
        obstacle_state_dim=obstacle_state_dim,
        device=device,
        save_every=save_every,
        load_model=False,
        # load_model_name="TD3-MARL-obstacle-6robots_epoch2400",
        # load_directory=Path("robot_nav/models/MARL/marlTD3/checkpoint/obstacle_6robots_v2"),
        load_model_name="TD3-MARL-obstacle-14robots",
        load_directory=Path("robot_nav/models/MARL/marlTD3/checkpoint/Feb.27_obstacle_14robot"),
        model_name="TD3-MARL-obstacle-14robots",
        save_directory=Path("robot_nav/models/MARL/marlTD3/checkpoint/Mar.15_obstacle_14robot_reward8"),
    )


    # ---- Setup replay buffer ----
    replay_buffer = ReplayBufferObstacle(buffer_size=buffer_size)

    # ---- Take initial step in environment ----
    (
        poses, distance, cos, sin, collision, goal, a, reward,
        positions, goal_positions, obstacle_states
    ) = sim.step([[0, 0] for _ in range(sim.num_robots)], None)

    running_goals = 0
    running_collisions = 0
    running_timesteps = 0
    
    # Checkpoint saving parameters
    checkpoint_every = 200  # Save checkpoint every N epochs

    print(f"Starting training...")
    print(f"Initial obstacle states shape: {obstacle_states.shape}\n")

    # ---- Main training loop ----
    while epoch < max_epochs:
        # Prepare robot state
        robot_state, terminal = model.prepare_state(
            poses, distance, cos, sin, collision, a, goal_positions
        )

        # Get action from model
        action, combined_weights = model.get_action(
            np.array(robot_state), obstacle_states, add_noise=True
        )

        # Scale action for environment
        a_in = [[(act[0] + 1) / 4, act[1]] for act in action]

        # Step environment
        (
            poses, distance, cos, sin, collision, goal, a, reward,
            positions, goal_positions, next_obstacle_states
        ) = sim.step(a_in, None, combined_weights)

        running_goals += sum(goal)
        running_collisions += sum(collision)
        running_timesteps += 1

        # Prepare next state
        next_robot_state, terminal = model.prepare_state(
            poses, distance, cos, sin, collision, a, goal_positions
        )

        # Add to replay buffer
        replay_buffer.add(
            robot_state,
            obstacle_states,
            action,
            reward,
            terminal,
            next_robot_state,
            next_obstacle_states,
            active_mask=sim.active_mask,
        )

        # Update obstacle states for next iteration
        obstacle_states = next_obstacle_states

        steps += 1
        episode += 1

        # Check termination conditions
        # Note: `all(goal)` is removed — with dwell-then-respawn, robots stay at
        # their goals and handle their own respawn lifecycle individually.
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
            epoch += 1

            # Training
            if episode >= train_every_n and replay_buffer.size() >= batch_size:
                # Log run metrics
                avg_goal_rate = running_goals / max(running_timesteps, 1)
                avg_collision_rate = running_collisions / max(running_timesteps, 1)
                model.writer.add_scalar(
                    "run/avg_goal", avg_goal_rate, model.iter_count
                )
                model.writer.add_scalar(
                    "run/avg_collision", avg_collision_rate, model.iter_count
                )
                model.writer.add_scalar(
                    "run/buffer_size", replay_buffer.size(), model.iter_count
                )
                # Log dwell statistics
                num_dwelling = sum(1 for c in sim.dwell_counters if c >= 0)
                model.writer.add_scalar(
                    "run/num_dwelling", num_dwelling, model.iter_count
                )
                running_goals = 0
                running_collisions = 0
                running_timesteps = 0
                
                model.train(
                    replay_buffer,
                    training_iterations,
                    batch_size,
                    connection_proximity_threshold_rr=5.0,  # Slightly larger for 14 robots
                    connection_proximity_threshold_ro=2.5,
                )
                episode = 0
                
                # Save checkpoint with epoch number
                if epoch % checkpoint_every == 0:
                    checkpoint_name = f"{model.model_name}_epoch{epoch}"
                    model.save(filename=checkpoint_name, directory=model.save_directory)
                    print(f"✅ Checkpoint saved: {checkpoint_name}")

                # # Save replay buffer every 1000 epochs
                # if epoch % 1000 == 0:
                #     buffer_path = model.save_directory / f"replay_buffer_epoch{epoch}.pkl"
                #     with open(buffer_path, "wb") as f:
                #         pickle.dump(replay_buffer, f)
                #     print(f"💾 Replay buffer saved: {buffer_path}")

                # Console logging
                if epoch % 10 == 0:
                    print(
                        f"Epoch {epoch}/{max_epochs} | "
                        f"Buffer: {replay_buffer.size()} | "
                        f"Goals: {avg_goal_rate*100:.1f}% | "
                        f"Collisions: {avg_collision_rate*100:.1f}%"
                    )

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    model.save(filename=model.model_name, directory=model.save_directory)
    print(f"Final model saved to: {model.save_directory}")


if __name__ == "__main__":
    main()
