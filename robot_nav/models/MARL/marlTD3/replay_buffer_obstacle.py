"""
Replay Buffer for MARL with Obstacle Graph Nodes.

This module provides a replay buffer that stores both robot and obstacle
observations separately, enabling proper batching for the obstacle-aware
graph attention network.
"""

import random
from collections import deque
import numpy as np


class ReplayBufferObstacle:
    """
    Experience replay buffer for obstacle-aware multi-agent RL.

    Stores tuples of (robot_state, obstacle_state, action, reward, done,
    next_robot_state, next_obstacle_state) up to a fixed capacity.

    Attributes:
        buffer_size (int): Maximum number of transitions to store.
        count (int): Current number of stored transitions.
        buffer (deque): Internal storage for transitions.
    """

    def __init__(self, buffer_size, random_seed=123):
        """
        Initialize the replay buffer.

        Args:
            buffer_size (int): Maximum number of transitions to store.
            random_seed (int): Seed for random number generation.
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, robot_state, obstacle_state, action, reward, done, 
            next_robot_state, next_obstacle_state, active_mask=None):
        """
        Add a transition to the buffer.

        Args:
            robot_state (np.ndarray): Robot states of shape (N_robots, robot_state_dim).
            obstacle_state (np.ndarray): Obstacle states of shape (N_obs, obstacle_state_dim).
            action (np.ndarray): Actions of shape (N_robots, action_dim).
            reward (list or np.ndarray): Per-robot rewards.
            done (list or np.ndarray): Per-robot done flags.
            next_robot_state (np.ndarray): Next robot states.
            next_obstacle_state (np.ndarray): Next obstacle states.
            active_mask (np.ndarray or None): Boolean mask for active robots (N_robots,).
                If None, all robots are assumed active.
        """
        if active_mask is None:
            active_mask = np.ones(len(robot_state), dtype=bool)
        
        experience = (
            robot_state,
            obstacle_state,
            action,
            reward,
            done,
            next_robot_state,
            next_obstacle_state,
            active_mask,
        )
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        """
        Get the number of elements currently in the buffer.

        Returns:
            int: Current buffer size.
        """
        return self.count

    def sample_batch(self, batch_size):
        """
        Sample a batch of experiences from the buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            tuple: (
                robot_states (np.ndarray): Shape (batch_size, N_robots, robot_state_dim).
                obstacle_states (np.ndarray): Shape (batch_size, N_obs, obstacle_state_dim).
                actions (np.ndarray): Shape (batch_size, N_robots, action_dim).
                rewards (np.ndarray): Shape (batch_size, N_robots).
                dones (np.ndarray): Shape (batch_size, N_robots).
                next_robot_states (np.ndarray): Shape (batch_size, N_robots, robot_state_dim).
                next_obstacle_states (np.ndarray): Shape (batch_size, N_obs, obstacle_state_dim).
                active_masks (np.ndarray): Shape (batch_size, N_robots), dtype bool or float.
            )
        """
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        robot_states = np.array([exp[0] for exp in batch])
        obstacle_states = np.array([exp[1] for exp in batch])
        actions = np.array([exp[2] for exp in batch])
        rewards = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        next_robot_states = np.array([exp[5] for exp in batch])
        next_obstacle_states = np.array([exp[6] for exp in batch])
        active_masks = np.array([exp[7] if len(exp) > 7 else np.ones(len(exp[0]), dtype=bool) for exp in batch])

        return (
            robot_states,
            obstacle_states,
            actions,
            rewards,
            dones,
            next_robot_states,
            next_obstacle_states,
            active_masks,
        )

    def return_buffer(self):
        """
        Return the entire buffer contents as separate arrays.

        Returns:
            tuple: Full arrays of all stored data.
        """
        robot_states = np.array([exp[0] for exp in self.buffer])
        obstacle_states = np.array([exp[1] for exp in self.buffer])
        actions = np.array([exp[2] for exp in self.buffer])
        rewards = np.array([exp[3] for exp in self.buffer])  # Shape: (N, N_robots)
        dones = np.array([exp[4] for exp in self.buffer])    # Shape: (N, N_robots)
        next_robot_states = np.array([exp[5] for exp in self.buffer])
        next_obstacle_states = np.array([exp[6] for exp in self.buffer])
        active_masks = np.array([exp[7] if len(exp) > 7 else np.ones(len(exp[0]), dtype=bool) for exp in self.buffer])

        return (
            robot_states,
            obstacle_states,
            actions,
            rewards,
            dones,
            next_robot_states,
            next_obstacle_states,
            active_masks,
        )

    def clear(self):
        """
        Clear all contents of the buffer.
        """
        self.buffer.clear()
        self.count = 0
