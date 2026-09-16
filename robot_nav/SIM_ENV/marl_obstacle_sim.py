"""
MARL Simulation Environment with Obstacle Graph Nodes.

This module extends MARL_SIM to include obstacle information for the
graph attention network. Uses Shapely geometry.distance() for efficient
clearance computation.

Key features:
- Extracts obstacle positions and headings from IR-SIM
- Computes robot-obstacle clearance using Shapely geometry
- Provides obstacle state observations for the GAT
"""

import warnings
import irsim
import numpy as np
import random
import torch
import logging
from scipy.spatial.distance import cdist
from shapely.geometry import Point
from typing import List, Tuple, Optional

from robot_nav.SIM_ENV.corridor_layout import (
    CorridorLayout,
    sample_obstacles,
    sample_starts,
)
from robot_nav.SIM_ENV.sim_env import SIM_ENV

# Suppress matplotlib color redundancy warning from irsim
warnings.filterwarnings("ignore", message="color is redundantly defined")


class MARL_SIM_OBSTACLE(SIM_ENV):
    """
    Simulation environment for multi-agent robot navigation with obstacle graph nodes.

    Extends SIM_ENV to provide obstacle observations for the graph attention network.
    Obstacles are represented as static nodes with position and heading information.

    Attributes:
        env (object): IRSim simulation environment instance.
        robot_goal (np.ndarray): Current goal position(s) for the robots.
        num_robots (int): Number of robots in the environment.
        num_obstacles (int): Number of obstacles in the environment.
        x_range (tuple): World x-range.
        y_range (tuple): World y-range.
    """

    def __init__(
        self,
        world_file: str = "multi_robot_world_lidar.yaml",
        disable_plotting: bool = False,
        reward_phase: int = 1,
        per_robot_goal_reset: bool = True,
        obstacle_proximity_threshold: float = 1.5,
        num_inactive_robots: int = 0,
        goal_dwell_min: int = 0,
        goal_respawn_prob: float = 1.0,
        station_keeping_reward: float = 1.0,
        layout: Optional[CorridorLayout] = None,
    ):
        """
        Initialize the MARL_SIM_OBSTACLE environment.

        Args:
            world_file (str): Path to an IRSim YAML world configuration.
            disable_plotting (bool): If True, disables all IRSim plotting.
            reward_phase (int): Selects the reward function variant (1, 2, or 3).
            per_robot_goal_reset (bool): If True, reset individual robot goals when reached.
            obstacle_proximity_threshold (float): Distance threshold for obstacle penalty in reward.
            num_inactive_robots (int): Number of robots to randomly select as inactive each episode.
                If 0 (default), all robots are active (original behavior).
            goal_dwell_min (int): Minimum number of steps a robot must stay at its goal
                before it becomes eligible for respawn. 0 = immediate respawn (legacy).
            goal_respawn_prob (float): Per-step probability of respawning after the
                dwell period expires. 1.0 = immediate respawn after dwell (legacy).
            station_keeping_reward (float): Small positive reward given each step a
                robot successfully holds at its goal without collision.
            layout (CorridorLayout or None): If given, ``reset`` draws a banded
                "corridor" episode — starts in one x-band, goals in the opposite
                one, obstacles concentrated in a band across the middle — instead
                of scattering all three over the whole arena.  ``None`` (default)
                keeps the scattered layout, draw for draw.
        """
        display = False if disable_plotting else True
        self.env = irsim.make(
            world_file, disable_all_plot=disable_plotting, display=display,
        )
        robot_info = self.env.get_robot_info(0)
        self.robot_goal = robot_info.goal
        self.num_robots = len(self.env.robot_list)
        self.num_obstacles = len(self.env.obstacle_list)
        self.x_range = self.env._world.x_range
        self.y_range = self.env._world.y_range
        self.reward_phase = reward_phase
        self.per_robot_goal_reset = per_robot_goal_reset
        self.obstacle_proximity_threshold = obstacle_proximity_threshold
        
        # Track previous distances for progress-based reward
        self.prev_distances = [None] * self.num_robots
        
        # Inactive robots support
        self.num_inactive_robots = num_inactive_robots
        self.inactive_ids = []  # List of inactive robot indices
        self.active_mask = np.ones(self.num_robots, dtype=bool)  # True for active robots

        # Goal-dwell parameters: robots stay at goal for a dwell period before respawn
        self.goal_dwell_min = goal_dwell_min
        self.goal_respawn_prob = goal_respawn_prob
        self.station_keeping_reward = station_keeping_reward
        # Per-robot dwell counter: -1 = not dwelling, >=0 = steps spent dwelling
        self.dwell_counters = [-1] * self.num_robots

        # Banded episode layout (None = the scattered default).  Per-robot
        # traverse directions are drawn by `reset`.
        self.layout = layout
        self._directions = None
        if layout is not None:
            layout.validate()

    @staticmethod
    def _bounding_radius(obj) -> float:
        """
        Radius of the disc bounding ``obj``'s current geometry.

        Exact for the circular obstacles/robots of the 14-robot worlds and
        conservative for anything else, which is the safe direction: the banded
        sampler then over-spaces rather than under-spaces.
        """
        minx, miny, maxx, maxy = obj.geometry.bounds
        return 0.5 * max(maxx - minx, maxy - miny)

    def _obstacle_discs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Obstacle centres ``(n, 2)`` and bounding radii ``(n,)``."""
        if not self.env.obstacle_list:
            return np.zeros((0, 2)), np.zeros(0)
        xy = np.array(
            [np.asarray(o.state).flatten()[:2] for o in self.env.obstacle_list],
            dtype=np.float64,
        )
        r = np.array(
            [self._bounding_radius(o) for o in self.env.obstacle_list],
            dtype=np.float64,
        )
        return xy, r

    def get_obstacle_states(self) -> np.ndarray:
        """
        Get obstacle state observations for graph nodes.

        Returns:
            np.ndarray: Obstacle states of shape (num_obstacles, 4).
                Each row: [x, y, cos(heading), sin(heading)]
                For static obstacles, heading is derived from their orientation.
        """
        if self.num_obstacles == 0:
            return np.zeros((0, 4))

        obs_states = np.array([obs.state.flatten() for obs in self.env.obstacle_list])
        # obs_states shape: (num_obstacles, 3) -> [x, y, theta]
        thetas = obs_states[:, 2] if obs_states.shape[1] > 2 else np.zeros(self.num_obstacles)
        return np.column_stack([
            obs_states[:, 0],       # x
            obs_states[:, 1],       # y
            np.cos(thetas),         # cos(heading)
            np.sin(thetas),         # sin(heading)
        ])

    def get_robot_obstacle_clearances(self) -> np.ndarray:
        """
        Compute clearance (distance to boundary) from each robot to each obstacle.

        Uses vectorized cdist for circle obstacles (center-to-center minus radius).
        Falls back to Shapely for non-circle obstacles.

        Returns:
            np.ndarray: Clearances of shape (num_robots, num_obstacles).
                clearances[i][j] = distance from robot i center to obstacle j boundary.
        """
        if self.num_obstacles == 0:
            return np.empty((self.num_robots, 0))

        robot_positions = np.array([r.position.flatten() for r in self.env.robot_list])
        obs_positions = np.array([o.position.flatten() for o in self.env.obstacle_list])
        obs_radii = np.array([o.radius for o in self.env.obstacle_list])

        # Vectorized: center-to-center distance minus radius, clamped to 0
        center_dists = cdist(robot_positions, obs_positions)  # (N_robots, N_obs)
        clearances = np.maximum(center_dists - obs_radii[np.newaxis, :], 0.0)

        return clearances

    def get_min_obstacle_clearance(self, robot_idx: int) -> float:
        """
        Get minimum clearance from a robot to any obstacle.

        Args:
            robot_idx (int): Index of the robot.

        Returns:
            float: Minimum clearance to any obstacle (or inf if no obstacles).
        """
        if self.num_obstacles == 0:
            return float('inf')

        robot_pos = self.env.robot_list[robot_idx].position.flatten()
        obs_positions = np.array([o.position.flatten() for o in self.env.obstacle_list])
        obs_radii = np.array([o.radius for o in self.env.obstacle_list])
        center_dists = np.linalg.norm(obs_positions - robot_pos[np.newaxis, :], axis=1)
        clearances = np.maximum(center_dists - obs_radii, 0.0)
        return float(clearances.min())

    def proximity_penalties(
        self,
        cl_threshold: float = 1.25,
        cl_weight: float = 1.0,
        obs_threshold: float = 1.5,
        obs_weight: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-robot robot–robot and obstacle proximity penalties for the *current*
        world state.

        Reuses the same quadratic-barrier form and default thresholds/weights as
        the phase-6 reward (``_compute_rewards_vectorized``), so the CAPSwitcher
        decision-level clearance shaping is consistent with the GAT training
        signal.  Recomputed from the live simulator state, so it reflects the
        state at the moment it is called (the switcher evaluates it once at
        decision end).

        Args:
            cl_threshold:  Robot–robot distance below which a penalty applies (m).
            cl_weight:     Weight on the robot–robot penalty.
            obs_threshold: Obstacle clearance below which a penalty applies (m).
            obs_weight:    Weight on the obstacle penalty.

        Returns:
            (cl_pen, obs_pen): two (N,) non-negative arrays.
        """
        robot_positions = np.array(
            [r.position.flatten() for r in self.env.robot_list]
        )

        # --- Robot–robot proximity (sum of squared barrier over neighbours) ---
        pairwise = cdist(robot_positions, robot_positions)
        np.fill_diagonal(pairwise, np.inf)  # exclude self
        close = pairwise < cl_threshold
        cl_pen = cl_weight * np.sum(
            np.where(close, (cl_threshold - pairwise) ** 2, 0.0), axis=1
        )

        # --- Obstacle proximity (squared barrier on nearest obstacle) ---
        clearances = self.get_robot_obstacle_clearances()  # (N, M)
        if clearances.size > 0:
            min_clearances = clearances.min(axis=1)  # (N,)
        else:
            min_clearances = np.full(self.num_robots, np.inf)
        obs_close = min_clearances < obs_threshold
        obs_pen = obs_weight * np.where(
            obs_close, (obs_threshold - min_clearances) ** 2, 0.0
        )

        return cl_pen, obs_pen

    def step(self, action, connection=None, combined_weights=None):
        """
        Perform a simulation step for all robots.

        Args:
            action (list): Actions for each robot [[lin_vel, ang_vel], ...].
            connection (Tensor or None): Connection logits (unused, for compatibility).
            combined_weights (Tensor or None): Visualization weights.

        Returns:
            tuple: (
                poses, distances, coss, sins, collisions, goals,
                action, rewards, positions, goal_positions, obstacle_states
            )
            obstacle_states (np.ndarray): Shape (num_obstacles, 4).
        """
        # Force inactive robots to have zero action
        action_copy = [list(act) for act in action]  # Deep copy
        for i in self.inactive_ids:
            action_copy[i] = [0.0, 0.0]

        # Use _objects_step (batch) instead of per-robot _object_step.
        # env.step(action_id=list, action=list) calls _object_step per robot,
        # which redundantly re-steps ALL other objects each time → O(N * total).
        # _objects_step steps each object exactly once → O(total).  ~7x faster.
        self.env._objects_step(action_copy)
        self.env.build_tree()
        self.env._objects_check_status()
        self.env._world.step()
        self.env.step_status()
        self.env.render()

        # === Vectorized state extraction ===
        all_states = np.array([r.state.flatten() for r in self.env.robot_list])   # (N, 3)
        all_goals = np.array([r.goal.flatten() for r in self.env.robot_list])     # (N, 3)
        all_collisions = [r.collision for r in self.env.robot_list]               # list of bool
        all_arrives = [r.arrive for r in self.env.robot_list]                     # list of bool

        robot_positions = all_states[:, :2]                                       # (N, 2)
        robot_thetas = all_states[:, 2]                                           # (N,)

        # === Vectorized pairwise robot-robot distances (cdist) ===
        pairwise_dists = cdist(robot_positions, robot_positions)                  # (N, N)
        np.fill_diagonal(pairwise_dists, np.inf)  # exclude self

        # === Vectorized robot-obstacle clearances ===
        clearances = self.get_robot_obstacle_clearances()                         # (N, M) ndarray
        if clearances.size > 0:
            min_clearances = clearances.min(axis=1)                               # (N,)
        else:
            min_clearances = np.full(self.num_robots, np.inf)

        # === Vectorized goal distances and cos/sin ===
        goal_vecs = all_goals[:, :2] - robot_positions                            # (N, 2)
        goal_dists = np.linalg.norm(goal_vecs, axis=1)                           # (N,)
        goal_dists_safe = np.where(goal_dists > 1e-10, goal_dists, 1e-10)

        heading_vecs = np.column_stack([np.cos(robot_thetas),
                                        np.sin(robot_thetas)])                    # (N, 2)
        # heading_vecs are already unit vectors, but normalize for safety
        h_norm = heading_vecs / np.linalg.norm(heading_vecs, axis=1, keepdims=True)
        g_norm = goal_vecs / goal_dists_safe[:, np.newaxis]

        all_coss = np.sum(h_norm * g_norm, axis=1)                               # (N,)
        all_sins = h_norm[:, 0] * g_norm[:, 1] - h_norm[:, 1] * g_norm[:, 0]    # (N,)

        # === Vectorized reward computation ===
        actions_arr = np.array(action)  # (N, 2)
        prev_dists_arr = np.array([d if d is not None else np.nan for d in self.prev_distances])

        rewards = self._compute_rewards_vectorized(
            all_arrives, all_collisions, actions_arr,
            pairwise_dists, goal_dists, min_clearances,
            self.reward_phase, prev_dists_arr,
        )

        # === Update prev_distances ===
        self.prev_distances = goal_dists.tolist()

        # === Build output lists (cheap, just slicing pre-computed arrays) ===
        poses = all_states.tolist()                                               # [[x, y, theta], ...]
        positions = robot_positions.tolist()                                       # [[x, y], ...]
        goal_positions_out = all_goals[:, :2].tolist()                            # [[gx, gy], ...]
        distances = goal_dists.tolist()
        coss = all_coss.tolist()
        sins = all_sins.tolist()
        collisions = all_collisions
        goals = all_arrives
        rewards_list = rewards.tolist()

        # === Visualization (only in eval mode when weights are provided) ===
        if combined_weights is not None:
            if combined_weights.dim() == 3:
                weights = combined_weights[0]
            else:
                weights = combined_weights
            
            weights_np = weights.cpu().numpy() if weights.is_cuda else weights.numpy()
            
            for i in range(self.num_robots):
                pos_i = positions[i]
                i_weights = weights_np[i]
                
                # Robot-robot attention lines
                for j in range(self.num_robots):
                    if i == j:
                        continue
                    w = float(i_weights[j])
                    if w > 0.01:
                        pos_j = positions[j]
                        self.env.draw_trajectory(
                            np.array([[pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]]]),
                            refresh=True, linewidth=w * 2, color='blue', alpha=0.6
                        )
                
                # Robot-obstacle attention lines
                for k in range(min(self.num_obstacles, len(i_weights) - self.num_robots)):
                    w = float(i_weights[self.num_robots + k])
                    if w > 0.01:
                        obs_pos = self.env.obstacle_list[k].position.flatten()
                        self.env.draw_trajectory(
                            np.array([[pos_i[0], obs_pos[0]], [pos_i[1], obs_pos[1]]]),
                            refresh=True, linewidth=w * 2, color='red', alpha=0.6
                        )

        # === Per-robot goal dwell + respawn ===
        if self.per_robot_goal_reset:
            for i in range(self.num_robots):
                is_at_goal = all_arrives[i]  # True while within goal_threshold

                # --- Transition: navigating → arrived (first entry) ---
                if is_at_goal and self.dwell_counters[i] < 0:
                    self.dwell_counters[i] = 0
                    # Keep the +100 goal reward from _compute_rewards_vectorized.
                    continue

                # --- Already in dwell phase ---
                if self.dwell_counters[i] >= 0:
                    self.dwell_counters[i] += 1

                    if all_collisions[i]:
                        # Collision during dwell: keep the collision penalty
                        # from _compute_rewards_vectorized.
                        rewards_list[i] = -50.0
                    elif is_at_goal:
                        # Holding position at goal → station-keeping reward
                        rewards_list[i] = self.station_keeping_reward
                    else:
                        rewards_list[i] = -10.0

                    # Respawn after dwell period ends (regardless of position —
                    # set_random_goal only changes the goal, not the robot's pose)
                    if self.dwell_counters[i] >= self.goal_dwell_min:
                        if random.random() < self.goal_respawn_prob:
                            # Corridor mode shuttles: a robot that arrived is
                            # sent back through the gate rather than given a
                            # goal on the side it is already standing on.
                            if self.layout is not None:
                                self._directions[i] *= -1
                                # In aligned mode the robot's current y keeps
                                # the shuttle in its lane; ignored otherwise.
                                range_limits = self.layout.goal_range_limits(
                                    self.x_range, self.y_range,
                                    int(self._directions[i]),
                                    start_y=float(
                                        self.env.robot_list[i].state[1]
                                    ),
                                )
                            else:
                                range_limits = [
                                    [self.x_range[0] + 1, self.y_range[0] + 1, -np.pi],
                                    [self.x_range[1] - 1, self.y_range[1] - 1, np.pi],
                                ]
                            self.env.robot_list[i].set_random_goal(
                                obstacle_list=self.env.obstacle_list,
                                init=True,
                                range_limits=range_limits,
                            )
                            self.prev_distances[i] = None
                            self.dwell_counters[i] = -1

        # Get obstacle states
        obstacle_states = self.get_obstacle_states()

        return (
            poses,
            distances,
            coss,
            sins,
            collisions,
            goals,
            action,
            rewards_list,
            positions,
            goal_positions_out,
            obstacle_states,
        )

    def reset(
        self,
        robot_state=None,
        robot_goal=None,
        random_obstacles=False,
        random_obstacle_ids=None,
        obstacle_states=None,
    ):
        """
        Reset the simulation environment.

        Args:
            robot_state: Initial robot state(s). If None, random states are assigned.
            robot_goal: Fixed goal position(s). If None, random goals are generated.
            random_obstacles (bool): If True, reposition obstacles randomly.
            random_obstacle_ids (list or None): IDs of obstacles to randomize.
            obstacle_states (np.ndarray or None): Obstacle states (N_obs, 3) as [[x], [y], [theta]].
                If provided, these exact positions are used instead of randomizing.

        Returns:
            tuple: (
                poses, distances, coss, sins, collisions, goals,
                action, rewards, positions, goal_positions, obstacle_states
            )
        """
        x_min = self.x_range[0] + 1
        x_max = self.x_range[1] - 1
        y_min = self.y_range[0] + 1
        y_max = self.y_range[1] - 1

        # Obstacles are placed *before* the robot poses, and the placement is
        # rejection-tested against everything except the robots.  Both halves of
        # that matter for replay: the pose loop below spaces robots against
        # ``obstacle_list``, so the obstacles must already hold this episode's
        # layout, and the obstacles must not look at the robots or their draw
        # count would depend on where the previous episode happened to leave
        # them.  Object positions survive ``reset`` (``set_state(init=True)``
        # rewrites ``_init_state``), and that carried-over state is the one
        # thing ``seed_episode`` cannot rewind — with either dependency in
        # place, the same seed replays a different world.
        if obstacle_states is not None:
            # Use provided obstacle states
            for oi, obs in enumerate(self.env.obstacle_list):
                obs.set_state(state=obstacle_states[oi], init=True)
        elif self.layout is not None:
            # Corridor mode: the whole obstacle set goes in the middle band,
            # spaced so a shielded coarse move can still thread it.  Placement
            # ignores `random_obstacle_ids` — the band *is* the scenery here.
            _, radii = self._obstacle_discs()
            states = sample_obstacles(
                radii, self.x_range, self.y_range, self.layout
            )
            for oi, obs in enumerate(self.env.obstacle_list):
                obs.set_state(state=states[oi].reshape(3, 1), init=True)
        elif random_obstacles:
            if random_obstacle_ids is None:
                random_obstacle_ids = [i + self.num_robots for i in range(7)]
            range_low = np.c_[[self.x_range[0], self.y_range[0], -np.pi]]
            range_high = np.c_[[self.x_range[1], self.y_range[1], np.pi]]
            # Static scenery (walls, non-randomized obstacles) is identical
            # every episode, so keeping it in the check costs no determinism.
            robots = set(id(r) for r in self.env.robot_list)
            existing = [
                obj for obj in self.env.objects
                if obj.id not in random_obstacle_ids and id(obj) not in robots
            ]
            for obs in self.env.obstacle_list:
                if obs.id not in random_obstacle_ids:
                    continue
                for _ in range(100):
                    obs.set_state(
                        state=np.random.uniform(range_low, range_high, (3, 1)),
                        init=True,
                    )
                    if not any(obs.check_collision(o) for o in existing):
                        break
                existing.append(obs)

        init_states = []
        if self.layout is not None:
            # Same acceptance test as below (0.6 apart, 0.5 clear of obstacles),
            # restricted to the start band — see corridor_layout.sample_starts.
            obstacle_xy, obstacle_r = self._obstacle_discs()
            starts = sample_starts(
                self.num_robots, self.x_range, self.y_range, self.layout,
                obstacle_xy, obstacle_r,
                robot_radius=self._bounding_radius(self.env.robot_list[0]),
            )
            for robot, start in zip(self.env.robot_list, starts):
                robot.set_state(state=start.reshape(3, 1), init=True)
            init_states = starts[:, :2].tolist()
        else:
            for robot in self.env.robot_list:
                conflict = True
                while conflict:
                    conflict = False
                    robot_state = [
                        [random.uniform(x_min, x_max)],
                        [random.uniform(y_min, y_max)],
                        [random.uniform(-np.pi, np.pi)],
                    ]
                    pos = [robot_state[0][0], robot_state[1][0]]

                    # Check robot-robot spacing
                    for loc in init_states:
                        vector = [pos[0] - loc[0], pos[1] - loc[1]]
                        if np.linalg.norm(vector) < 0.6:
                            conflict = True
                            break

                    # Check robot-obstacle clearance
                    if not conflict:
                        robot_point = Point(pos[0], pos[1])
                        for obs in self.env.obstacle_list:
                            if obs.geometry.distance(robot_point) < 0.5:
                                conflict = True
                                break

                init_states.append(pos)
                robot.set_state(state=np.array(robot_state), init=True)

        # Goal band per robot: the far side in corridor mode (so every episode
        # is a full traverse of the obstacle band), the whole arena otherwise.
        # Kept on self because a per-robot goal respawn flips it (see `step`).
        self._directions = directions = (
            self.layout.directions(self.num_robots)
            if self.layout is not None else None
        )
        # Ensure randomized goals are at least 0.5 away from each other
        goal_positions = []
        for ri, robot in enumerate(self.env.robot_list):
            if robot_goal is None:
                range_limits = (
                    self.layout.goal_range_limits(
                        self.x_range, self.y_range, int(directions[ri]),
                        start_y=init_states[ri][1],
                    )
                    if self.layout is not None
                    else [
                        [self.x_range[0] + 1, self.y_range[0] + 1, -np.pi],
                        [self.x_range[1] - 1, self.y_range[1] - 1, np.pi],
                    ]
                )
                for _ in range(10):
                    robot.set_random_goal(
                        obstacle_list=self.env.obstacle_list,
                        init=True,
                        range_limits=range_limits,
                    )
                    # Check distance to all previous goals
                    pos = robot.goal[:2] if hasattr(robot, 'goal') else robot.state[:2]
                    conflict = False
                    for g in goal_positions:
                        if np.linalg.norm(np.array(pos) - np.array(g)) < 0.5:
                            conflict = True
                            break
                    if not conflict:
                        break
                goal_positions.append(pos)
            else:
                robot.set_goal(np.array(robot_goal), init=True)

        self.env.reset()
        self.robot_goal = self.env.robot.goal
        
        # Reset previous distances for progress-based reward
        self.prev_distances = [None] * self.num_robots
        
        # Reset dwell counters
        self.dwell_counters = [-1] * self.num_robots
        
        # Randomly select inactive robots based on configured num_inactive_robots
        if self.num_inactive_robots > 0:
            self.inactive_ids = random.sample(range(self.num_robots), self.num_inactive_robots)
        else:
            self.inactive_ids = []
        
        # Update active mask
        self.active_mask = np.ones(self.num_robots, dtype=bool)
        for i in self.inactive_ids:
            self.active_mask[i] = False
        
        action = [[0.0, 0.0] for _ in range(self.num_robots)]
        (
            poses, distances, coss, sins, _, _, action, rewards,
            positions, goal_positions, obstacle_states
        ) = self.step(action, None)

        return (
            poses,
            distances,
            coss,
            sins,
            [False] * self.num_robots,
            [False] * self.num_robots,
            action,
            rewards,
            positions,
            goal_positions,
            obstacle_states,
        )

    @staticmethod
    def _compute_rewards_vectorized(
        goals, collisions, actions, pairwise_dists, distances,
        min_clearances, phase, prev_distances,
    ):
        """
        Compute rewards for ALL robots at once using numpy vectorization.

        Args:
            goals (list[bool]): Per-robot goal-reached flags.
            collisions (list[bool]): Per-robot collision flags.
            actions (np.ndarray): Actions of shape (N, 2).
            pairwise_dists (np.ndarray): Robot-robot distance matrix (N, N) with
                inf on the diagonal.
            distances (np.ndarray): Goal distances (N,).
            min_clearances (np.ndarray): Min obstacle clearances (N,).
            phase (int): Reward phase.
            prev_distances (np.ndarray): Previous goal distances (N,), NaN if None.

        Returns:
            np.ndarray: Per-robot rewards (N,).
        """
        N = len(goals)
        goals_arr = np.array(goals, dtype=bool)
        collisions_arr = np.array(collisions, dtype=bool)
        lin_vel = actions[:, 0]
        ang_vel = actions[:, 1]

        # --- Helper to compute robot-robot proximity penalty ---
        def _cl_penalty(threshold: float, weight: float = 1.0) -> np.ndarray:
            """Sum of weighted (threshold - dist)^2 for neighbours closer than threshold."""
            close_mask = pairwise_dists < threshold
            return weight * np.sum(
                np.where(close_mask, (threshold - pairwise_dists) ** 2, 0.0), axis=1
            )

        # --- Helper to compute obstacle proximity penalty ---
        def _obs_penalty(threshold: float, weight: float = 1.0) -> np.ndarray:
            """Weighted (threshold - clearance)^2 for obstacles closer than threshold."""
            obs_close = min_clearances < threshold
            return weight * np.where(
                obs_close, (threshold - min_clearances) ** 2, 0.0
            )

        # Default: all get the "normal" reward, then override goal/collision
        rewards = np.zeros(N)

        match phase:
            case 1:
                cl_penalties = _cl_penalty(threshold=1.25, weight=1.0)
                obs_pen = _obs_penalty(threshold=1.5, weight=1.0)
                r_dist = 1.5 / np.maximum(distances, 1e-10)
                rewards = lin_vel - 0.5 * np.abs(ang_vel) - cl_penalties - obs_pen + r_dist
                rewards = np.where(goals_arr, 100.0, rewards)
                rewards = np.where(collisions_arr, -100.0 * 3 * lin_vel, rewards)

            case 2:
                rewards = np.full(N, -0.1)
                rewards = np.where(goals_arr, 100.0, rewards)
                rewards = np.where(collisions_arr, -100.0, rewards)

            case 3:
                cl_penalties = _cl_penalty(threshold=1.25, weight=1.0)
                obs_pen = _obs_penalty(threshold=1.5, weight=2.0)
                r_dist = 10 * np.exp(-distances)
                rewards = lin_vel - 0.5 * np.abs(ang_vel) - cl_penalties - obs_pen + r_dist
                rewards = np.where(goals_arr, 100.0, rewards)
                rewards = np.where(collisions_arr, -100.0 * 3 * lin_vel, rewards)

            case 5:
                cl_penalties = _cl_penalty(threshold=1.25, weight=1.0)
                obs_pen = _obs_penalty(threshold=1.5, weight=2.0)
                k_p = 5.0
                has_prev = ~np.isnan(prev_distances)
                progress = np.where(has_prev, prev_distances - distances, 0.0)
                r_progress = k_p * progress
                rewards = lin_vel - 0.5 * np.abs(ang_vel) - cl_penalties - obs_pen + r_progress
                rewards = np.where(goals_arr, 100.0, rewards)
                rewards = np.where(collisions_arr, -100.0 * 3 * lin_vel, rewards)

            case 6:
                cl_penalties = _cl_penalty(threshold=1.25, weight=1.0)
                obs_pen = _obs_penalty(threshold=1.5, weight=2.0)
                k_p = 5.0
                has_prev = ~np.isnan(prev_distances)
                progress = np.where(has_prev, prev_distances - distances, 0.0)
                r_progress = k_p * progress
                rewards = lin_vel - cl_penalties - obs_pen + r_progress
                rewards = np.where(goals_arr, 100.0, rewards)
                rewards = np.where(collisions_arr, -100.0 * 3 * lin_vel, rewards)
            case 7:
                cl_penalties = _cl_penalty(threshold=1.5, weight=1.5)
                obs_pen = _obs_penalty(threshold=1.5, weight=2.0)
                k_p = 5.0
                has_prev = ~np.isnan(prev_distances)
                progress = np.where(has_prev, prev_distances - distances, 0.0)
                r_progress = k_p * progress
                rewards = - cl_penalties - obs_pen + r_progress
                rewards = np.where(goals_arr, 100.0, rewards)
                rewards = np.where(collisions_arr, -100.0 * 3 * lin_vel, rewards)
            case 8:
                cl_penalties = _cl_penalty(threshold=1.5, weight=1.5)
                obs_pen = _obs_penalty(threshold=1.5, weight=2.0)
                k_p = 5.0
                has_prev = ~np.isnan(prev_distances)
                progress = np.where(has_prev, prev_distances - distances, 0.0)
                r_progress = k_p * progress
                rewards = lin_vel- cl_penalties - obs_pen + r_progress
                rewards = np.where(goals_arr, 100.0, rewards)
                rewards = np.where(collisions_arr, -100.0 * 3 * lin_vel, rewards)

            case 9:
                # Phase 9: Group mixing training — proximity + collision only.
                # No progress reward.  Proximity penalty prevents the
                # zero-velocity degenerate solution (standing still near
                # robots/obstacles still accumulates penalty).
                cl_penalties = _cl_penalty(threshold=1.5, weight=1.5)
                obs_pen = _obs_penalty(threshold=1.5, weight=2.0)
                rewards = -cl_penalties - obs_pen
                rewards = np.where(goals_arr, 10.0, rewards)
                rewards = np.where(collisions_arr, -100.0, rewards)
         
            case _:
                raise ValueError(f"Unknown reward phase: {phase}")

        return rewards

    @staticmethod
    def get_reward(
        goal, collision, action, closest_robots, distance,
        min_obstacle_clearance, obstacle_threshold, phase=3,
        prev_distance=None
    ):
        """
        Compute the reward for a single robot with obstacle penalty.

        Args:
            goal (bool): Whether the robot reached its goal.
            collision (bool): Whether the robot collided.
            action (list): [linear_velocity, angular_velocity].
            closest_robots (list): Distances to other robots.
            distance (float): Current distance to the goal.
            min_obstacle_clearance (float): Minimum clearance to any obstacle.
            obstacle_threshold (float): Distance threshold for obstacle penalty.
            phase (int): Reward configuration (1, 2, 3, or 5).
            prev_distance (float or None): Previous distance to goal (for progress reward).

        Returns:
            float: Computed scalar reward.
        """
        match phase:
            case 1:
                if goal:
                    return 100.0
                elif collision:
                    return -100.0 * 3 * action[0]
                else:
                    r_dist = 1.5 / distance

                    # Robot proximity penalty
                    cl_pen = 0
                    for rob in closest_robots:
                        add = (1.25 - rob) ** 2 if rob < 1.25 else 0
                        cl_pen += add

                    # Obstacle proximity penalty
                    obs_pen = 0
                    if min_obstacle_clearance < obstacle_threshold:
                        obs_pen = (obstacle_threshold - min_obstacle_clearance) ** 2

                    return action[0] - 0.5 * abs(action[1]) - cl_pen - obs_pen + r_dist

            case 2:
                if goal:
                    return 100.0
                elif collision:
                    return -100.0 
                else:
                    return -0.1

            case 3:
                # Phase 3: Stronger obstacle avoidance emphasis
                if goal:
                    return 100.0
                elif collision:
                    return -100.0 * 3 * action[0]
                else:
                    r_dist = 10 * np.exp(-distance)

                    # Robot proximity penalty
                    cl_pen = 0
                    for rob in closest_robots:
                        add = (1.25 - rob) ** 2 if rob < 1.25 else 0
                        cl_pen += add

                    # Stronger obstacle penalty with exponential scaling
                    obs_pen = 0
                    if min_obstacle_clearance < obstacle_threshold:
                        obs_pen = 2.0 * (obstacle_threshold - min_obstacle_clearance) ** 2

                    return action[0] - 0.5 * abs(action[1]) - cl_pen - obs_pen + r_dist
            
            case 5:
                # Phase 5: Progress-based reward (change in distance to goal)
                if goal:
                    return 100.0
                elif collision:
                    return -100.0 * 3 * action[0]
                else:
                    # Progress reward: positive if moving closer, negative if moving away
                    k_p = 5.0  # Progress reward scaling factor
                    if prev_distance is not None:
                        progress = prev_distance - distance
                        r_progress = k_p * progress
                    else:
                        # First step: no previous distance, use small baseline
                        r_progress = 0.0

                    # Robot proximity penalty
                    cl_pen = 0
                    for rob in closest_robots:
                        add = (1.25 - rob) ** 2 if rob < 1.25 else 0
                        cl_pen += add

                    # Obstacle proximity penalty
                    obs_pen = 0
                    if min_obstacle_clearance < obstacle_threshold:
                        obs_pen = 2.0 * (obstacle_threshold - min_obstacle_clearance) ** 2

                    return action[0] - 0.5 * abs(action[1]) - cl_pen - obs_pen + r_progress
                
            case 6:
                # Phase 6: No restriction on robot rotation
                if goal:
                    return 100.0
                elif collision:
                    return -100.0 * 3 * action[0]
                else:
                    # Progress reward: positive if moving closer, negative if moving away
                    k_p = 5.0  # Progress reward scaling factor
                    if prev_distance is not None:
                        progress = prev_distance - distance
                        r_progress = k_p * progress
                    else:
                        # First step: no previous distance, use small baseline
                        r_progress = 0.0

                    # Robot proximity penalty
                    cl_pen = 0
                    for rob in closest_robots:
                        add = (1.25 - rob) ** 2 if rob < 1.25 else 0
                        cl_pen += add

                    # Obstacle proximity penalty
                    obs_pen = 0
                    if min_obstacle_clearance < obstacle_threshold:
                        obs_pen = 2.0 * (obstacle_threshold - min_obstacle_clearance) ** 2

                    return action[0] - cl_pen - obs_pen + r_progress
                
            case _:
                raise ValueError(f"Unknown reward phase: {phase}")
