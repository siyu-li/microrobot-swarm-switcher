"""
CAPSwitcher (Coarse-And-Precise Switcher) for coupled unicycle swarms.

Exports the complete CAPSwitcher implementation:

Policies (frozen backbone + coarse steering + readout)
  CoarseSteering       — least-squares pinv(A) group steering
  GATBackbone          — frozen TD3Obstacle actor wrapper (per-robot embeddings)
  DeepSetsHead         — permutation-invariant readout over per-robot embeddings
  
Environment / planning
  SwitcherEnv          — Gym-like environment (binary mode selection)

Usage:
    python -m robot_nav.eval_mpc
"""

from robot_nav.models.MARL.capswitcher.policies.coarse_steering import CoarseSteering
from robot_nav.models.MARL.capswitcher.policies.gat_backbone import GATBackbone
from robot_nav.models.MARL.capswitcher.policies.deep_sets_head import DeepSetsHead
from robot_nav.models.MARL.capswitcher.rl.switcher_env import SwitcherEnv

__all__ = [
    "CoarseSteering",
    "GATBackbone",
    "DeepSetsHead",
    "SwitcherEnv",
]
