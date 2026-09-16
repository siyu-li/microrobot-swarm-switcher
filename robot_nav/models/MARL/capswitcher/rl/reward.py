"""
Switcher action ids, shared by every CAPSwitcher module.

Decision pricing lives in ``rl/cost.py`` (:class:`SwitcherCost`, loaded from
the per-system cost YAML).  The old ``PathCostReward`` — terminal
collision/all-goal/out-of-bounds bonuses plus a flat per-decision precise
cost — is retired: the environment's reward is simply ``−path_cost`` and
terminal events are ``done`` flags, not reward terms.
"""

from __future__ import annotations

# Action ids.
COARSE: int = 0
PRECISE: int = 1
