"""
Base policies (priors) over lazy branch stubs.

A prior is any callable ``prior(model, ms, branches) -> logits`` (one logit per
:class:`Branch` stub, same order).  It must be **stub-compatible**: at call
time a branch carries only ``(mode, group, step_cost)`` — no candidate, no
child, no shield statistics.  This is the corrected rationale of the lazy
design: the prior is the *substitute* for per-edge verification, so it can
never be allowed to depend on verification output.  (The 6-robot
``HeuristicPrior`` — built from shield vetting statistics — is therefore
unusable here by construction; that dependence was exactly the cost the prior
was supposed to save.)

A prior may additionally expose ``feasibility(model, ms, branches) -> bool
array`` — a *predicted* shield mask used to steer selection away from coarse
edges the net expects to be refuted.  It is advisory only: the real shield
vet corrects it the moment an edge is materialised, and the executed root
action is always exactly vetted.

``UniformPrior`` (flat, no feasibility) is the phase-0 data-collection base
policy; the learned prior (``prior_net.LearnedPrior``) plugs in behind the
same signature from iteration 1 on.
"""

from __future__ import annotations

import numpy as np


class UniformPrior:
    """Flat prior — every edge equally likely (phase 0 / tests / baseline)."""

    def __call__(self, model, ms, branches) -> np.ndarray:
        return np.zeros(len(branches), dtype=np.float64)
