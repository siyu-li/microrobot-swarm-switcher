# Learning to Switch Between Coarse and Precise Motion in Group-Controlled Microrobot Swarms

Code for the paper. A closed-loop, budgeted planner for a group-controlled
MicroStressBot swarm (14 robots) that switches between low-cost **coarse**
group primitives and a costly **precise** per-robot primitive. The switcher is
a budgeted Gumbel AlphaZero (GAZ) tree search guided by three learned models —
a feasibility estimator, a scheduling policy, and a value function — trained
iteratively from the planner's own search data, without expert demonstrations.

## Layout

```
robot_nav/
├── SIM_ENV/                          # IR-SIM wrappers (14-robot obstacle world)
├── worlds/*_14robots.yaml            # IR-SIM world definitions
├── models/MARL/
│   ├── Attention/                    # graph-attention building blocks (GAT)
│   ├── marlTD3/                      # TD3 trainer for the GAT local controller
│   ├── capswitcher/                  # robot-count-generic switcher utilities:
│   │   ├── policies/                 #   frozen GAT backbone wrapper, coarse steering
│   │   └── rl/                       #   cost model, collision shield, switcher env, value nets
│   └── capswitcher_14/               # the 14-robot system used in the paper:
│       ├── cost_14robots.yaml        #   primitive library costs (Tables III–IV)
│       ├── policies/                 #   22 coarse groups, BFGS coarse steering (Eq. 4–6),
│       │                             #   coupled precise execution (Eq. 8)
│       ├── rl/forward_model.py       #   primitive transition maps Φ_P
│       └── rl/search/                #   budgeted GAZ (gumbel.py, tree.py),
│                                     #   A*/PHS baselines (best_first.py),
│                                     #   feasibility/policy nets (feas_policy_net.py),
│                                     #   value residual (value_residual.py), features
├── marl_train_obstacle_14robots.py   # train the GAT local controller (TD3, all robots active)
├── marl_finetune_partial_inactive.py # fine-tune under partially-active configurations
├── scripts/run_az_cycles.sh          # AlphaZero-style improvement loop (Sec. V-B)
├── train_feas_policy14.py            # feasibility estimator + scheduling policy training
├── train_value_residual14.py         # value-correction network training (Eq. 13–14)
├── train_prior.py                    # legacy monolithic prior (masked-KL loss reused above)
├── eval_gaz14_lazy.py                # GAZ evaluation / self-play collection (Tables V–VI)
├── eval_gaz14_baselines.py           # guided/analytic A*, PHS baselines (Table VI)
├── plot_traj14.py                    # trajectory figures (Fig. 4)
└── plot_coarse_move14.py             # coarse-primitive illustration (Fig. 1b)
```

## Setup

Python ≥ 3.10. Dependencies are listed in `pyproject.toml`; the main ones are
`ir-sim`, `torch`, `torch-geometric`, `scipy`, `matplotlib`. All commands are
run from the repository root with `PYTHONPATH=.`.

Planning and evaluation run entirely on CPU; the GPU is only used for training
the GAT controller.

## Reproducing the paper

**1. Local controller (Sec. IV-B).** Train the graph-attention controller with
TD3, then fine-tune it under partially-active configurations to match the
precise-execution distribution:

```bash
python -m robot_nav.marl_train_obstacle_14robots
python -m robot_nav.marl_finetune_partial_inactive
```

The frozen checkpoint is consumed by all switcher scripts via
`--backbone-ckpt` (default path under `robot_nav/models/MARL/marlTD3/checkpoint/`).

**2. Policy-improvement training (Sec. V-B).** Alternate self-play data
collection and training of the feasibility, policy, and value models:

```bash
bash robot_nav/scripts/run_az_cycles.sh
```

Cycle 0 plans with a uniform prior and the analytic leaf value; each later
cycle deploys the previous cycle's nets (`--feas-policy-dir`,
`--value-residual`). Outputs land in `runs/az6/cycle<t>/{data,stats,nets,value}`.

**3. GAZ evaluation (Tables V–VI).**

```bash
python -m robot_nav.eval_gaz14_lazy --episodes 100 --budgets 200 \
    --feas-policy-dir runs/az6/cycle<t>/nets \
    --value-residual runs/az6/cycle<t>/value/value_best.pt
```

`--budgets` accepts several values for the budget ablation (Table V). For the
execution-disturbance study (Sec. V-A.a, Table VI), add the heading-noise
knobs — `--coarse-heading-bias-deg` is the per-episode/per-robot bias σ_b and
`--coarse-heading-white-deg` the per-primitive white component σ_w:

```bash
python -m robot_nav.eval_gaz14_lazy ... \
    --coarse-heading-bias-deg 5 --coarse-heading-white-deg 2
```

**4. Best-first baselines (Table VI).** Guided/analytic A* and PHS over the
same primitive space, with the same learned guidance, in receding-horizon
mode:

```bash
python -m robot_nav.eval_gaz14_baselines --episodes 100 --receding \
    --algos astar levints phs phs-star \
    --feas-policy-dir runs/az6/cycle<t>/nets \
    --value-residual runs/az6/cycle<t>/value/value_best.pt
```

Both eval scripts accept `--seed` blocks plus `--out`/`--merge` for exact
multi-worker table recombination, and share the disturbance flags so the
robustness comparison is defined identically across planners.

**5. Figures.**

```bash
python -m robot_nav.plot_traj14         # planned/executed trajectories (Fig. 4)
python -m robot_nav.plot_coarse_move14  # coarse primitive illustration (Fig. 1b)
```

## Notes

- Trained checkpoints (`robot_nav/models/MARL/marlTD3/checkpoint/`, `runs/`)
  are not included in the repository; the full pipeline above regenerates
  them (~11–12 h for the 12 improvement cycles on a 12-core desktop CPU).
- `capswitcher/` contains the robot-count-generic utilities shared by the
  14-robot system; `capswitcher_14/` holds everything specific to the paper's
  14-robot instantiation.
