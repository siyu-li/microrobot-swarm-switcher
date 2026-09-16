#!/usr/bin/env bash
# AlphaZero plan->distill cycles for the 14-robot lazy Gumbel switcher.
#
# Each cycle: collect EPISODES episodes at BUDGET with WORKERS parallel
# workers (disjoint seed blocks, shared policy seed -> exact-merge), then
# train the split feas/policy prior and the value residual on the shard MIX
# of every cycle so far (replay guard).  Cycle 0 runs the uniform prior with
# the analytic leaf; cycle t>0 deploys cycle t-1's three nets.  The root
# stays eagerly vetted in every cycle (balanced feasibility labels).
#
# Workers run on CPU, 2 torch threads each: per-transition speed measured
# equal to CUDA for this GAT (batch-1 inference), no GPU contention, and no
# CUDA-nondeterministic forward drift.
#
# Usage (from the repo root):
#   bash robot_nav/scripts/run_az_cycles.sh
# Layout: runs/az6/cycle<t>/{data/w<k>/,stats/,nets/,value/,worker<k>.log}

set -uo pipefail

PY=${PY:-/home/siyu/miniconda3/envs/DRL_nav/bin/python}
ROOT=${ROOT:-runs/az6}
CYCLES=${CYCLES:-6}
EPISODES=${EPISODES:-240}
WORKERS=${WORKERS:-8}
BUDGET=${BUDGET:-200}
SEED0=${SEED0:-20000}
# Stall break OFF by default: the loop must stand without the forced-precise
# crutch (stall-on value labels bake the rescue into the learned cost).
STALL_STEPS=${STALL_STEPS:-0}
# Extra eval_gaz14_lazy args appended to every worker (word-split on
# purpose), e.g. EXTRA_ARGS="--coupled-precise" for the pinv physics.
EXTRA_ARGS=${EXTRA_ARGS:-}
export PYTHONPATH=.

PER=$((EPISODES / WORKERS))
mkdir -p "$ROOT"
echo "=== $(date) az cycles: $CYCLES x $EPISODES eps, b$BUDGET, $WORKERS workers ==="

for ((t = 0; t < CYCLES; t++)); do
    CDIR=$ROOT/cycle$t
    if [[ -f "$CDIR/value/value_best.pt" ]]; then
        echo "=== cycle $t already complete, skipping ==="
        continue
    fi
    mkdir -p "$CDIR/data" "$CDIR/stats"

    EXTRA=()
    if ((t > 0)); then
        PREV=$ROOT/cycle$((t - 1))
        EXTRA=(--feas-policy-dir "$PREV/nets"
               --value-residual "$PREV/value/value_best.pt")
    fi

    echo "=== $(date) cycle $t: collecting $EPISODES episodes ==="
    pids=()
    for ((w = 0; w < WORKERS; w++)); do
        seed=$((SEED0 + t * 1000 + w * PER))
        OMP_NUM_THREADS=2 $PY -m robot_nav.eval_gaz14_lazy \
            --episodes "$PER" --seed "$seed" --budgets "$BUDGET" \
            --policy-seed $((77 + t)) --device cpu \
            --stall-steps "$STALL_STEPS" \
            --log-pi-targets "$CDIR/data/w$w" --out "$CDIR/stats" \
            "${EXTRA[@]}" $EXTRA_ARGS > "$CDIR/worker$w.log" 2>&1 &
        pids+=($!)
        sleep 3          # stagger imports (RAM spike at startup)
    done
    fail=0
    for p in "${pids[@]}"; do
        wait "$p" || fail=1
    done
    if ((fail)); then
        echo "!!! cycle $t: a worker failed — see $CDIR/worker*.log" >&2
        exit 1
    fi

    $PY -m robot_nav.eval_gaz14_lazy --merge "$CDIR/stats" \
        > "$CDIR/merged_table.txt" 2>&1 || true
    tail -n 40 "$CDIR/merged_table.txt"

    # Replay mix: every cycle's data dirs up to and including this one.
    DATA=()
    for ((s = 0; s <= t; s++)); do
        DATA+=("$ROOT/cycle$s/data")
    done

    echo "=== $(date) cycle $t: training feas + policy ==="
    $PY -m robot_nav.train_feas_policy14 --data "${DATA[@]}" \
        --out-dir "$CDIR/nets" > "$CDIR/train_fp.log" 2>&1 \
        || { echo "!!! cycle $t: feas/policy training failed" >&2; exit 1; }
    tail -n 3 "$CDIR/train_fp.log"

    echo "=== $(date) cycle $t: training value residual (geometry base) ==="
    $PY -m robot_nav.train_value_residual14 --data "${DATA[@]}" \
        --out-dir "$CDIR/value" --value-base geometry \
        > "$CDIR/train_value.log" 2>&1 \
        || { echo "!!! cycle $t: value training failed" >&2; exit 1; }
    tail -n 3 "$CDIR/train_value.log"

    echo "=== $(date) cycle $t done ==="
done
echo "=== $(date) all $CYCLES cycles complete ==="
