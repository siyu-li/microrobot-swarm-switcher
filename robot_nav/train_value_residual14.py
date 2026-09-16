"""
Train the bounded value residual (``value_residual.py``) — self-contained:
everything comes from the ``eval_gaz14_lazy --log-pi-targets`` shards (no
``collect_leaf_data`` dependency; that collector is slated for deletion).

Data per root decision: v2 global context ``global_feats_v2``, realized
suffix cost ``cost_to_go`` (harvested by the in-harness label collector),
episode ``outcome``, ``episode_seed``, plus the shard-level analytic ``alpha``
and ``goal_threshold``.  The analytic level ``h = α · Σ_{d>thr} d`` is
recomputed here from the stored raw ``robot_states`` (columns 0/1 = position,
9/10 = goal — the GAT layout), so feature iteration never re-runs episodes.

Model: ``V = B · (1 + band · tanh(MLP([B/h_scale, G0..G3])))`` with a
selectable base level ``--value-base``:

* ``analytic``  — ``B = h = α·Σd`` (free at the leaf, but realized/h
  measured ≈3.5–3.9, so small bands saturate);
* ``geometry``  — ``B`` = the per-robot learned precise cost-to-go
  (``--geometry-ckpt``, feature="geometry": its 11-col input is exactly the
  stored ``robot_states`` rows, so the base is reproduced offline bit-for-bit
  with deploy).  Its realized-rollout level should sit far closer to the
  labels; shards must carry ``precise_unit``/``selection_interval``.

Training is in **ratio space** — the target is ``clip(y/B − 1, ±band)/band``
— because the modulation is multiplicative and the retired-value post-mortem
showed the level error, not the slope, is what the base cannot see.  Rows by
outcome:

* SUCCESS — exact labels, Huber;
* TIMEOUT — censored (the episode was cut, true completion cost is at least
  the realized suffix): one-sided hinge pushing the prediction *up* only
  (``--failures hinge``, the default) or dropped (``--failures drop``);
* COLLISION / UNSOLVABLE / ENDED — always dropped (no completion-cost
  reading exists).

Near-terminal states (h below ``--h-min`` metres of aggregate distance × α)
are skipped: the ratio explodes and the leaf there is decided by exact costs
anyway.

The printed ``band coverage`` is the fraction of exact labels inside the
±25% band — if most sit outside, the tanh saturates and the band (not the
net) is the binding constraint: report it before believing any other metric.

The analytic baseline (predict ``h`` itself) is evaluated on the same val
rows; the residual must beat it or it is dead weight at the leaf.

Usage:
    PYTHONPATH=. python -m robot_nav.train_value_residual14 \
        --data data/pi_targets_v2 [dir2 ...] --out-dir runs/gaz14_fp/value0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from robot_nav.models.MARL.capswitcher_14.rl.search.value_residual import (
    ValueResidualNet,
)

_NEED = ("global_feats_v2", "cost_to_go", "outcome", "episode_seed",
         "robot_states")
SUCCESS, TIMEOUT = 0, 1     # eval_gaz14_lazy.OUTCOME_CODES, frozen here


def load_shards(dirs: list[str]) -> dict:
    paths = sorted(p for d in dirs for p in Path(d).rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no .npz shards under {dirs}")
    parts: dict[str, list] = {c: [] for c in _NEED}
    alpha = thr = None
    for p in paths:
        z = np.load(p, allow_pickle=False)
        missing = [c for c in (*_NEED, "alpha", "goal_threshold")
                   if c not in z.files]
        if missing:
            raise SystemExit(
                f"{p} lacks {missing} — recollect with the updated "
                "eval_gaz14_lazy --log-pi-targets (value-label collector)")
        for c in _NEED:
            parts[c].append(z[c])
        a, t = float(z["alpha"]), float(z["goal_threshold"])
        if alpha is not None and (abs(a - alpha) > 1e-9 or abs(t - thr) > 1e-9):
            raise SystemExit(f"{p}: alpha/goal_threshold differ across shards "
                             "— cost labels are not in one unit system")
        alpha, thr = a, t
    data = {c: np.concatenate(parts[c], axis=0) for c in _NEED}
    data["alpha"], data["goal_threshold"] = alpha, thr
    # Optional pricing block (post-2026-09-08 shards) for the geometry base.
    z0 = np.load(paths[0], allow_pickle=False)
    if "precise_unit" in z0.files:
        data["precise_unit"] = float(z0["precise_unit"])
        data["selection_interval"] = float(z0["selection_interval"])
    # Analytic level from raw states: cols 0/1 position, 9/10 goal.
    rs = data["robot_states"].astype(np.float64)
    d = np.hypot(rs[:, :, 9] - rs[:, :, 0], rs[:, :, 10] - rs[:, :, 1])
    data["dist"] = d
    data["h"] = (alpha * (d * (d > thr)).sum(axis=1)).astype(np.float64)
    print(f"Loaded {len(data['h'])} decisions from {len(paths)} shards "
          f"(alpha={alpha:.2f}, outcomes: "
          f"{np.bincount(data['outcome'].clip(0, 4), minlength=5).tolist()})")
    return data


def geometry_base(data: dict, ckpt_path: str) -> np.ndarray:
    """Offline replica of ``LearnedCostToGo`` over the stored robot_states."""
    from robot_nav.models.MARL.capswitcher.rl.value_net import (
        load_value_checkpoint,
    )

    if "precise_unit" not in data:
        raise SystemExit("shards lack precise_unit/selection_interval — "
                         "recollect with the updated eval_gaz14_lazy to use "
                         "--value-base geometry")
    net, ckpt = load_value_checkpoint(ckpt_path)
    if ckpt["feature"] != "geometry":
        raise SystemExit(f"{ckpt_path} is a {ckpt['feature']!r} value net — "
                         "the offline base needs feature='geometry' (its "
                         "input is the stored robot_states rows)")
    rs = torch.as_tensor(data["robot_states"], dtype=torch.float32)
    x = (rs - torch.as_tensor(ckpt["x_mean"])) / torch.as_tensor(ckpt["x_std"])
    with torch.no_grad():
        v = net(x).clamp_min(0.0).numpy()                       # (S, N)
    unreached = data["dist"] > data["goal_threshold"]
    per_decision = data["precise_unit"] * data["selection_interval"]
    return (per_decision * (v * unreached).sum(axis=1)).astype(np.float64)


def episode_split(seeds: np.ndarray, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    uniq = np.unique(seeds)
    val_eps = rng.permutation(uniq)[: max(1, int(len(uniq) * val_frac))]
    val = np.isin(seeds, val_eps)
    return ~val, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, nargs="+", required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--band", type=float, default=0.5)
    ap.add_argument("--value-base", choices=("analytic", "geometry"),
                    default="analytic",
                    help="base level the residual modulates")
    ap.add_argument("--geometry-ckpt", type=str,
                    default="robot_nav/models/MARL/capswitcher/checkpoint/"
                            "value_local/value_geometry.pt",
                    help="per-robot geometry value net "
                         "(--value-base geometry)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--failures", choices=("hinge", "drop"), default="hinge",
                    help="TIMEOUT rows: one-sided lower-bound hinge, or drop")
    ap.add_argument("--h-min", type=float, default=0.5,
                    help="skip states with < this many metres of aggregate "
                         "unreached distance (ratio labels explode)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device

    data = load_shards(args.data)
    if args.value_base == "geometry":
        h = geometry_base(data, args.geometry_ckpt)
        print(f"geometry base: median {np.median(h):.0f} vs analytic "
              f"{np.median(data['h']):.0f}")
    else:
        h = data["h"]
    y = data["cost_to_go"].astype(np.float64)
    exact = (data["outcome"] == SUCCESS) & np.isfinite(y)
    censored = ((data["outcome"] == TIMEOUT) & np.isfinite(y)
                if args.failures == "hinge"
                else np.zeros_like(exact))
    usable = (exact | censored) & (h > data["alpha"] * args.h_min)
    tr_rows, va_rows = episode_split(
        data["episode_seed"], args.val_frac, args.seed)

    ratio = np.where(h > 0, y / np.maximum(h, 1e-9), 1.0)
    m_target = np.clip(ratio - 1.0, -args.band, args.band) / args.band
    in_band = np.abs(ratio - 1.0) <= args.band
    print(f"rows: {int((usable & tr_rows).sum())} train / "
          f"{int((usable & va_rows).sum())} val "
          f"({int(censored[usable].sum())} censored) | "
          f"band coverage (exact rows): {in_band[exact & usable].mean():.2%} "
          f"| median ratio {np.median(ratio[exact & usable]):.3f}")

    if not (usable & tr_rows).any() or not (usable & va_rows).any():
        raise SystemExit(
            f"empty split: {int((usable & tr_rows).sum())} train / "
            f"{int((usable & va_rows).sum())} val usable rows — collect more "
            "episodes (labels come only from SUCCESS/TIMEOUT episodes)")

    h_scale = float(np.median(h[usable & tr_rows]))
    net = ValueResidualNet(
        hidden=args.hidden, band=args.band, h_scale=h_scale,
        alpha=data["alpha"], base=args.value_base,
        base_ckpt=(args.geometry_ckpt if args.value_base == "geometry"
                   else None),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    def tensors(mask):
        idx = np.flatnonzero(mask)
        return (torch.as_tensor(h[idx], dtype=torch.float32),
                torch.as_tensor(data["global_feats_v2"][idx]),
                torch.as_tensor(m_target[idx], dtype=torch.float32),
                torch.as_tensor(censored[idx]),
                torch.as_tensor(y[idx], dtype=torch.float32))

    tr = tensors(usable & tr_rows)
    va = tuple(t.to(device) for t in tensors(usable & va_rows))
    va_exact = ~va[3]
    if not bool(va_exact.any()):
        print("WARNING: no exact (SUCCESS) rows in the val split — model "
              "selection falls back to the censored hinge; cost-MAE and the "
              "must-beat verdict are unavailable this round")

    # Must-beat bar on val exact rows: predict the (unmodulated) base itself.
    base_mae = (float((va[0][va_exact] - va[4][va_exact]).abs().mean())
                if bool(va_exact.any()) else float("nan"))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(json.dumps(vars(args), indent=2))
    log = open(out / "log.jsonl", "a")
    n_train = len(tr[0])
    best = float("inf")
    best_row = None
    for epoch in range(args.epochs):
        net.train()
        order = torch.randperm(n_train)
        tot = nb = 0
        for i in range(0, n_train, args.batch):
            idx = order[i:i + args.batch]
            hb, gb, mb, cb, _ = (t[idx].to(device) for t in tr)
            m = net(hb, gb)
            loss_exact = F.huber_loss(m[~cb], mb[~cb]) if (~cb).any() else 0.0
            # Censored: realized suffix is a lower bound — push up only.
            loss_cens = (F.relu(mb[cb] - m[cb]).pow(2).mean()
                         if cb.any() else 0.0)
            loss = loss_exact + loss_cens
            if not torch.is_tensor(loss):      # batch had no usable rows
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        net.eval()
        with torch.no_grad():
            hb, gb, mb, cb, yb = va
            m = net(hb, gb)
            v = hb * (1.0 + args.band * m)
            # Selection metric from whatever val evidence exists: exact Huber
            # plus the censored hinge (always defined when either side is).
            val_sel = 0.0
            val_huber = float("nan")
            if bool(va_exact.any()):
                val_huber = float(F.huber_loss(m[va_exact], mb[va_exact]))
                val_sel += val_huber
            if bool(cb.any()):
                val_sel += float(F.relu(mb[cb] - m[cb]).pow(2).mean())
            val_mae = (float((v[va_exact] - yb[va_exact]).abs().mean())
                       if bool(va_exact.any()) else float("nan"))
        row = {"epoch": epoch, "train_loss": tot / max(nb, 1),
               "val_sel": val_sel, "val_huber_m": val_huber,
               "val_mae_cost": val_mae, "baseline_mae_cost": base_mae}
        log.write(json.dumps(row) + "\n")
        log.flush()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"ep {epoch:3d} | loss {row['train_loss']:.4f} | val m-huber "
                  f"{val_huber:.4f} cost-MAE {val_mae:.0f} "
                  f"(bare {args.value_base} base: {base_mae:.0f})")
        if np.isfinite(val_sel) and val_sel < best:
            best = val_sel
            best_row = row
            net.save(out / "value_best.pt", extra=row)
    net.save(out / "value_last.pt", extra=row)
    if best_row is None:
        raise SystemExit("no epoch produced a finite val loss — check labels")
    if np.isfinite(base_mae):
        verdict = ("BEATS" if best_row["val_mae_cost"] < base_mae
                   else "DOES NOT BEAT")
        print(f"best val cost-MAE {best_row['val_mae_cost']:.0f} {verdict} "
              f"bare {args.value_base} base {base_mae:.0f} — deploy with "
              f"eval_gaz14_lazy --value-residual {out / 'value_best.pt'}")
    else:
        print(f"best val selection loss {best:.4f} (no exact val rows — "
              f"censored-only round, verdict n/a) — deploy with "
              f"eval_gaz14_lazy --value-residual {out / 'value_best.pt'}")


if __name__ == "__main__":
    main()
