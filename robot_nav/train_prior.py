"""
Train the learned prior (PriorNet) from logged root decisions — sim-free.

Data: ``.npz`` shards written by ``eval_gaz14_lazy --log-pi-targets`` (one sample
per real decision).  Pass **all** iteration directories to ``--data``: mixing
shards across plan→distill iterations is the replay guard against
distribution shift between teachers (do not train on only the latest).

Losses (jointly, one trunk):

* **Policy** — masked KL(π′ ‖ softmax(logits)): the settled root-only
  distillation target; illegal edges are masked out of the softmax (π′ is
  already zero there).
* **Feasibility** — Huber on the predicted clearance margin against the exact
  shield label ``clearance − d_safe``, clipped to ±``--margin-clip`` (raw
  clearances are unbounded above / +inf when nothing is sweepable).

Usage:
    python -m robot_nav.train_prior --data data/pi_targets_14 [dir2 ...] \
        --out-dir robot_nav/models/MARL/capswitcher_14/checkpoint/prior_iter1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from robot_nav.models.MARL.capswitcher_14.rl.search.prior_net import PriorNet


def load_shards(dirs: list[str]) -> dict[str, np.ndarray]:
    """Concatenate every shard found under ``dirs`` (recursive glob)."""
    paths = sorted(p for d in dirs for p in Path(d).rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no .npz shards under {dirs}")
    cols = ("group_feats", "global_feats", "pi_prime", "legal", "clearance")
    parts = {c: [] for c in cols}
    margins = []
    for p in paths:
        z = np.load(p, allow_pickle=False)
        for c in cols:
            parts[c].append(z[c])
        d_safe = float(z["d_safe"])
        margins.append(z["clearance"] - d_safe)
    data = {c: np.concatenate(parts[c], axis=0) for c in cols}
    data["margin"] = np.concatenate(margins, axis=0)
    print(f"Loaded {data['pi_prime'].shape[0]} samples from {len(paths)} shards")
    return data


def masked_policy_kl(
    logits: torch.Tensor, pi_target: torch.Tensor, legal: torch.Tensor
) -> torch.Tensor:
    """Mean KL(π′ ‖ softmax(logits over legal edges)); illegal edges masked."""
    masked = logits.masked_fill(~legal, float("-inf"))
    logp = F.log_softmax(masked, dim=1)
    # π′ is zero on illegal edges; guard 0·(−inf) → 0.
    ll = torch.where(pi_target > 0, logp, torch.zeros_like(logp))
    return -(pi_target * ll).sum(dim=1).mean()


def train_prior(
    data_dirs: list[str],
    out_dir: str | Path,
    *,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden: int = 128,
    w_feas: float = 1.0,
    margin_clip: float = 2.0,
    val_frac: float = 0.1,
    seed: int = 0,
    device: torch.device | None = None,
    log_every: int = 10,
) -> dict:
    """
    Distil the logged root policies into a :class:`PriorNet`.

    Pass **every** iteration's shard directory in ``data_dirs`` — mixing across
    teachers is the replay guard against distribution shift.

    Returns:
        Metrics of the run: ``n_samples``, ``best_val``, final-epoch
        ``val_kl`` / ``val_feas`` / ``feas_acc``, and ``best_path``.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_shards(list(data_dirs))
    n = data["pi_prime"].shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def to_tensors(idx):
        return (
            torch.as_tensor(data["group_feats"][idx], dtype=torch.float32),
            torch.as_tensor(data["global_feats"][idx], dtype=torch.float32),
            torch.as_tensor(data["pi_prime"][idx], dtype=torch.float32),
            torch.as_tensor(data["legal"][idx], dtype=torch.bool),
            torch.as_tensor(
                np.clip(data["margin"][idx], -margin_clip, margin_clip),
                dtype=torch.float32,
            ),
        )

    train = to_tensors(train_idx)
    val = tuple(t.to(device) for t in to_tensors(val_idx))

    net = PriorNet(hidden=hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    n_train = train[0].shape[0]
    val_kl = val_feas = acc = float("nan")

    for epoch in range(epochs):
        net.train()
        order = torch.randperm(n_train)
        tot, nb = 0.0, 0
        for i in range(0, n_train, batch_size):
            idx = order[i : i + batch_size]
            gf, glf, pi, legal, margin = (t[idx].to(device) for t in train)
            logits, margin_pred = net(gf, glf)
            loss = (
                masked_policy_kl(logits, pi, legal)
                + w_feas * F.huber_loss(margin_pred, margin)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1

        net.eval()
        with torch.no_grad():
            gf, glf, pi, legal, margin = val
            logits, margin_pred = net(gf, glf)
            val_kl = float(masked_policy_kl(logits, pi, legal))
            val_feas = float(F.huber_loss(margin_pred, margin))
            # Feasibility accuracy at the deploy threshold (margin >= 0).
            acc = float(((margin_pred >= 0) == (margin >= 0)).float().mean())
        val_loss = val_kl + w_feas * val_feas
        if val_loss < best_val:
            best_val = val_loss
            net.save(out_dir / "prior_best.pt")
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(
                f"epoch {epoch:4d}  train {tot / max(nb, 1):.4f}  "
                f"val kl {val_kl:.4f}  val feas {val_feas:.4f}  "
                f"feas acc {acc:.3f}  best {best_val:.4f}"
            )

    net.save(out_dir / "prior_last.pt")
    print(f"Saved prior_best.pt / prior_last.pt to {out_dir}")
    return {
        "n_samples": int(n),
        "best_val": float(best_val),
        "val_kl": float(val_kl),
        "val_feas": float(val_feas),
        "feas_acc": float(acc),
        "best_path": str(out_dir / "prior_best.pt"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, nargs="+", required=True,
                    help="shard directories (pass every iteration — replay mix)")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--w-feas", type=float, default=1.0,
                    help="weight of the clearance-margin loss")
    ap.add_argument("--margin-clip", type=float, default=2.0,
                    help="clip clearance-margin labels to ±this (m)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_prior(
        args.data, args.out_dir,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, hidden=args.hidden,
        w_feas=args.w_feas, margin_clip=args.margin_clip,
        val_frac=args.val_frac, seed=args.seed,
    )


if __name__ == "__main__":
    main()
