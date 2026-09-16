"""
Two-stage trainer for the split feasibility / group-scheduling prior —
sim-free, no A* teacher anywhere.

Data: ``.npz`` shards written by ``eval_gaz14_lazy --log-pi-targets`` *after*
the v2 feature hook landed (shards must carry ``feas_feats`` /
``policy_feats`` / ``global_feats_v2`` / ``episode_seed``; older shards
error out with a recollect hint).  Pass **all** iteration directories to
``--data`` — mixing shards across plan→distill iterations is the replay
guard against teacher distribution shift.

Stage A — :class:`FeasNet14`, BCE on the root shield-vet outcome ``safe``
(1 = the exact swept vet passed).  These labels are harvested for free at
every eagerly-vetted root; class imbalance (safe is usually the minority)
is handled with an automatic ``pos_weight``.  Model selection: balanced
accuracy.  Two baselines the net must beat are printed once: the majority
class, and the scalar proxy ``min ahead clearance − d_safe``.

Stage B — :class:`GroupPolicyNet14`, masked KL(π′ ‖ softmax(logits)) against
the search's improved root policy, with the *frozen* stage-A probabilities
appended as policy feature 12.  Freezing keeps the feasibility net honest to
its own labels (no CE gradient can bend the probability into a generic
logit), and lets either net be retrained alone.  Model selection: val KL.

Split discipline: train/val by **episode seed** — decisions within an
episode are heavily correlated, so a row split would leak.

Usage:
    PYTHONPATH=. python -m robot_nav.train_feas_policy14 \
        --data data/pi_targets_v2 [dir2 ...] --out-dir runs/gaz14_fp/iter0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from robot_nav.models.MARL.capswitcher_14.rl.search.feas_policy_net import (
    FeasNet14,
    GroupPolicyNet14,
)
from robot_nav.train_prior import masked_policy_kl

_V2_KEYS = ("feas_feats", "policy_feats", "global_feats_v2", "episode_seed")


def load_shards(dirs: list[str]) -> dict[str, np.ndarray]:
    paths = sorted(p for d in dirs for p in Path(d).rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no .npz shards under {dirs}")
    cols = (*_V2_KEYS, "pi_prime", "legal", "safe", "clearance")
    parts: dict[str, list] = {c: [] for c in cols}
    d_safe = None
    for p in paths:
        z = np.load(p, allow_pickle=False)
        missing = [c for c in _V2_KEYS if c not in z.files]
        if missing:
            raise SystemExit(
                f"{p} lacks {missing} — recollect with the updated "
                "eval_gaz14_lazy --log-pi-targets (v2 feature hook)")
        for c in cols:
            parts[c].append(z[c])
        d_safe = float(z["d_safe"])
    data = {c: np.concatenate(parts[c], axis=0) for c in cols}
    data["d_safe"] = d_safe
    n = data["pi_prime"].shape[0]
    if (data["episode_seed"] < 0).any():
        print("WARNING: some rows lack an episode seed (-1) — they form one "
              "shared split bucket")
    print(f"Loaded {n} decisions from {len(paths)} shards "
          f"({len(np.unique(data['episode_seed']))} episodes)")
    return data


def episode_split(seeds: np.ndarray, val_frac: float, seed: int):
    """Row masks (train, val) with whole episodes on one side."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(seeds)
    val_eps = rng.permutation(uniq)[: max(1, int(len(uniq) * val_frac))]
    val = np.isin(seeds, val_eps)
    return ~val, val


# ---------------------------------------------------------------------------
# Stage A — feasibility
# ---------------------------------------------------------------------------

def _feas_metrics(prob: np.ndarray, y: np.ndarray, tag: str) -> dict:
    pred = prob >= 0.5
    safe, unsafe = y, ~y
    out = {
        f"{tag}_acc": float((pred == y).mean()),
        f"{tag}_bal_acc": float(
            0.5 * (pred[safe].mean() if safe.any() else 1.0)
            + 0.5 * ((~pred[unsafe]).mean() if unsafe.any() else 1.0)),
        f"{tag}_unsafe_recall": float((~pred[unsafe]).mean()) if unsafe.any() else 1.0,
        f"{tag}_unsafe_precision": float(unsafe[~pred].mean()) if (~pred).any() else 1.0,
        f"{tag}_safe_recall": float(pred[safe].mean()) if safe.any() else 1.0,
    }
    return out


def train_feas(data, tr_rows, va_rows, args, device, out, log) -> FeasNet14:
    K = data["feas_feats"].shape[1]
    X = data["feas_feats"].reshape(-1, data["feas_feats"].shape[-1])
    y = data["safe"].reshape(-1)
    valid = np.isfinite(data["clearance"].reshape(-1))
    tr = np.repeat(tr_rows, K) & valid
    va = np.repeat(va_rows, K) & valid

    Xtr = torch.as_tensor(X[tr], dtype=torch.float32)
    ytr = torch.as_tensor(y[tr], dtype=torch.float32)
    Xva = torch.as_tensor(X[va], dtype=torch.float32, device=device)
    yva = y[va]
    n_pos, n_neg = int(y[tr].sum()), int((~y[tr]).sum())
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), device=device)
    print(f"[feas] {tr.sum()} train / {va.sum()} val rows, "
          f"frac safe {y[tr].mean():.3f}, pos_weight {float(pos_weight):.2f}")

    # Baselines printed once: majority class + the scalar clearance proxy
    # (min ahead clearance, feature 4, un-normalised, vs d_safe).
    cap = 2.0
    proxy = (X[va][:, 4] * cap - data["d_safe"]) >= 0.0
    base = {
        "majority_acc": float(max(yva.mean(), 1.0 - yva.mean())),
        **_feas_metrics(proxy.astype(np.float64), yva, "proxy"),
    }
    print(f"[feas] baselines: {json.dumps(base)}")
    log.write(json.dumps({"stage": "feas_baselines", **base}) + "\n")

    net = FeasNet14(hidden=args.feas_hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    best = -1.0
    for epoch in range(args.feas_epochs):
        net.train()
        order = torch.randperm(len(Xtr))
        tot = nb = 0
        for i in range(0, len(Xtr), args.batch):
            idx = order[i:i + args.batch]
            xb = Xtr[idx].to(device)
            yb = ytr[idx].to(device)
            loss = F.binary_cross_entropy_with_logits(
                net(xb[:, None, :]).squeeze(1), yb, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        net.eval()
        with torch.no_grad():
            prob = torch.sigmoid(
                net(Xva[:, None, :]).squeeze(1)).cpu().numpy()
        row = {"stage": "feas", "epoch": epoch,
               "train_bce": tot / max(nb, 1),
               **_feas_metrics(prob, yva, "val")}
        log.write(json.dumps(row) + "\n")
        log.flush()
        if epoch % 10 == 0 or epoch == args.feas_epochs - 1:
            print(f"[feas] ep {epoch:3d} | bce {row['train_bce']:.4f} | "
                  f"acc {row['val_acc']:.3f} bal {row['val_bal_acc']:.3f} "
                  f"unsafe R/P {row['val_unsafe_recall']:.3f}/"
                  f"{row['val_unsafe_precision']:.3f}")
        if row["val_bal_acc"] > best:
            best = row["val_bal_acc"]
            net.save(out / "feas_best.pt", extra=row)
    net.save(out / "feas_last.pt", extra=row)
    print(f"[feas] best val balanced acc {best:.3f}")
    return FeasNet14.load(out / "feas_best.pt", map_location=device)


# ---------------------------------------------------------------------------
# Stage B — policy
# ---------------------------------------------------------------------------

def train_policy(data, feas_net, tr_rows, va_rows, args, device, out, log):
    feas_net = feas_net.to(device).eval()
    with torch.no_grad():
        prob = torch.sigmoid(feas_net(torch.as_tensor(
            data["feas_feats"], dtype=torch.float32, device=device)))
    base = torch.as_tensor(data["policy_feats"], dtype=torch.float32)
    glob = torch.as_tensor(data["global_feats_v2"], dtype=torch.float32)
    pi = torch.as_tensor(data["pi_prime"], dtype=torch.float32)
    legal = torch.as_tensor(data["legal"], dtype=torch.bool)
    prob = prob.cpu()

    tr = np.flatnonzero(tr_rows)
    va_t = tuple(t[np.flatnonzero(va_rows)].to(device)
                 for t in (base, prob, glob, pi, legal))

    # Reference: KL of the uniform-over-legal prior (what phase 0 ran with).
    n_legal = legal[np.flatnonzero(va_rows)].sum(dim=1).clamp(min=1)
    uniform_kl = float(torch.log(n_legal.float()).mean())
    teacher = va_t[3].argmax(dim=1)
    print(f"[policy] {len(tr)} train / {len(teacher)} val decisions, "
          f"uniform-prior KL {uniform_kl:.3f}, "
          f"teacher precise frac {(teacher == pi.shape[1] - 1).float().mean():.3f}")

    net = GroupPolicyNet14(hidden=args.policy_hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    best = float("inf")
    for epoch in range(args.policy_epochs):
        net.train()
        order = np.random.permutation(len(tr))
        tot = nb = 0
        for i in range(0, len(tr), args.batch):
            idx = tr[order[i:i + args.batch]]
            b, p, g, pib, lb = (t[idx].to(device)
                                for t in (base, prob, glob, pi, legal))
            loss = masked_policy_kl(net(b, p, g), pib, lb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        net.eval()
        with torch.no_grad():
            b, p, g, piv, lv = va_t
            logits = net(b, p, g)
            val_kl = float(masked_policy_kl(logits, piv, lv))
            pred = logits.masked_fill(~lv, float("-inf")).argmax(dim=1)
            top1 = float((pred == teacher).float().mean())
            coarse = teacher < logits.shape[1] - 1
            top1_coarse = (float((pred[coarse] == teacher[coarse])
                                 .float().mean()) if coarse.any() else 1.0)
        row = {"stage": "policy", "epoch": epoch,
               "train_kl": tot / max(nb, 1), "val_kl": val_kl,
               "val_top1": top1, "val_top1_coarse": top1_coarse}
        log.write(json.dumps(row) + "\n")
        log.flush()
        if epoch % 10 == 0 or epoch == args.policy_epochs - 1:
            print(f"[policy] ep {epoch:3d} | KL {row['train_kl']:.4f} | val "
                  f"KL {val_kl:.4f} top1 {top1:.3f} "
                  f"(coarse {top1_coarse:.3f}, uniform {uniform_kl:.3f})")
        if val_kl < best:
            best = val_kl
            net.save(out / "policy_best.pt", extra=row)
    net.save(out / "policy_last.pt", extra=row)
    print(f"[policy] best val KL {best:.4f} (uniform {uniform_kl:.4f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, nargs="+", required=True,
                    help="shard directories (pass every iteration — replay mix)")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--feas-hidden", type=int, default=64)
    ap.add_argument("--policy-hidden", type=int, default=128)
    ap.add_argument("--feas-epochs", type=int, default=60)
    ap.add_argument("--policy-epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--feas-ckpt", type=str, default=None,
                    help="skip stage A; use this trained FeasNet14 instead")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device

    data = load_shards(args.data)
    tr_rows, va_rows = episode_split(
        data["episode_seed"], args.val_frac, args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(json.dumps(vars(args), indent=2))
    log = open(out / "log.jsonl", "a")

    if args.feas_ckpt:
        feas_net = FeasNet14.load(args.feas_ckpt, map_location=device)
        feas_net.save(out / "feas_best.pt")   # deploy dir stays self-contained
        print(f"[feas] reusing {args.feas_ckpt}")
    else:
        feas_net = train_feas(data, tr_rows, va_rows, args, device, out, log)

    train_policy(data, feas_net, tr_rows, va_rows, args, device, out, log)
    print(f"Saved feas_best.pt / policy_best.pt to {out} — deploy with "
          f"eval_gaz14_lazy --feas-policy-dir {out}")


if __name__ == "__main__":
    main()
