"""
train multifeature-hrm: patch-hrm with engineered multi-channel inputs.

channels: z-scored log return, z-scored log realized vol, sin/cos minute-of-day,
two z-scored momenta (5-bar, 20-bar). all causal.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append('.')

from src.data.features import build_multifeature
from src.data.splits import temporal_split
from src.data.zigzag import IGNORE_LABEL, zigzag
from src.models.multifeature_hrm import MultiFeatureHRM, MultiFeatureHRMConfig
from src.utils import set_seed


RAW_SEQ_LEN = 64
PATCH_SIZE = 4


def build_windows(features: np.ndarray, labels: np.ndarray, start: int, stop: int):
    starts = np.arange(start, stop - RAW_SEQ_LEN + 1, RAW_SEQ_LEN, dtype=np.int64)
    xs = np.stack([features[s : s + RAW_SEQ_LEN] for s in starts], axis=0).astype(np.float32)
    ys = np.stack(
        [labels[s + PATCH_SIZE - 1 : s + RAW_SEQ_LEN : PATCH_SIZE] for s in starts],
        axis=0,
    ).astype(np.int64)
    return torch.from_numpy(xs), torch.from_numpy(ys)


def iter_batches(x, y, batch_size, shuffle, seed):
    n = x.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in range(0, n, batch_size):
        sel = idx[i:i + batch_size]
        yield x[sel], y[sel]


def evaluate(model, x, y, device, batch_size):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, valid = 0.0, 0
    with torch.no_grad():
        for xb, yb in iter_batches(x, y, batch_size, shuffle=False, seed=0):
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            logits = out["logits"]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), yb.reshape(-1), ignore_index=IGNORE_LABEL
            )
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            total_loss += loss.item()
            valid += 1
            lf = logits.reshape(-1, logits.shape[-1])
            yf = yb.reshape(-1)
            mask = yf != IGNORE_LABEL
            preds = torch.argmax(lf, dim=-1)
            all_preds.append(preds[mask].cpu().numpy())
            all_labels.append(yf[mask].cpu().numpy())

    preds = np.concatenate(all_preds) if all_preds else np.empty(0)
    labels = np.concatenate(all_labels) if all_labels else np.empty(0)
    if labels.size == 0:
        return {'loss': float('inf'), 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'mcc': 0, 'n': 0}

    acc = float(np.mean(preds == labels))
    tp = float(np.sum((preds == 1) & (labels == 1)))
    fp = float(np.sum((preds == 1) & (labels == 0)))
    tn = float(np.sum((preds == 0) & (labels == 0)))
    fn = float(np.sum((preds == 0) & (labels == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    denom = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
    return {'loss': total_loss / max(valid, 1), 'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1, 'mcc': mcc, 'n': int(labels.size)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="dataset/long_short_ratio.csv")
    p.add_argument("--target_col", type=str, default="long_short_ratio")
    p.add_argument("--timestamp_col", type=str, default="timestamp")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--z_window", type=int, default=30)
    p.add_argument("--zigzag_deviation", type=float, default=0.01)
    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_csv", type=str, default="results/multifeature_hrm.csv")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    df = pd.read_csv(args.data_path)
    prices = df[args.target_col].values.astype(np.float64)
    ts = df[args.timestamp_col].values.astype(np.int64)

    features = build_multifeature(prices, ts, z_window=args.z_window)
    _, labels = zigzag(prices, deviation_pct=args.zigzag_deviation)
    n, C = features.shape
    print(f"features: {n} timesteps x {C} channels")

    embargo = args.z_window + RAW_SEQ_LEN
    split = temporal_split(n, args.train_frac, args.val_frac, embargo)
    print(f"split: train=[{split.train.start},{split.train.stop}) "
          f"test=[{split.test.start},{split.test.stop}) embargo={embargo}")

    x_tr, y_tr = build_windows(features, labels, split.train.start, split.train.stop)
    x_te, y_te = build_windows(features, labels, split.test.start, split.test.stop)
    print(f"train windows: {x_tr.shape[0]}, test windows: {x_te.shape[0]}")
    print(f"train label balance: {(y_tr[y_tr != IGNORE_LABEL] == 1).float().mean():.3f}")
    print(f"test  label balance: {(y_te[y_te != IGNORE_LABEL] == 1).float().mean():.3f}")

    cfg = MultiFeatureHRMConfig(
        raw_seq_len=RAW_SEQ_LEN, patch_size=PATCH_SIZE, num_features=C,
        hidden_size=128, num_heads=4, expansion=4.0,
        H_cycles=2, L_cycles=4, H_layers=2, L_layers=2,
    )
    model = MultiFeatureHRM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"multifeature-hrm params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_loss = float('inf')
    ckpt_path = "checkpoints_multifeature/best_multifeature_hrm.pt"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for ep in range(args.epochs):
        model.train()
        ep_loss, valid = 0.0, 0
        pbar = tqdm(iter_batches(x_tr, y_tr, args.batch_size, shuffle=True, seed=args.seed + ep),
                    total=(x_tr.shape[0] + args.batch_size - 1) // args.batch_size,
                    desc=f"epoch {ep+1}/{args.epochs}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            logits = out["logits"]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), yb.reshape(-1), ignore_index=IGNORE_LABEL
            )
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            valid += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        sched.step()
        avg = ep_loss / max(valid, 1)
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    m = evaluate(model, x_te, y_te, device, args.batch_size)
    print("\nheld-out test:", m)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    pd.DataFrame([{
        'Model': 'MultiFeature-HRM', 'Params': f"{n_params:,}", 'TestN': m['n'],
        'Loss': f"{m['loss']:.4f}", 'Accuracy': f"{m['accuracy']:.4f}",
        'Precision': f"{m['precision']:.4f}", 'Recall': f"{m['recall']:.4f}",
        'F1': f"{m['f1']:.4f}", 'MCC': f"{m['mcc']:.4f}",
    }]).to_csv(args.out_csv, index=False)
    print(f"saved to {args.out_csv}")


if __name__ == "__main__":
    main()
