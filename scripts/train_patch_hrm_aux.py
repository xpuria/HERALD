"""
train patch-hrm with vol-aux head + label smoothing.

main loss: label-smoothed cross-entropy for UP/DOWN direction.
aux loss:  huber loss on forward log realized volatility (regularizer).
total = ce_smooth + aux_weight * huber(vol_pred, vol_target).
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

from src.data.features import realized_vol
from src.data.preprocessing import calculate_log_returns, preprocess_pipeline
from src.data.splits import temporal_split
from src.data.zigzag import IGNORE_LABEL, zigzag
from src.models.patch_hrm_aux import PatchHRMAux, PatchHRMAuxConfig
from src.utils import set_seed


RAW_SEQ_LEN = 64
PATCH_SIZE = 4
NUM_PATCHES = RAW_SEQ_LEN // PATCH_SIZE


def build_windows(features, labels, vol_target, start: int, stop: int):
    """returns (raw_window, dir_label_per_patch, vol_target_per_patch)."""
    starts = np.arange(start, stop - RAW_SEQ_LEN + 1, RAW_SEQ_LEN, dtype=np.int64)
    xs = np.stack([features[s : s + RAW_SEQ_LEN] for s in starts], axis=0).astype(np.float32)
    ys = np.stack(
        [labels[s + PATCH_SIZE - 1 : s + RAW_SEQ_LEN : PATCH_SIZE] for s in starts],
        axis=0,
    ).astype(np.int64)
    # per-patch vol target uses the patch_size raw bars FOLLOWING each patch's end.
    vs = np.stack(
        [vol_target[s + PATCH_SIZE - 1 : s + RAW_SEQ_LEN : PATCH_SIZE] for s in starts],
        axis=0,
    ).astype(np.float32)
    return torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(vs)


def iter_batches(*tensors, batch_size, shuffle, seed):
    n = tensors[0].shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in range(0, n, batch_size):
        sel = idx[i:i + batch_size]
        yield tuple(t[sel] for t in tensors)


def evaluate(model, x, y, device, batch_size):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, valid = 0.0, 0
    with torch.no_grad():
        for xb, yb in iter_batches(x, y, batch_size=batch_size, shuffle=False, seed=0):
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
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--preprocessing_window", type=int, default=30)
    p.add_argument("--vol_window", type=int, default=8)
    p.add_argument("--aux_weight", type=float, default=0.3)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--zigzag_deviation", type=float, default=0.01)
    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_csv", type=str, default="results/patch_hrm_aux.csv")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    prices = pd.read_csv(args.data_path)[args.target_col].values.astype(np.float64)
    features = preprocess_pipeline(prices, window=args.preprocessing_window)
    _, labels = zigzag(prices, deviation_pct=args.zigzag_deviation)
    rets = calculate_log_returns(prices)
    vol_target_all = realized_vol(rets, window=args.vol_window)
    # standardize the vol target globally (using train-only stats — computed below)

    n = len(features)
    embargo = args.preprocessing_window + RAW_SEQ_LEN
    split = temporal_split(n, args.train_frac, args.val_frac, embargo)
    print(f"split: train=[{split.train.start},{split.train.stop}) "
          f"test=[{split.test.start},{split.test.stop}) embargo={embargo}")

    # train-only standardization of vol target
    tr_slice = vol_target_all[split.train.start : split.train.stop]
    vt_mu, vt_sd = float(np.mean(tr_slice)), float(np.std(tr_slice) + 1e-8)
    vol_target_std = (vol_target_all - vt_mu) / vt_sd
    print(f"vol_target stats (train): mu={vt_mu:.3f} sd={vt_sd:.3f}")

    x_tr, y_tr, v_tr = build_windows(features, labels, vol_target_std, split.train.start, split.train.stop)
    x_te, y_te, _ = build_windows(features, labels, vol_target_std, split.test.start, split.test.stop)
    print(f"train windows: {x_tr.shape[0]}, test windows: {x_te.shape[0]}")

    cfg = PatchHRMAuxConfig(
        raw_seq_len=RAW_SEQ_LEN, patch_size=PATCH_SIZE,
        hidden_size=128, num_heads=4, expansion=4.0,
        H_cycles=2, L_cycles=4, H_layers=2, L_layers=2,
    )
    model = PatchHRMAux(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"patch-hrm-aux params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_loss = float('inf')
    ckpt_path = "checkpoints_aux/best_patch_hrm_aux.pt"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for ep in range(args.epochs):
        model.train()
        ep_loss, ep_aux, valid = 0.0, 0.0, 0
        pbar = tqdm(iter_batches(x_tr, y_tr, v_tr, batch_size=args.batch_size, shuffle=True,
                                 seed=args.seed + ep),
                    total=(x_tr.shape[0] + args.batch_size - 1) // args.batch_size,
                    desc=f"epoch {ep+1}/{args.epochs}")
        for xb, yb, vb in pbar:
            xb, yb, vb = xb.to(device), yb.to(device), vb.to(device)
            opt.zero_grad()
            out = model(xb)
            logits = out["logits"]
            vol_pred = out["vol_pred"]

            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                yb.reshape(-1),
                ignore_index=IGNORE_LABEL,
                label_smoothing=args.label_smoothing,
            )
            aux = F.huber_loss(vol_pred, vb)
            loss = ce + args.aux_weight * aux
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_loss += ce.item()
            ep_aux += aux.item()
            valid += 1
            pbar.set_postfix({'ce': f'{ce.item():.4f}', 'aux': f'{aux.item():.4f}'})
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
        'Model': 'PatchHRM-Aux', 'Params': f"{n_params:,}", 'TestN': m['n'],
        'Loss': f"{m['loss']:.4f}", 'Accuracy': f"{m['accuracy']:.4f}",
        'Precision': f"{m['precision']:.4f}", 'Recall': f"{m['recall']:.4f}",
        'F1': f"{m['f1']:.4f}", 'MCC': f"{m['mcc']:.4f}",
    }]).to_csv(args.out_csv, index=False)
    print(f"saved to {args.out_csv}")


if __name__ == "__main__":
    main()
