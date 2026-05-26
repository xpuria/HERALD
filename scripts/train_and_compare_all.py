"""
train and compare all models with a proper chronological split.

key correctness properties:
- temporal train/val/test split with embargo (no window straddles a boundary)
- vq-vae is assumed to have been fit on the train slice only (train_vqvae.py
  enforces this); the same checkpoint is loaded here for tokenization
- classifiers are trained on the train slice tokens and evaluated on the
  held-out test slice tokens
- recurrent state is reset between non-contiguous batch rows during evaluation
- labels marked IGNORE_LABEL (default -100) are excluded from loss and metrics
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

from src.data.splits import temporal_split
from src.data.zigzag import IGNORE_LABEL
from src.models.hrm import HierarchicalReasoningModel_ACTV1Config
from src.models.lora import add_lora_to_linear
from src.models.lstm_baseline import LSTMBaseline, LSTMConfig
from src.models.transformer_baseline import TransformerBaseline, TransformerConfig
from src.models.vq_vae import VQVAE
from src.models.working_hrm import WorkingHierarchicalReasoningModel as WorkingHRM
from src.utils import set_seed


TOKEN_SEQ_LEN = 16          # tokens per window emitted by the vq-vae
VQVAE_WINDOW = 64           # raw timesteps per vq-vae window
DOWNSAMPLE = VQVAE_WINDOW // TOKEN_SEQ_LEN


def load_series(csv_path: str, target_col: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' not found. available: {list(df.columns)}")
    return df[target_col].values.astype(np.float64)


def tokenize_range(
    vqvae: VQVAE,
    features: np.ndarray,
    labels: np.ndarray,
    start: int,
    stop: int,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    tokenize non-overlapping length-64 windows in [start, stop).
    returns flat token and label tensors aligned 1:1, with labels downsampled
    by taking the last raw label inside each 4-step block.
    """
    vqvae.eval()
    tokens_out = []
    labels_out = []

    # build all valid window-start indices then process in mini-batches
    starts = np.arange(start, stop - VQVAE_WINDOW + 1, VQVAE_WINDOW, dtype=np.int64)
    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            chunk = starts[i : i + batch_size]
            xs = np.stack([features[s : s + VQVAE_WINDOW] for s in chunk], axis=0)
            x = torch.tensor(xs, dtype=torch.float32, device=device).unsqueeze(1)  # (B, 1, 64)
            _, _, _, encoding_indices = vqvae(x)
            toks = encoding_indices.view(len(chunk), TOKEN_SEQ_LEN).cpu()
            ys = np.stack(
                [labels[s + DOWNSAMPLE - 1 : s + VQVAE_WINDOW : DOWNSAMPLE] for s in chunk],
                axis=0,
            )
            tokens_out.append(toks)
            labels_out.append(torch.tensor(ys, dtype=torch.long))

    if not tokens_out:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    tokens_flat = torch.cat(tokens_out, dim=0).view(-1)
    labels_flat = torch.cat(labels_out, dim=0).view(-1)
    return tokens_flat, labels_flat


def make_batches(tokens: torch.Tensor, labels: torch.Tensor, batch_size: int, seq_len: int):
    """
    chop a flat token stream into (batch_size, num_segments * seq_len). each
    batch row is a contiguous time slice — recurrent state may carry within a
    row but must be reset across rows.
    """
    total = tokens.shape[0]
    per_row = (total // batch_size) // seq_len * seq_len
    if per_row <= 0:
        raise ValueError(f"not enough tokens ({total}) for batch_size={batch_size}, seq_len={seq_len}")
    usable = batch_size * per_row
    tokens_b = tokens[:usable].reshape(batch_size, per_row)
    labels_b = labels[:usable].reshape(batch_size, per_row)
    return tokens_b, labels_b


def masked_xent(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=IGNORE_LABEL,
    )


def train_loop(model, tokens_b, labels_b, device, seq_len, num_epochs, lr, model_type, save_path, state_reset_every=20):
    """generic train loop. resets recurrent state every `state_reset_every` segments."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    _, total_steps = tokens_b.shape
    num_segments = total_steps // seq_len
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        valid = 0
        state = None
        hidden = None

        pbar = tqdm(range(num_segments), desc=f"{model_type} epoch {epoch+1}/{num_epochs}")
        for i in pbar:
            s, e = i * seq_len, (i + 1) * seq_len
            inputs = tokens_b[:, s:e].to(device)
            targets = labels_b[:, s:e].to(device)

            if i % state_reset_every == 0:
                state, hidden = None, None

            optimizer.zero_grad()
            if model_type == "lstm":
                outputs = model(inputs, hidden)
                hidden = outputs.get("hidden_state")
                if hidden is not None:
                    hidden = tuple(h.detach() for h in hidden)
            elif model_type in ("hrm_lora", "working_hrm"):
                outputs = model(inputs, state)
                state = outputs["state"]
                if state is not None:
                    state.z_H = state.z_H.detach()
                    state.z_L = state.z_L.detach()
            else:
                outputs = model(inputs)

            loss = masked_xent(outputs["logits"], targets)
            if torch.isnan(loss) or torch.isinf(loss):
                state, hidden = None, None
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            valid += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()

        if valid > 0:
            avg = epoch_loss / valid
            if avg < best_loss:
                best_loss = avg
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
    return model


def evaluate(model, tokens_b, labels_b, device, model_type, seq_len, state_reset_every=20):
    model.eval()
    _, total_steps = tokens_b.shape
    num_segments = total_steps // seq_len

    state = None
    hidden = None
    total_loss = 0.0
    valid = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for i in tqdm(range(num_segments), desc=f"eval {model_type}"):
            s, e = i * seq_len, (i + 1) * seq_len
            inputs = tokens_b[:, s:e].to(device)
            targets = labels_b[:, s:e].to(device)

            if i % state_reset_every == 0:
                state, hidden = None, None

            if model_type == "lstm":
                outputs = model(inputs, hidden)
                hidden = outputs.get("hidden_state")
            elif model_type in ("hrm_lora", "working_hrm"):
                outputs = model(inputs, state)
                state = outputs["state"]
            else:
                outputs = model(inputs)

            logits = outputs["logits"]
            loss = masked_xent(logits, targets)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            total_loss += loss.item()
            valid += 1

            logits_flat = logits.reshape(-1, 2)
            labels_flat = targets.reshape(-1)
            mask = labels_flat != IGNORE_LABEL
            preds = torch.argmax(logits_flat, dim=-1)
            all_preds.append(preds[mask].cpu().numpy())
            all_labels.append(labels_flat[mask].cpu().numpy())

    preds = np.concatenate(all_preds) if all_preds else np.empty(0)
    labels = np.concatenate(all_labels) if all_labels else np.empty(0)

    if labels.size == 0:
        return {'loss': float('inf'), 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'mcc': 0, 'n': 0}

    accuracy = float(np.mean(preds == labels))
    tp = float(np.sum((preds == 1) & (labels == 1)))
    fp = float(np.sum((preds == 1) & (labels == 0)))
    tn = float(np.sum((preds == 0) & (labels == 0)))
    fn = float(np.sum((preds == 0) & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    denom = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

    return {
        'loss': total_loss / max(valid, 1),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'n': int(labels.size),
    }


def build_hrm_config(seq_len: int) -> HierarchicalReasoningModel_ACTV1Config:
    return HierarchicalReasoningModel_ACTV1Config(
        batch_size=8, seq_len=seq_len, vocab_size=512,
        H_cycles=2, L_cycles=4, H_layers=2, L_layers=2,
        hidden_size=128, expansion=4.0, num_heads=4,
        halt_max_steps=1, halt_exploration_prob=0.0, forward_dtype='float32',
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset/long_short_ratio.csv")
    parser.add_argument("--target_col", type=str, default="long_short_ratio")
    parser.add_argument("--vqvae_checkpoint", type=str, default="checkpoints_vq/vqvae_epoch_20.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--preprocessing_window", type=int, default=30)
    parser.add_argument("--zigzag_deviation", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    if not os.path.exists(args.vqvae_checkpoint):
        raise FileNotFoundError(
            f"vq-vae checkpoint not found at {args.vqvae_checkpoint}. "
            f"train it first with scripts/train_vqvae.py (which now fits on train slice only)."
        )

    vqvae = VQVAE(
        num_hiddens=64, num_residual_layers=2, num_residual_hiddens=32,
        num_embeddings=512, embedding_dim=64, commitment_cost=0.25,
    ).to(device)
    vqvae.load_state_dict(torch.load(args.vqvae_checkpoint, map_location=device))
    vqvae.eval()

    # load raw series, compute causal features and causal labels
    from src.data.preprocessing import preprocess_pipeline
    from src.data.zigzag import zigzag
    prices = load_series(args.data_path, args.target_col)
    features = preprocess_pipeline(prices, window=args.preprocessing_window)
    _, labels = zigzag(prices, deviation_pct=args.zigzag_deviation)

    n = len(features)
    embargo = args.preprocessing_window + VQVAE_WINDOW
    split = temporal_split(n, train_frac=args.train_frac, val_frac=args.val_frac, embargo=embargo)
    print(f"split: train=[{split.train.start},{split.train.stop}) "
          f"val=[{split.val.start},{split.val.stop}) "
          f"test=[{split.test.start},{split.test.stop}) embargo={embargo}")

    # tokenize each split independently
    tr_tokens, tr_labels = tokenize_range(vqvae, features, labels, split.train.start, split.train.stop, device)
    te_tokens, te_labels = tokenize_range(vqvae, features, labels, split.test.start, split.test.stop, device)
    print(f"train tokens: {tr_tokens.shape[0]}, test tokens: {te_tokens.shape[0]}")
    print(f"train label balance: {(tr_labels[tr_labels != IGNORE_LABEL] == 1).float().mean():.3f}")
    print(f"test  label balance: {(te_labels[te_labels != IGNORE_LABEL] == 1).float().mean():.3f}")
    print(f"train ignore frac:   {(tr_labels == IGNORE_LABEL).float().mean():.3f}")
    print(f"test  ignore frac:   {(te_labels == IGNORE_LABEL).float().mean():.3f}")

    tr_b = make_batches(tr_tokens, tr_labels, args.batch_size, TOKEN_SEQ_LEN)
    te_b = make_batches(te_tokens, te_labels, args.batch_size, TOKEN_SEQ_LEN)

    ckpt_dir = "checkpoints_comparison"
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- lstm ----
    lstm = LSTMBaseline(LSTMConfig(
        vocab_size=512, hidden_size=128, num_layers=4, dropout=0.1, seq_len=TOKEN_SEQ_LEN
    )).to(device)
    lstm = train_loop(lstm, *tr_b, device, TOKEN_SEQ_LEN, args.epochs, 1e-4, "lstm",
                      os.path.join(ckpt_dir, "best_lstm.pt"))

    # ---- transformer ----
    tfm = TransformerBaseline(TransformerConfig(
        vocab_size=512, hidden_size=128, num_layers=4, num_heads=4,
        ff_dim=512, dropout=0.1, seq_len=TOKEN_SEQ_LEN,
    )).to(device)
    tfm = train_loop(tfm, *tr_b, device, TOKEN_SEQ_LEN, args.epochs, 5e-4, "transformer",
                     os.path.join(ckpt_dir, "best_transformer.pt"))

    # ---- working hrm + lora ----
    hrm_cfg = build_hrm_config(TOKEN_SEQ_LEN)
    base = WorkingHRM(hrm_cfg)
    hrm_lora = add_lora_to_linear(
        base, rank=8, alpha=16.0, dropout=0.1, freeze_base=False,
        target_modules=['qkv_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    ).to(device)
    hrm_lora = train_loop(hrm_lora, *tr_b, device, TOKEN_SEQ_LEN, args.epochs, 1e-4, "hrm_lora",
                          os.path.join(ckpt_dir, "best_hrm_lora.pt"))

    # ---- working hrm (std) ----
    working_hrm = WorkingHRM(hrm_cfg).to(device)
    working_hrm = train_loop(working_hrm, *tr_b, device, TOKEN_SEQ_LEN, args.epochs, 1e-4, "working_hrm",
                             os.path.join(ckpt_dir, "best_working_hrm.pt"))

    # ---- evaluate on held-out test ----
    print("\nevaluating on held-out test split")
    results = {
        'LSTM': evaluate(lstm, *te_b, device, "lstm", TOKEN_SEQ_LEN),
        'Transformer': evaluate(tfm, *te_b, device, "transformer", TOKEN_SEQ_LEN),
        'Fin-HRM (LoRA)': evaluate(hrm_lora, *te_b, device, "hrm_lora", TOKEN_SEQ_LEN),
        'Fin-HRM (Std)': evaluate(working_hrm, *te_b, device, "working_hrm", TOKEN_SEQ_LEN),
    }
    params = {
        'LSTM': sum(p.numel() for p in lstm.parameters()),
        'Transformer': sum(p.numel() for p in tfm.parameters()),
        'Fin-HRM (LoRA)': sum(p.numel() for p in hrm_lora.parameters()),
        'Fin-HRM (Std)': sum(p.numel() for p in working_hrm.parameters()),
    }

    rows = []
    for name, r in results.items():
        rows.append({
            'Model': name,
            'Params': f"{params[name]:,}",
            'TestN': r['n'],
            'Loss': f"{r['loss']:.4f}",
            'Accuracy': f"{r['accuracy']:.4f}",
            'Precision': f"{r['precision']:.4f}",
            'Recall': f"{r['recall']:.4f}",
            'F1': f"{r['f1']:.4f}",
            'MCC': f"{r['mcc']:.4f}",
        })
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/model_comparison.csv", index=False)
    print("\nresults saved to results/model_comparison.csv")


if __name__ == "__main__":
    main()
