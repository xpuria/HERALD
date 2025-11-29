"""
train and compare all models

this script trains and evaluates four models on financial time series data:
1. lstm baseline
2. transformer baseline
3. fin-hrm (standard)
4. fin-hrm (lora)

it generates a unified comparison table with metrics.
"""

import argparse
import torch
import torch.nn.functional as F
import os
import sys
sys.path.append('.')

import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import pandas as pd

# import models
from src.models.lstm_baseline import LSTMBaseline, LSTMConfig
from src.models.transformer_baseline import TransformerBaseline, TransformerConfig
from src.models.working_hrm import WorkingHierarchicalReasoningModel as WorkingHRM
from src.models.hrm import HierarchicalReasoningModel_ACTV1Config
from src.models.lora import add_lora_to_linear, count_lora_parameters
from src.models.vq_vae import VQVAE
from src.data.dataset import HRMDataset
from src.utils import set_seed, setup_logging


def train_lstm(tokens, labels, config, device, num_epochs=15):
    """train lstm baseline"""
    print("\n" + "="*70)
    print("training lstm baseline")
    print("="*70)
    
    save_path = 'checkpoints_comparison/best_lstm.pt'
    lstm_config = LSTMConfig(
        vocab_size=512,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=0.1,
        seq_len=config['seq_len']
    )
    model = LSTMBaseline(lstm_config).to(device)
    
    # skip if exists
    if os.path.exists(save_path):
        print(f"loading existing lstm model from {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))
        return model
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    batch_size, total_steps = tokens.shape
    seq_len = config['seq_len']
    num_segments = total_steps // seq_len
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0
        hidden_state = None
        
        pbar = tqdm(range(num_segments), desc=f"lstm epoch {epoch+1}/{num_epochs}")
        
        for i in pbar:
            start_idx = i * seq_len
            end_idx = start_idx + seq_len
            
            input_tokens = tokens[:, start_idx:end_idx].to(device)
            target_labels = labels[:, start_idx:end_idx].to(device)
            
            if i % 20 == 0:
                hidden_state = None
            
            try:
                optimizer.zero_grad()
                outputs = model(input_tokens, hidden_state)
                hidden_state = outputs.get("hidden_state")
                
                if hidden_state is not None:
                    hidden_state = tuple(h.detach() for h in hidden_state)
                
                logits = outputs["logits"]
                logits_flat = logits.reshape(-1, 2)
                labels_flat = target_labels.reshape(-1)
                
                loss = F.cross_entropy(logits_flat, labels_flat, reduction='mean')
                
                if torch.isnan(loss) or torch.isinf(loss):
                    hidden_state = None
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                epoch_loss += loss.item()
                valid_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                hidden_state = None
                continue
        
        scheduler.step()
        
        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            if avg_loss < best_loss:
                best_loss = avg_loss
                os.makedirs("checkpoints_comparison", exist_ok=True)
                torch.save(model.state_dict(), save_path)
    
    # load best model before returning
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
    return model


def train_transformer(tokens, labels, config, device, num_epochs=15):
    """train transformer baseline"""
    print("\n" + "="*70)
    print("training transformer baseline")
    print("="*70)
    
    save_path = 'checkpoints_comparison/best_transformer.pt'
    transformer_config = TransformerConfig(
        vocab_size=512,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_heads=4,
        ff_dim=config['hidden_size'] * 4,
        dropout=0.1,
        seq_len=config['seq_len']
    )
    model = TransformerBaseline(transformer_config).to(device)
    
    # skip if exists
    if os.path.exists(save_path):
        print(f"loading existing transformer model from {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))
        return model
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    batch_size, total_steps = tokens.shape
    seq_len = config['seq_len']
    num_segments = total_steps // seq_len
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0
        
        pbar = tqdm(range(num_segments), desc=f"transformer epoch {epoch+1}/{num_epochs}")
        
        for i in pbar:
            start_idx = i * seq_len
            end_idx = start_idx + seq_len
            
            input_tokens = tokens[:, start_idx:end_idx].to(device)
            target_labels = labels[:, start_idx:end_idx].to(device)
            
            try:
                optimizer.zero_grad()
                outputs = model(input_tokens)
                
                logits = outputs["logits"]
                logits_flat = logits.reshape(-1, 2)
                labels_flat = target_labels.reshape(-1)
                
                loss = F.cross_entropy(logits_flat, labels_flat, reduction='mean')
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                valid_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                continue
        
        scheduler.step()
        
        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            if avg_loss < best_loss:
                best_loss = avg_loss
                os.makedirs("checkpoints_comparison", exist_ok=True)
                torch.save(model.state_dict(), save_path)
    
    # load best model before returning
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
    return model


def train_hrm_lora(tokens, labels, config, device, num_epochs=15):
    """train working fin-hrm with lora using standard loop"""
    print("\n" + "="*70)
    print("training fin-hrm with lora (stable)")
    print("="*70)
    
    save_path = 'checkpoints_comparison/best_hrm_lora.pt'
    
    # create hrm model
    hrm_config = {
        'batch_size': 8,
        'seq_len': config['seq_len'],
        'vocab_size': 512,
        'H_cycles': 2,
        'L_cycles': 4,
        'H_layers': 2,
        'L_layers': 2,
        'hidden_size': 128,
        'expansion': 4.0,
        'num_heads': 4,
        'halt_max_steps': 1,
        'halt_exploration_prob': 0.0,
        'forward_dtype': 'float32'
    }
    
    # initialize base model
    # note: workinghrm matches the simple interface
    working_hrm_config = HierarchicalReasoningModel_ACTV1Config(**hrm_config)
    base_model = WorkingHRM(working_hrm_config)
    
    # initialize with reasonable weights (workinghrm does this in init, but just to be safe/consistent)
    # we are training from scratch here as we don't have a pretrained base for this config in this script context
    # if we wanted to finetune, we'd load state dict here.
    # base_model.load_state_dict(...) 
    
    # add lora adapters to linear layers
    # workinghrm uses nn.linear, so this works directly
    model = add_lora_to_linear(
        base_model,
        rank=8,
        alpha=16.0,
        dropout=0.1,
        freeze_base=False,  # training everything + lora parameters for this comparison
        target_modules=['qkv_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'] # target transformer layers
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    batch_size, total_steps = tokens.shape
    seq_len = config['seq_len']
    num_segments = total_steps // seq_len
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0
        state = None
        
        pbar = tqdm(range(num_segments), desc=f"lora-hrm epoch {epoch+1}/{num_epochs}")
        
        for i in pbar:
            start_idx = i * seq_len
            end_idx = start_idx + seq_len
            
            input_tokens = tokens[:, start_idx:end_idx].to(device)
            target_labels = labels[:, start_idx:end_idx].to(device)
            
            if i % 20 == 0:
                state = None
            
            try:
                optimizer.zero_grad()
                outputs = model(input_tokens, state)
                state = outputs["state"]
                
                # detach state to prevent backprop through entire history
                if state is not None:
                    state.z_H = state.z_H.detach()
                    state.z_L = state.z_L.detach()
                
                logits = outputs["logits"]
                logits_flat = logits.reshape(-1, 2)
                labels_flat = target_labels.reshape(-1)
                
                # standard classification loss
                loss = F.cross_entropy(logits_flat, labels_flat, reduction='mean')
                
                if torch.isnan(loss) or torch.isinf(loss) or loss > 100:
                    state = None
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                epoch_loss += loss.item()
                valid_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                state = None
                continue
        
        scheduler.step()
        
        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            if avg_loss < best_loss:
                best_loss = avg_loss
                os.makedirs("checkpoints_comparison", exist_ok=True)
                torch.save(model.state_dict(), save_path)
                
    # load best model before returning
    if os.path.exists(save_path):
        print(f"loading best lora model with loss {best_loss:.4f}")
        model.load_state_dict(torch.load(save_path, map_location=device))
    else:
        print("warning: lora training failed to save any checkpoint. returning last model.")
        
    return model


def evaluate_model(model, tokens, labels, device, model_type="lstm", seq_len=16):
    """evaluate a model and return comprehensive metrics"""
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    # handle 2d batched tokens: (batch_size, total_len)
    if len(tokens.shape) == 2:
        batch_size, total_steps = tokens.shape
        num_segments = total_steps // seq_len
    else:
        # fallback for 1d tokens (not used with corrected logic)
        batch_size = 1
        total_steps = tokens.shape[0]
        num_segments = total_steps // seq_len
        tokens = tokens.unsqueeze(0)
        labels = labels.unsqueeze(0)
    
    carry = None
    hidden_state = None
    valid_segments = 0
    
    with torch.no_grad():
        for i in tqdm(range(num_segments), desc=f"evaluating {model_type.upper()}"):
            start_idx = i * seq_len
            end_idx = start_idx + seq_len
            
            input_tokens = tokens[:, start_idx:end_idx].to(device)
            target_labels = labels[:, start_idx:end_idx].to(device)
            
            try:
                if model_type == "hrm_lora" or model_type == "working_hrm":
                    # standard hrm interface
                    if hidden_state is None or i % 20 == 0: 
                        hidden_state = None 
                    outputs = model(input_tokens, hidden_state)
                    hidden_state = outputs["state"]
                elif model_type == "lstm":
                    if i % 20 == 0:
                        hidden_state = None
                    outputs = model(input_tokens, hidden_state)
                    hidden_state = outputs.get("hidden_state")
                else:
                    outputs = model(input_tokens)
                
                logits = outputs["logits"]
                # flatten batch and seq dims
                logits_flat = logits.reshape(-1, 2)
                labels_flat = target_labels.reshape(-1)
                
                loss = F.cross_entropy(logits_flat, labels_flat, reduction='mean')
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    preds = torch.argmax(logits_flat, dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels_flat.cpu().numpy())
                    valid_segments += 1
                    
            except Exception as e:
                carry = None
                hidden_state = None
                continue
    
    # metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = np.mean(all_preds == all_labels)
    avg_loss = total_loss / valid_segments if valid_segments > 0 else float('inf')
    
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    tn = np.sum((all_preds == 0) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset/long_short_ratio.csv")
    parser.add_argument("--vqvae_checkpoint", type=str, default="checkpoints_vq/vqvae_epoch_20.pt")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}\n")
    
    # 1. load vq-vae
    print("loading vq-vae...")
    if not os.path.exists(args.vqvae_checkpoint):
        print(f"error: vq-vae checkpoint not found at {args.vqvae_checkpoint}")
        print("please train vq-vae first: python scripts/train_vqvae.py ...")
        return

    vqvae = VQVAE(
        num_hiddens=64, num_residual_layers=2, num_residual_hiddens=32,
        num_embeddings=512, embedding_dim=64, commitment_cost=0.25
    ).to(device)
    vqvae.load_state_dict(torch.load(args.vqvae_checkpoint, map_location=device))
    vqvae.eval()
    
    # 2. prepare data
    print("preparing data...")
    dataset = HRMDataset(
        csv_path=args.data_path,
        vqvae_model=vqvae,
        seq_len=64,
        device=str(device),
        target_col="long_short_ratio"
    )
    
    tokens, labels = dataset.get_all_tokens_and_labels()
    print(f"total tokens: {tokens.shape[0]}\n")
    
    # batching
    batch_size = 8
    stream_len = tokens.shape[0] // batch_size
    tokens_batched = tokens[:batch_size * stream_len].reshape(batch_size, stream_len)
    labels_batched = labels[:batch_size * stream_len].reshape(batch_size, stream_len)
    
    config = {'hidden_size': 128, 'num_layers': 4, 'seq_len': 16}
    
    # 3. train models
    # lstm
    lstm_model = train_lstm(tokens_batched, labels_batched, config, device, args.epochs)
    
    # transformer
    transformer_model = train_transformer(tokens_batched, labels_batched, config, device, args.epochs)
    
    # fin-hrm (lora)
    hrm_lora_model = train_hrm_lora(tokens_batched, labels_batched, config, device, args.epochs)
    
    # fin-hrm (standard) - load pre-trained if available, else skip or simple init
    standard_hrm_path = "checkpoints_working_hrm/best_working_hrm.pt"
    working_hrm = None
    if os.path.exists(standard_hrm_path):
        print("\nloading pre-trained standard fin-hrm...")
        hrm_config_simple = {
            'batch_size': 8, 'seq_len': 16, 'vocab_size': 512,
            'H_cycles': 2, 'L_cycles': 4, 'H_layers': 2, 'L_layers': 2,
            'hidden_size': 128, 'expansion': 4.0, 'num_heads': 4,
            'halt_max_steps': 1, 'halt_exploration_prob': 0.0, 'forward_dtype': 'float32'
        }
        # note: workinghrm matches the simple interface
        working_hrm_config = HierarchicalReasoningModel_ACTV1Config(**hrm_config_simple)
        working_hrm = WorkingHRM(working_hrm_config).to(device)
        working_hrm.load_state_dict(torch.load(standard_hrm_path, map_location=device))
    
    # 4. evaluate all
    print("\n" + "="*70)
    print("EVALUATING ALL MODELS")
    print("="*70)
    
    results = {}
    
    print("\nevaluating lstm...")
    results['LSTM'] = evaluate_model(lstm_model, tokens_batched, labels_batched, device, "lstm", 16)
    results['LSTM']['params'] = sum(p.numel() for p in lstm_model.parameters())
    
    print("\nevaluating transformer...")
    results['Transformer'] = evaluate_model(transformer_model, tokens_batched, labels_batched, device, "transformer", 16)
    results['Transformer']['params'] = sum(p.numel() for p in transformer_model.parameters())
    
    print("\nevaluating fin-hrm (lora)...")
    results['Fin-HRM (LoRA)'] = evaluate_model(hrm_lora_model, tokens_batched, labels_batched, device, "hrm_lora", 16)
    results['Fin-HRM (LoRA)']['params'] = sum(p.numel() for p in hrm_lora_model.parameters())
    
    if working_hrm:
        print("\nevaluating fin-hrm (standard)...")
        results['Fin-HRM (Std)'] = evaluate_model(working_hrm, tokens_batched, labels_batched, device, "working_hrm", 16)
        results['Fin-HRM (Std)']['params'] = sum(p.numel() for p in working_hrm.parameters())
    
    # 5. generate table
    print("\n" + "="*70)
    print("MODEL COMPARISON TABLE")
    print("="*70 + "\n")
    
    df_data = []
    for model_name in results:
        r = results[model_name]
        df_data.append({
            'Model': model_name,
            'Params': f"{r['params']:,}",
            'Loss': f"{r['loss']:.4f}",
            'Accuracy': f"{r['accuracy']:.4f}",
            'Precision': f"{r['precision']:.4f}",
            'Recall': f"{r['recall']:.4f}",
            'F1': f"{r['f1']:.4f}",
            'MCC': f"{r['mcc']:.4f}"
        })
    
    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    
    # save
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/model_comparison.csv", index=False)
    
    print(f"\n✓ results saved to results/model_comparison.csv")


if __name__ == "__main__":
    main()
