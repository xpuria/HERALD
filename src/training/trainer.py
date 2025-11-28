import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from typing import Optional

from src.models.hrm import HierarchicalReasoningModel_ACTV1, ACTLossHead, HierarchicalReasoningModel_ACTV1Config
from src.training.optimizer import AdamAtan2

class HRMTrainer:
    def __init__(self, 
                 model: HierarchicalReasoningModel_ACTV1, 
                 config: HierarchicalReasoningModel_ACTV1Config,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 checkpoint_dir: str = "checkpoints"):
        self.model = model
        self.loss_head = ACTLossHead(model)
        self.config = config
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
    def train_epoch(self, 
                    token_stream: torch.Tensor, 
                    label_stream: torch.Tensor, 
                    epoch: int):
        """
        Args:
            token_stream: (Batch, TotalSteps) tensor of tokens.
            label_stream: (Batch, TotalSteps) tensor of labels.
        """
        self.model.train()
        
        # Initialize Carry
        # We need to construct a dummy batch to init carry
        dummy_batch = {
            "inputs": torch.zeros(self.config.batch_size, self.config.seq_len, device=self.device, dtype=torch.long),
            "labels": torch.zeros(self.config.batch_size, self.config.seq_len, device=self.device, dtype=torch.long),
            "puzzle_identifiers": torch.zeros(self.config.batch_size, device=self.device, dtype=torch.long)
        }
        carry = self.loss_head.initial_carry(dummy_batch)
        
        total_steps = token_stream.shape[1]
        seq_len = self.config.seq_len
        num_segments = total_steps // seq_len
        
        epoch_loss = 0
        progress_bar = tqdm(range(num_segments), desc=f"Epoch {epoch}")
        
        for i in progress_bar:
            # Prepare batch
            start = i * seq_len
            end = start + seq_len
            
            batch_tokens = token_stream[:, start:end].to(self.device)
            batch_labels = label_stream[:, start:end].to(self.device)
            
            # Update carry current data
            carry.current_data["inputs"] = batch_tokens
            carry.current_data["labels"] = batch_labels
            carry.current_data["puzzle_identifiers"] = dummy_batch["puzzle_identifiers"] # Dummy
            
            # Reset halted status for new segment? 
            # The HRM logic provided handles halted sequences by resetting them.
            # If we are training on a continuous stream, we typically DON'T want to halt permanently.
            # The provided HRM code is designed for "Reasoning until Answer" (halt).
            # But here we have a continuous stream.
            # We should force un-halt at the start of each segment?
            # OR: The model halts for *this specific segment's prediction*.
            # Yes, ACT is per-step or per-segment. 
            # In the provided code: `new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)`
            # If halted, it resets the inner state (L/H init). This breaks the long-term context!
            # FOR CONTINUOUS TIME SERIES: We probably do NOT want to reset context when it "halts" (finishes thinking about the current tick).
            # However, the provided code seems to treat "halted" as "done with this sequence".
            # We want "done thinking about this step, move to next step".
            # The provided code is for sequence-to-sequence where each sequence is independent?
            # "carry.halted" defaults to True in `initial_carry`.
            # Then it resets.
            # If we want continuous context, we should ensure `halted` is False coming into the new segment?
            # Actually, the `carry` object is passed out.
            # If `halted` is True, `reset_carry` is called.
            # For infinite stream, we want to maintain state.
            # So we must manually set `carry.halted` to False before the forward pass?
            # But `halted` comes from ACT.
            
            # ADAPTATION:
            # We force `halted` to False for the new segment to continue context,
            # UNLESS we want to explicitly reset context (e.g. end of trading day).
            # For now, let's assume continuous context.
            if i > 0:
                carry.halted.fill_(False)
            # For i == 0, carry.halted is True (default), so reset_carry will init states properly.
            
            carry.steps.fill_(0) # Reset thinking steps counter for the new input
            
            self.optimizer.zero_grad()
            
            # Forward
            # We use ACTLossHead which returns (new_carry, loss, metrics, outputs, all_halted)
            # It calls model(carry, batch)
            new_carry, loss, metrics, outputs, all_halted = self.loss_head(
                return_keys=["logits"],
                carry=carry, # Pass the carry explicitly if modified loss head?
                # The provided ACTLossHead signature: `forward(self, return_keys, **model_kwargs)`
                # It calls `self.model(**model_kwargs)`.
                # So we pass `carry` as kwarg.
                batch=carry.current_data
            )
            
            # STABILITY CHECK: Detect exploding loss early
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1000:
                print(f"WARNING: Unstable loss detected: {loss.item():.2e}")
                print("Skipping this batch to prevent explosion...")
                carry = new_carry  # Still update carry
                continue
            
            # Backward
            loss.backward()
            
            # AGGRESSIVE GRADIENT CLIPPING
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            
            # Skip update if gradients are still too large
            if grad_norm > 10.0:
                print(f"WARNING: Large gradient norm {grad_norm:.2f}, skipping update")
                self.optimizer.zero_grad()
                carry = new_carry
                continue
                
            self.optimizer.step()
            
            # Detach carry for BPTT
            # We need to detach the inner states in new_carry
            # The `HierarchicalReasoningModel_ACTV1` detaches `new_carry` in `forward` 
            # "new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())"
            # So it is already detached! 
            # Wait, check the code.
            # In `HierarchicalReasoningModel_ACTV1_Inner.forward`:
            # `new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())`
            # Yes, it detaches. This implements the 1-step gradient approximation (Truncated BPTT with window 1 segment).
            
            carry = new_carry
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), acc=metrics.get("accuracy", 0).item())
            
        print(f"Epoch {epoch} Average Loss: {epoch_loss / num_segments:.4f}")
        
        # Save
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f"hrm_epoch_{epoch}.pt"))

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

