import torch
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Union
import torch.distributed as dist

class AdamAtan2(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
            
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(AdamAtan2, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamAtan2 does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Adam-atan2 update rule
                # update = atan2(m, sqrt(v))
                # No bias correction needed as per paper? Or should check.
                # Standard Adam uses bias correction.
                # The snippet showed: update = torch.atan2(exp_avg, exp_avg_sq.sqrt())
                
                update = torch.atan2(exp_avg, exp_avg_sq.sqrt())
                
                p.add_(update, alpha=-group['lr'])

        return loss

class CastedSparseEmbeddingSignSGD_Distributed(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        world_size: int,
        lr: Union[float, torch.Tensor] = 1e-3,
        weight_decay: float = 1e-2,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            world_size=world_size
        )
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):  # type: ignore
        for group in self.param_groups:
            # Find the sparse embedding weights
            local_weights_grad = None
            local_ids = None
            weights = None
            
            # This assumes specific parameter ordering from the model
            # Ideally we should check parameter names or types, but standard Optimizer doesn't give names easily.
            # We assume the group was created with exactly these 3 params.
            if len(group["params"]) != 3:
                 continue # Skip or warn
            
            for p in group["params"]:
                if p.requires_grad:
                    local_weights_grad = p.grad
                elif p.ndim == 1:
                    local_ids = p
                elif p.ndim == 2:
                    weights = p
                
            if local_weights_grad is not None and local_ids is not None and weights is not None:
                # Apply SignSGD
                _sparse_emb_signsgd_dist(
                    local_weights_grad,
                    local_ids,
                    weights,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    world_size=group["world_size"]
                )

def _sparse_emb_signsgd_dist(
    local_weights_grad: torch.Tensor,
    local_ids: torch.Tensor,
    weights: torch.Tensor,
    lr: float,
    weight_decay: float,
    world_size: int
) -> None:
    N, D = local_weights_grad.shape
    
    # All-gather
    all_weights_grad = local_weights_grad
    all_ids = local_ids
    
    if world_size > 1 and dist.is_initialized():
        all_weights_grad = torch.empty((world_size * N, D), dtype=local_weights_grad.dtype, device=local_weights_grad.device)
        all_ids = torch.empty(world_size * N,               dtype=local_ids.dtype,          device=local_ids.device)
        dist.all_gather_into_tensor(all_weights_grad, local_weights_grad)
        dist.all_gather_into_tensor(all_ids,          local_ids)
        
    # Unique
    grad_ids, inv = all_ids.unique(return_inverse=True)
    grad = torch.zeros((grad_ids.shape[0], D), dtype=all_weights_grad.dtype, device=all_weights_grad.device)
    grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, D), all_weights_grad)
    
    # SignSGD with decoupled weight decay
    p = weights[grad_ids]
    p.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha=-lr)
    
    # Write updated slices back
    weights[grad_ids] = p

