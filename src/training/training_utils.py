"""
Helper functions for adjusting learning rate and creating optimizers
"""

from omegaconf import DictConfig
import math
import torch


def adjust_learning_rate(
    optimizer: torch.optim, step: int, cfg: DictConfig
) -> float:
    """
    Adjusts learning rate to follow cosine annealing with warmup + a (possible)
    constant cooldown at the end.
    """
    lr = cfg.training.max_lr
    if step < cfg.training.warmup:
        lr = step * cfg.training.max_lr / cfg.training.warmup
    else:
        if step < cfg.training.anneal_steps:
            if cfg.training.no_lr_decay == False:
                lr = (cfg.training.min_lr) + 0.5 * (
                    cfg.training.max_lr - cfg.training.min_lr
                ) * (
                    1.0
                    + math.cos(
                        math.pi
                        * (step - cfg.training.warmup)
                        / (cfg.training.anneal_steps - cfg.training.warmup)
                    )
                )
            else:
                pass
        else:
            lr = cfg.training.min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def adjust_learning_rate_inv_sqrt(
    optimizer: torch.optim, step: int, cfg: DictConfig
) -> float:
    """
    Adjusts learning rate to follow inverse sqrt with warmup. Learning rate
    decays to 0 at the end of training.
    """
    lr = cfg.training.max_lr

    scale_factor = cfg.training.max_lr * math.sqrt(cfg.training.warmup)

    if step < cfg.training.warmup:
        lr = step * cfg.training.max_lr / cfg.training.warmup
    else:
        if step < cfg.training.anneal_steps:
            # figure out scale factor from this
            lr = scale_factor / math.sqrt(step)
        else:
            final_lr = scale_factor / math.sqrt(cfg.training.anneal_steps)
            remaining_total_steps = (
                cfg.training.steps - cfg.training.anneal_steps
            )
            remaining_steps = step - cfg.training.anneal_steps
            lr = (
                -1
                * (remaining_steps - remaining_total_steps)
                * final_lr
                / remaining_total_steps
            )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def create_optimizer(
    model: torch.nn.Module, weight_decay: float, lr: float
) -> torch.optim:
    """
    Creates AdamW optimizer and disables weight decay for the following params:
        1. All bias terms
        2. LayerNorm alpha/beta
        3. Position embeddings

    AdamW params (\beta_1,\beta_2) follow `Language Models are Few-Shot Learners
    <https://arxiv.org/abs/2005.14165>`
    """
    params = []
    for key, value in model.named_parameters():

        if (
            "fc.bias" in key
            or "bias" in key
            or "alpha" in key
            or "beta" in key
            or "ln" in key
            or "wpe" in key
        ):
            apply_weight_decay = 0
            params += [
                {
                    "params": [value],
                    "lr": lr,
                    "weight_decay": apply_weight_decay,
                }
            ]

        else:
            apply_weight_decay = weight_decay
            params += [
                {
                    "params": [value],
                    "lr": lr,
                    "weight_decay": apply_weight_decay,
                }
            ]

    return torch.optim.AdamW(params, lr, betas=(0.9, 0.95), eps=1e-8)
