# --------------------------------------------------------
# LoRA adapters for timm 0.3.2 VisionTransformer
# Injects low-rank adapters into attention Q/V/K/Proj and
# optionally MLP layers. All non-LoRA params are frozen.
# --------------------------------------------------------

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a frozen base + trainable LoRA."""

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float = None):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank

        # Frozen base weight + bias
        self.weight = base_linear.weight
        self.weight.requires_grad = False
        self.bias = base_linear.bias
        if self.bias is not None:
            self.bias.requires_grad = False

        # Trainable LoRA factors
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = nn.functional.linear(x, self.weight, self.bias)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out


def inject_lora(model: nn.Module, rank: int, alpha: float = None,
                target_modules: str = "qkv_proj") -> dict:
    """
    Inject LoRA adapters into a timm VisionTransformer.

    Args:
        model: A VisionTransformer instance (timm 0.3.2).
        rank: LoRA rank.
        alpha: LoRA scaling alpha. Defaults to rank.
        target_modules: Which modules to adapt.
            "qkv"       — only the fused qkv projection in each Attention block
            "qkv_proj"  — qkv + output projection
            "all"       — qkv + output projection + MLP fc1/fc2

    Returns:
        dict with injection stats.
    """
    injected = 0
    total_lora_params = 0

    for block in model.blocks:
        attn = block.attn
        mlp = block.mlp

        # --- Attention QKV (fused) ---
        if target_modules in ("qkv", "qkv_proj", "all"):
            old_qkv = attn.qkv
            new_qkv = LoRALinear(old_qkv, rank=rank, alpha=alpha)
            attn.qkv = new_qkv
            injected += 1
            total_lora_params += rank * old_qkv.in_features + old_qkv.out_features * rank

        # --- Attention output projection ---
        if target_modules in ("qkv_proj", "all"):
            old_proj = attn.proj
            new_proj = LoRALinear(old_proj, rank=rank, alpha=alpha)
            attn.proj = new_proj
            injected += 1
            total_lora_params += rank * old_proj.in_features + old_proj.out_features * rank

        # --- MLP fc1 and fc2 ---
        if target_modules == "all":
            old_fc1 = mlp.fc1
            new_fc1 = LoRALinear(old_fc1, rank=rank, alpha=alpha)
            mlp.fc1 = new_fc1
            injected += 1
            total_lora_params += rank * old_fc1.in_features + old_fc1.out_features * rank

            old_fc2 = mlp.fc2
            new_fc2 = LoRALinear(old_fc2, rank=rank, alpha=alpha)
            mlp.fc2 = new_fc2
            injected += 1
            total_lora_params += rank * old_fc2.in_features + old_fc2.out_features * rank

    return {"injected_layers": injected, "lora_params": total_lora_params}


def freeze_non_lora(model: nn.Module, freeze_head: bool = False, freeze_lora_a: bool = False):
    """Freeze all parameters except LoRA adapters and (optionally) the head.
    If freeze_lora_a is True, applies FFA-LoRA logic (freezes A matrix to eliminate aggregation bias).
    """
    for name, param in model.named_parameters():
        if "lora_A" in name and freeze_lora_a:
            param.requires_grad = False
        elif "lora_" in name:
            param.requires_grad = True
        elif not freeze_head and "head" in name:
            param.requires_grad = True
        elif not freeze_head and "fc_norm" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def get_lora_state_dict(model: nn.Module) -> dict:
    """Extract only the LoRA parameters for communication-efficient aggregation."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def get_trainable_state_dict(model: nn.Module) -> dict:
    """Extract all trainable parameters (LoRA + head + fc_norm)."""
    return {k: v for k, v in model.state_dict().items()
            if any(t in k for t in ("lora_", "head.", "fc_norm."))}
