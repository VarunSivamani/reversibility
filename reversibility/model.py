"""A small GPT-style decoder-only transformer with a switchable block type
and a switchable FFN type.

block_type="euler"        -> ordinary pre-norm residual transformer
                              (non-reversible baseline, full activation
                              caching). This is the paper's own "Euler"
                              terminology for the standard residual update.
block_type="midpoint"     -> reversible midpoint coupling (Eq. 2.4).
block_type="leapfrog"     -> reversible leapfrog / second-order coupling
                              (Eq. 2.6).
block_type="hamiltonian"  -> reversible Hamiltonian coupling with two
                              streams, position p and momentum q (Eq. 2.8-2.9).

All three reversible variants are executed through custom autograd
Functions in reversible.py so activations are not cached.

ffn_type="dense"  -> a single feedforward MLP per block (today's default).
ffn_type="moe"    -> a mixture-of-experts FFN: top-k routed experts plus one
                      always-on shared expert (DeepSeek-style, matches
                      S13.pdf Sections 11-13).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reversible import (
    EulerReversibleStack,
    HamiltonianReversibleStack,
    LeapfrogReversibleStack,
    MidpointReversibleStack,
)

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _HAS_SDPA_KERNEL_CTX = True
except ImportError:  # older torch without the explicit backend-selection API
    _HAS_SDPA_KERNEL_CTX = False


@dataclass
class GPTConfig:
    vocab_size: int = 50257  # GPT-2 BPE
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False

    block_type: str = "euler"  # "euler" | "midpoint" | "leapfrog" | "hamiltonian"
    step_h: float = 0.5  # step size h in Eq. 2.4 / 2.6

    ffn_type: str = "dense"  # "dense" | "moe"
    n_experts: int = 8
    expert_hidden_mult: float = 0.5
    moe_top_k: int = 2
    moe_aux_loss_coef: float = 0.01

    force_flash_attention: bool = True  # forces (and verifies) the flash SDPA backend on CUDA


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.force_flash = config.force_flash_attention
        self._checked_backend = False

    def _run_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        dropout_p = self.attn_dropout if self.training else 0.0
        use_forced_flash = self.force_flash and _HAS_SDPA_KERNEL_CTX and q.is_cuda
        if use_forced_flash and q.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                f"forced FlashAttention requires fp16/bf16 q/k/v, got {q.dtype}. "
                "On CUDA, run forward (and reversible recompute) under "
                "torch.autocast(device_type='cuda', dtype=torch.bfloat16)."
            )
        if use_forced_flash:
            # Force the flash backend specifically; this raises rather than
            # silently falling back to the slower math/mem-efficient kernel
            # if flash isn't usable for the given shapes/dtype, so a
            # misconfiguration is caught immediately instead of just being slow.
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
            if not self._checked_backend:
                print("[attention] forced FlashAttention backend confirmed usable")
                self._checked_backend = True
            return out
        # CPU/MPS dry-runs, or force_flash disabled: default backend selection.
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = self._run_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """A single dense feedforward network, the classic transformer FFN."""

    def __init__(self, config: GPTConfig, hidden_mult: float = 4.0):
        super().__init__()
        hidden = int(config.n_embd * hidden_mult)
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class MoEFFN(nn.Module):
    """Mixture-of-experts FFN: a router selects the top-k of n_experts small
    MLP experts per token, and one always-on shared expert is added to every
    token unconditionally (DeepSeek-V3/V4 style, per S13.pdf Sections 11-13).

    Unlike the dense MLP, this module also produces a load-balancing
    auxiliary loss (encouraging the router to spread tokens evenly across
    experts), exposed via `self.last_aux_loss` after each forward call so
    the outer model can add it to the LM loss.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.moe_top_k
        self.aux_loss_coef = config.moe_aux_loss_coef

        self.router = nn.Linear(config.n_embd, config.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [MLP(config, hidden_mult=config.expert_hidden_mult) for _ in range(config.n_experts)]
        )
        self.shared_expert = MLP(config, hidden_mult=config.expert_hidden_mult)

        self.last_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        flat_x = x.reshape(-1, C)  # (B*T, C)
        n_tokens = flat_x.shape[0]

        router_logits = self.router(flat_x)  # (B*T, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_idx = torch.topk(router_probs, self.top_k, dim=-1)  # (B*T, top_k)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # renormalize

        out = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            # Which (token, slot) pairs routed to this expert, across all top_k slots.
            mask = topk_idx == expert_id  # (B*T, top_k)
            if not mask.any():
                continue
            token_idx, slot_idx = mask.nonzero(as_tuple=True)
            weights = topk_probs[token_idx, slot_idx].unsqueeze(-1)
            expert_out = expert(flat_x[token_idx])
            out.index_add_(0, token_idx, weights * expert_out)

        out = out + self.shared_expert(flat_x)

        # Load-balancing auxiliary loss: encourage uniform dispatch fraction
        # across experts (Switch-Transformer-style), scaled by n_experts so
        # the "uniform" target has value 1 regardless of n_experts.
        dispatch_mask = F.one_hot(topk_idx, num_classes=self.n_experts).float().sum(dim=1)  # (B*T, n_experts)
        dispatch_fraction = dispatch_mask.mean(dim=0)  # (n_experts,)
        router_prob_mean = router_probs.mean(dim=0)  # (n_experts,)
        self.last_aux_loss = self.aux_loss_coef * self.n_experts * (dispatch_fraction * router_prob_mean).sum()

        return out.reshape(B, T, C)


class Block(nn.Module):
    """One transformer block, used both standalone (euler mode) and as the
    f_theta inside the reversible stacks. For the Hamiltonian variant, the
    attention and MLP halves are also exposed separately as attn_part/
    mlp_part, since that update has two coupled sub-steps rather than one
    fused f_theta call."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp: nn.Module = MoEFFN(config) if config.ffn_type == "moe" else MLP(config)

    def _mlp_aux_loss(self) -> torch.Tensor:
        if isinstance(self.mlp, MoEFFN):
            return self.mlp.last_aux_loss
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def f_theta(self, p: torch.Tensor) -> torch.Tensor:
        """The whole block as one function, matching Eq. (2.5) of the paper:
        f(p) = Attn(LN1(p)) + MLP(LN2(p + Attn(LN1(p))))."""
        a = self.attn(self.ln_1(p))
        return a + self.mlp(self.ln_2(p + a))

    def attn_part(self, p: torch.Tensor) -> torch.Tensor:
        """Attention half only, used by the Hamiltonian coupling: Attn(LN1(p))."""
        return self.attn(self.ln_1(p))

    def mlp_part(self, q: torch.Tensor) -> torch.Tensor:
        """MLP half only, used by the Hamiltonian coupling: MLP(LN2(q))."""
        return self.mlp(self.ln_2(q))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard pre-norm residual update, used only in "euler" mode."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


_REVERSIBLE_STACKS = {
    "midpoint": MidpointReversibleStack,
    "leapfrog": LeapfrogReversibleStack,
    "hamiltonian": HamiltonianReversibleStack,
}


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying, standard nanoGPT trick to save params
        self.tok_emb.weight = self.lm_head.weight

        if config.block_type == "euler":
            self.rev_stack = None
        elif config.block_type in _REVERSIBLE_STACKS:
            if config.dropout != 0.0:
                raise ValueError(
                    f"Reversible ({config.block_type}) mode requires dropout=0: the "
                    "backward pass recomputes f_theta and needs it to be deterministic."
                )
            stack_cls = _REVERSIBLE_STACKS[config.block_type]
            if config.block_type == "hamiltonian":
                self.rev_stack = stack_cls(
                    [(b.attn_part, b.mlp_part) for b in self.blocks]
                )
            else:
                self.rev_stack = stack_cls([b.f_theta for b in self.blocks], h=config.step_h)
        else:
            raise ValueError(f"unknown block_type {config.block_type!r}")

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n

    def _moe_aux_loss(self) -> torch.Tensor:
        if self.config.ffn_type != "moe":
            return torch.tensor(0.0, device=self.tok_emb.weight.device)
        return sum((b._mlp_aux_loss() for b in self.blocks), torch.tensor(0.0, device=self.tok_emb.weight.device))

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.config.block_size
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        if self.config.block_type == "euler":
            for block in self.blocks:
                x = block(x)
        elif self.config.block_type == "hamiltonian":
            x = self.rev_stack(x)  # stack internally carries (p, q), returns p
        else:
            x = self.rev_stack(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            loss = lm_loss + self._moe_aux_loss()
        return logits, loss
