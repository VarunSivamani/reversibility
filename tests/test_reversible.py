"""Gradient-correctness checks for the reversible stacks.

For midpoint, leapfrog, and hamiltonian, each custom autograd Function is
compared against a plain-autograd reference that builds the ordinary
(memory-heavy) graph for the exact same recursion, on identical inputs and
parameters. If both give the same loss and the same parameter/input
gradients (within floating-point tolerance), the custom backward correctly
implements the analytic recursion.

Euler is not reversible and uses ordinary autograd directly (no custom
Function to verify), so it has no check here.
"""

import copy

import torch
import torch.nn as nn

from reversibility.reversible import (
    HamiltonianReversibleStack,
    LeapfrogReversibleStack,
    MidpointReversibleStack,
)


class TinyLayer(nn.Module):
    """Small nonlinear layer standing in for a transformer block's f_theta
    (or, for Hamiltonian, one half of it)."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(torch.tanh(self.lin1(x)))


def _assert_layers_match(layers_a, layers_b, label: str) -> bool:
    ok = True
    for i, (la, lb) in enumerate(zip(layers_a, layers_b)):
        for (na, pa), (nb, pb) in zip(la.named_parameters(), lb.named_parameters()):
            close = torch.allclose(pa.grad, pb.grad, atol=1e-5)
            ok &= close
            diff = (pa.grad - pb.grad).abs().max().item()
            print(f"  [{label}] layer {i} param {na}: grad match {close} (max abs diff {diff:.3e})")
    return ok


# --------------------------------------------------------------------------
# Midpoint
# --------------------------------------------------------------------------


def reference_midpoint(layers, p0, h):
    prev, curr = p0, p0
    for layer in layers:
        nxt = prev + 2.0 * h * layer(curr)
        prev, curr = curr, nxt
    return curr


def run_check_midpoint(n_layers=4, dim=8, batch=3, seq=5, h=0.3, seed=0) -> bool:
    print("\n=== midpoint ===")
    torch.manual_seed(seed)
    layers_a = [TinyLayer(dim) for _ in range(n_layers)]
    layers_b = [copy.deepcopy(layer) for layer in layers_a]

    x0 = torch.randn(batch, seq, dim, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)

    stack = MidpointReversibleStack([layer.forward for layer in layers_a], h=h)
    out_custom = stack(x0)
    out_custom.pow(2).mean().backward()

    out_ref = reference_midpoint(layers_b, x1, h)
    out_ref.pow(2).mean().backward()

    fwd_close = torch.allclose(out_custom, out_ref, atol=1e-6)
    grad_close = torch.allclose(x0.grad, x1.grad, atol=1e-5)
    print(f"  forward match: {fwd_close}  input grad match: {grad_close}")
    ok = fwd_close and grad_close
    ok &= _assert_layers_match(layers_a, layers_b, "midpoint")
    return ok


# --------------------------------------------------------------------------
# Leapfrog
# --------------------------------------------------------------------------


def reference_leapfrog(layers, p0, h):
    prev, curr = p0, p0
    h2 = h * h
    for layer in layers:
        nxt = 2.0 * curr - prev + h2 * layer(curr)
        prev, curr = curr, nxt
    return curr


def run_check_leapfrog(n_layers=4, dim=8, batch=3, seq=5, h=0.3, seed=1) -> bool:
    print("\n=== leapfrog ===")
    torch.manual_seed(seed)
    layers_a = [TinyLayer(dim) for _ in range(n_layers)]
    layers_b = [copy.deepcopy(layer) for layer in layers_a]

    x0 = torch.randn(batch, seq, dim, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)

    stack = LeapfrogReversibleStack([layer.forward for layer in layers_a], h=h)
    out_custom = stack(x0)
    out_custom.pow(2).mean().backward()

    out_ref = reference_leapfrog(layers_b, x1, h)
    out_ref.pow(2).mean().backward()

    fwd_close = torch.allclose(out_custom, out_ref, atol=1e-6)
    grad_close = torch.allclose(x0.grad, x1.grad, atol=1e-5)
    print(f"  forward match: {fwd_close}  input grad match: {grad_close}")
    ok = fwd_close and grad_close
    ok &= _assert_layers_match(layers_a, layers_b, "leapfrog")
    return ok


# --------------------------------------------------------------------------
# Hamiltonian
# --------------------------------------------------------------------------


def reference_hamiltonian(layer_pairs, p0):
    p, q = p0, p0
    for attn_part, mlp_part in layer_pairs:
        q = q + attn_part(p)
        p = p + mlp_part(q)
    return p


def run_check_hamiltonian(n_layers=4, dim=8, batch=3, seq=5, seed=2) -> bool:
    print("\n=== hamiltonian ===")
    torch.manual_seed(seed)
    attn_layers_a = [TinyLayer(dim) for _ in range(n_layers)]
    mlp_layers_a = [TinyLayer(dim) for _ in range(n_layers)]
    attn_layers_b = [copy.deepcopy(layer) for layer in attn_layers_a]
    mlp_layers_b = [copy.deepcopy(layer) for layer in mlp_layers_a]

    x0 = torch.randn(batch, seq, dim, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)

    pairs_a = [(a.forward, m.forward) for a, m in zip(attn_layers_a, mlp_layers_a)]
    stack = HamiltonianReversibleStack(pairs_a)
    out_custom = stack(x0)
    out_custom.pow(2).mean().backward()

    pairs_b = list(zip(attn_layers_b, mlp_layers_b))
    out_ref = reference_hamiltonian([(a.forward, m.forward) for a, m in pairs_b], x1)
    out_ref.pow(2).mean().backward()

    fwd_close = torch.allclose(out_custom, out_ref, atol=1e-6)
    grad_close = torch.allclose(x0.grad, x1.grad, atol=1e-5)
    print(f"  forward match: {fwd_close}  input grad match: {grad_close}")
    ok = fwd_close and grad_close
    ok &= _assert_layers_match(attn_layers_a, attn_layers_b, "hamiltonian-attn")
    ok &= _assert_layers_match(mlp_layers_a, mlp_layers_b, "hamiltonian-mlp")
    return ok


if __name__ == "__main__":
    results = {
        "midpoint": run_check_midpoint(),
        "leapfrog": run_check_leapfrog(),
        "hamiltonian": run_check_hamiltonian(),
    }
    print("\n=== SUMMARY ===")
    for name, ok in results.items():
        print(f"{name}: {'PASSED' if ok else 'FAILED'}")
    raise SystemExit(0 if all(results.values()) else 1)
