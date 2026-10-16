"""Custom autograd Functions for the reversible transformer stacks used in
this project (Section 16 of S13.pdf; Eqs. 2.4, 2.6, 2.8-2.9 of "Reversing
Large Language Models for Efficient Training and Fine-Tuning", Gal et al.
2025).

Standard autograd needs every layer's input activation to compute the
backward pass, so a stack of L blocks stores L activation tensors. Each
reversible update below is analytically invertible, so the backward pass
can walk back down the stack recomputing each layer's local graph on demand
instead of caching it -- that is the actual memory saving this project
measures, not just a different forward formula.

Four variants, matching the paper:

  - euler:       NOT reversible. p^{l+1} = p^l + f_l(p^l). Included only as
                 the non-reversible baseline; routed through Block.forward's
                 ordinary residual loop in model.py (EulerReversibleStack
                 below is a thin, uniform-interface wrapper around that
                 loop, using standard autograd -- no custom Function).
  - midpoint:    p^{l+1} = p^{l-1} + 2h f_l(p^l)                    (Eq. 2.4)
  - leapfrog:    p^{l+1} = 2 p^l - p^{l-1} + h^2 f_l(p^l)           (Eq. 2.6)
  - hamiltonian: q^l = q^{l-1} + Attn_l(LN1(p^{l-1}))
                 p^l = p^{l-1} + MLP_l(LN2(q^l))                    (Eq. 2.8-2.9)

Midpoint and leapfrog share the same (prev, curr) two-state bookkeeping
pattern (only the combination coefficients differ); Hamiltonian carries two
coupled streams (p, q) instead.
"""

from __future__ import annotations

from typing import Callable, Sequence

import contextlib

import torch
import torch.nn as nn


def _collect_param_refs(owners: Sequence[nn.Module]) -> list[nn.Parameter]:
    seen: set[int] = set()
    refs: list[nn.Parameter] = []
    for owner in owners:
        if owner is None:
            continue
        for p in owner.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                refs.append(p)
    return refs


def _capture_autocast_state() -> tuple[bool, str | None, torch.dtype | None]:
    """Snapshot the outer autocast state at forward time so backward can
    recompute f_theta under the identical dtype.

    Without this, training under ``torch.autocast(cuda, bf16)`` runs forward
    in bf16 (flash attention works) but the custom backward recomputes in
    fp32 (flash has no fp32 kernel -> "No available kernel" Dynamo crash),
    and the reconstruction no longer matches forward numerically.
    """
    # CUDA autocast (the training path in train.py).
    try:
        try:
            cuda_enabled = torch.is_autocast_enabled("cuda")  # type: ignore[call-arg]
        except TypeError:
            cuda_enabled = torch.is_autocast_enabled()
    except Exception:  # noqa: BLE001
        cuda_enabled = False
    if cuda_enabled:
        try:
            dtype = torch.get_autocast_gpu_dtype()
        except Exception:  # noqa: BLE001
            try:
                dtype = torch.get_autocast_dtype("cuda")  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                dtype = torch.bfloat16
        return True, "cuda", dtype
    # CPU autocast (rare; no flash issue, but preserve for numerics).
    try:
        cpu_enabled = torch.is_autocast_cpu_enabled()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        cpu_enabled = False
    if cpu_enabled:
        try:
            dtype = torch.get_autocast_cpu_dtype()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            dtype = torch.bfloat16
        return True, "cpu", dtype
    return False, None, None


def _recompute_autocast_ctx(enabled: bool, device_type: str | None, dtype: torch.dtype | None):
    """Re-enter the forward autocast state during backward recomputation."""
    if enabled and device_type is not None and dtype is not None:
        return torch.autocast(device_type=device_type, dtype=dtype)
    return contextlib.nullcontext()


# --------------------------------------------------------------------------
# Euler (non-reversible baseline, uniform-interface wrapper)
# --------------------------------------------------------------------------


class EulerReversibleStack(nn.Module):
    """Not actually reversible. Plain residual stack, p^{l+1} = p^l + f_l(p^l),
    using ordinary autograd (full activation caching). Exists only so
    model.py can build every block_type through the same *Stack interface;
    the memory-saving property does not apply here."""

    def __init__(self, fns: Sequence[Callable[[torch.Tensor], torch.Tensor]], h: float = 1.0):
        super().__init__()
        self.fns = list(fns)

    def forward(self, p0: torch.Tensor) -> torch.Tensor:
        p = p0
        for fn in self.fns:
            p = p + fn(p)
        return p


# --------------------------------------------------------------------------
# Midpoint (Eq. 2.4) -- unchanged from the original implementation.
# --------------------------------------------------------------------------


class MidpointReversibleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p0, h, fns, param_refs, *flat_params):
        # Bootstrap: p^{-1} = p^{0} = the input embedding (no layer before
        # the first one), matching the paper's midpoint initialisation.
        prev, curr = p0, p0
        with torch.no_grad():
            for fn in fns:
                nxt = prev + 2.0 * h * fn(curr)
                prev, curr = curr, nxt

        ctx.h = h
        ctx.fns = fns
        ctx.param_refs = param_refs
        ctx.autocast_enabled, ctx.autocast_device_type, ctx.autocast_dtype = _capture_autocast_state()
        ctx.save_for_backward(prev.detach(), curr.detach())
        return curr

    @staticmethod
    def backward(ctx, grad_curr):
        h = ctx.h
        fns = ctx.fns
        prev, curr = ctx.saved_tensors  # live pair AFTER the last layer

        grad_prev = torch.zeros_like(grad_curr)
        param_grads: dict[int, torch.Tensor] = {}

        for fn in reversed(fns):
            # f_l's input was `prev` (== p^l for this layer, since the live
            # pair here is the post-step pair (curr_of_step, next_of_step)).
            # allow_unused=True (+ filtering None below) is required for an
            # MoE f_theta: some experts may receive zero tokens in a given
            # batch, so their weights legitimately have no gradient path
            # through this particular call.
            f_input = prev.detach().requires_grad_(True)
            params_this_layer = list(fn.__self__.parameters()) if hasattr(fn, "__self__") else []
            with torch.enable_grad():
                with _recompute_autocast_ctx(ctx.autocast_enabled, ctx.autocast_device_type, ctx.autocast_dtype):
                    f_val = fn(f_input)  # f_l(p^l)

            grads = torch.autograd.grad(
                f_val,
                [f_input, *params_this_layer],
                grad_outputs=2.0 * h * grad_curr,
                retain_graph=False,
                allow_unused=True,
            )
            grad_f_input, grads_params = grads[0], grads[1:]
            for p, g in zip(params_this_layer, grads_params):
                if g is None:
                    continue
                key = id(p)
                param_grads[key] = g if key not in param_grads else param_grads[key] + g

            with torch.no_grad():
                old_prev = curr - 2.0 * h * f_val  # reconstructed p^{l-1}

            # Gradient recursion for one step of
            #   (old_prev, old_curr) -> (old_curr, next)
            # where old_curr == the tensor currently called `prev`, and
            # next == the tensor currently called `curr`.
            #   d/d(old_prev) : only via `next = old_prev + 2h f(old_curr)`.
            #   d/d(old_curr) : via grad_prev (passed through unchanged to
            #                   become the next pair's `prev`) AND via f_l.
            grad_curr, grad_prev = grad_prev + grad_f_input, grad_curr

            prev, curr = old_prev, prev  # step the live pair back one layer

        # Loop ends with (prev, curr) == (p0, p0); total grad wrt the shared
        # input is the sum of both channels.
        grad_p0 = grad_curr + grad_prev
        param_grad_list = [param_grads.get(id(p)) for p in ctx.param_refs]
        return (grad_p0, None, None, None, *param_grad_list)


class MidpointReversibleStack(nn.Module):
    """Wraps a sequence of layer callables into one reversible stack
    evaluated via the midpoint rule of Eq. (2.4), with activation memory
    independent of depth."""

    def __init__(self, fns: Sequence[Callable[[torch.Tensor], torch.Tensor]], h: float = 0.5):
        super().__init__()
        self.fns = list(fns)
        self.h = h
        self._param_refs = _collect_param_refs([getattr(fn, "__self__", None) for fn in self.fns])

    def forward(self, p0: torch.Tensor) -> torch.Tensor:
        return MidpointReversibleFunction.apply(p0, self.h, self.fns, self._param_refs, *self._param_refs)


# --------------------------------------------------------------------------
# Leapfrog (Eq. 2.6): p^{l+1} = 2 p^l - p^{l-1} + h^2 f_l(p^l)
# --------------------------------------------------------------------------


class LeapfrogReversibleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p0, h, fns, param_refs, *flat_params):
        prev, curr = p0, p0
        h2 = h * h
        with torch.no_grad():
            for fn in fns:
                nxt = 2.0 * curr - prev + h2 * fn(curr)
                prev, curr = curr, nxt

        ctx.h2 = h2
        ctx.fns = fns
        ctx.param_refs = param_refs
        ctx.autocast_enabled, ctx.autocast_device_type, ctx.autocast_dtype = _capture_autocast_state()
        ctx.save_for_backward(prev.detach(), curr.detach())
        return curr

    @staticmethod
    def backward(ctx, grad_curr):
        h2 = ctx.h2
        fns = ctx.fns
        prev, curr = ctx.saved_tensors  # live pair AFTER the last layer

        grad_prev = torch.zeros_like(grad_curr)
        param_grads: dict[int, torch.Tensor] = {}

        for fn in reversed(fns):
            # f_l's input was `prev` (== p^l for this layer). allow_unused=True
            # (+ filtering None below) is required for an MoE f_theta: some
            # experts may receive zero tokens in a given batch, so their
            # weights legitimately have no gradient path through this call.
            f_input = prev.detach().requires_grad_(True)
            params_this_layer = list(fn.__self__.parameters()) if hasattr(fn, "__self__") else []
            with torch.enable_grad():
                with _recompute_autocast_ctx(ctx.autocast_enabled, ctx.autocast_device_type, ctx.autocast_dtype):
                    f_val = fn(f_input)

            grads = torch.autograd.grad(
                f_val,
                [f_input, *params_this_layer],
                grad_outputs=h2 * grad_curr,
                retain_graph=False,
                allow_unused=True,
            )
            grad_f_input, grads_params = grads[0], grads[1:]
            for p, g in zip(params_this_layer, grads_params):
                if g is None:
                    continue
                key = id(p)
                param_grads[key] = g if key not in param_grads else param_grads[key] + g

            with torch.no_grad():
                # next = 2*old_curr - old_prev + h2*f(old_curr)
                # => old_prev = 2*old_curr - next + h2*f(old_curr) = 2*prev - curr + h2*f_val
                old_prev = 2.0 * prev - curr + h2 * f_val

            # Gradient recursion for one step of
            #   (old_prev, old_curr) -> (old_curr, next)
            # next = 2*old_curr - old_prev + h2*f(old_curr)
            #   d/d(old_prev) : coefficient -1 on next.
            #   d/d(old_curr) : coefficient +2 on next, plus via f_l, plus
            #                   grad_prev passed through (old_curr becomes
            #                   the next pair's `prev`, same as midpoint).
            new_grad_curr = 2.0 * grad_curr + grad_f_input + grad_prev
            new_grad_prev = -grad_curr

            grad_curr, grad_prev = new_grad_curr, new_grad_prev
            prev, curr = old_prev, prev  # step the live pair back one layer

        grad_p0 = grad_curr + grad_prev
        param_grad_list = [param_grads.get(id(p)) for p in ctx.param_refs]
        return (grad_p0, None, None, None, *param_grad_list)


class LeapfrogReversibleStack(nn.Module):
    """Wraps a sequence of layer callables into one reversible stack
    evaluated via the leapfrog rule of Eq. (2.6)."""

    def __init__(self, fns: Sequence[Callable[[torch.Tensor], torch.Tensor]], h: float = 0.5):
        super().__init__()
        self.fns = list(fns)
        self.h = h
        self._param_refs = _collect_param_refs([getattr(fn, "__self__", None) for fn in self.fns])

    def forward(self, p0: torch.Tensor) -> torch.Tensor:
        return LeapfrogReversibleFunction.apply(p0, self.h, self.fns, self._param_refs, *self._param_refs)


# --------------------------------------------------------------------------
# Hamiltonian (Eq. 2.8-2.9): two coupled streams p ("position"), q ("momentum")
#   q^l = q^{l-1} + Attn_l(LN1(p^{l-1}))
#   p^l = p^{l-1} + MLP_l(LN2(q^l))
# with a_l = b_l = 1 (per the paper's choice; see Section 3.1's stability
# analysis, which requires a*b = 1 for forward/backward stability).
# --------------------------------------------------------------------------


class HamiltonianReversibleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p0, fns, param_refs, *flat_params):
        # fns is a list of (attn_part, mlp_part) callable pairs, one per layer.
        p, q = p0, p0  # bootstrap: q^{-1} = p^{-1} = the input embedding
        with torch.no_grad():
            for attn_part, mlp_part in fns:
                q = q + attn_part(p)
                p = p + mlp_part(q)

        ctx.fns = fns
        ctx.param_refs = param_refs
        ctx.autocast_enabled, ctx.autocast_device_type, ctx.autocast_dtype = _capture_autocast_state()
        ctx.save_for_backward(p.detach(), q.detach())
        return p, q

    @staticmethod
    def backward(ctx, grad_p, grad_q):
        fns = ctx.fns
        p, q = ctx.saved_tensors  # live pair AFTER the last layer

        if grad_q is None:
            # The caller (HamiltonianReversibleStack) only consumes the
            # position stream p, so autograd supplies None for the
            # discarded momentum output's incoming gradient.
            grad_q = torch.zeros_like(grad_p)

        param_grads: dict[int, torch.Tensor] = {}

        for attn_part, mlp_part in reversed(fns):
            # Forward step was:
            #   q_new = q_old + attn_part(p_old)
            #   p_new = p_old + mlp_part(q_new)
            # Live (p, q) here are (p_new, q_new). Reconstruct q_old, p_old.
            # attn_part and mlp_part are both bound methods of the same
            # Block instance, so their __self__.parameters() includes
            # params neither call actually touches (e.g. mlp_part's graph
            # never uses attn's weights). allow_unused=True + filtering
            # None handles that correctly, without either double-counting
            # a param across both calls or coupling this file to Block's
            # internal attribute layout.
            q_new_input = q.detach().requires_grad_(True)
            mlp_params = list(mlp_part.__self__.parameters()) if hasattr(mlp_part, "__self__") else []
            with torch.enable_grad():
                with _recompute_autocast_ctx(ctx.autocast_enabled, ctx.autocast_device_type, ctx.autocast_dtype):
                    mlp_val = mlp_part(q_new_input)  # mlp_part(q_new)

            grads = torch.autograd.grad(
                mlp_val,
                [q_new_input, *mlp_params],
                grad_outputs=grad_p,
                retain_graph=False,
                allow_unused=True,
            )
            grad_q_from_mlp, mlp_param_grads = grads[0], grads[1:]
            for prm, g in zip(mlp_params, mlp_param_grads):
                if g is None:
                    continue
                key = id(prm)
                param_grads[key] = g if key not in param_grads else param_grads[key] + g

            with torch.no_grad():
                p_old = p - mlp_val  # p_new - mlp_part(q_new)

            # d/d(p_new) contributes 1 directly (p_new = p_old + mlp(...)),
            # so grad wrt p_old gets grad_p straight through; grad wrt q_new
            # gets grad_q (direct, since q_new feeds forward unchanged into
            # next layer's q slot) plus grad_q_from_mlp (via mlp_part).
            grad_q_new_total = grad_q + grad_q_from_mlp

            p_new_input = p_old.detach().requires_grad_(True)
            attn_params = list(attn_part.__self__.parameters()) if hasattr(attn_part, "__self__") else []
            with torch.enable_grad():
                with _recompute_autocast_ctx(ctx.autocast_enabled, ctx.autocast_device_type, ctx.autocast_dtype):
                    attn_val = attn_part(p_new_input)  # attn_part(p_old)

            grads = torch.autograd.grad(
                attn_val,
                [p_new_input, *attn_params],
                grad_outputs=grad_q_new_total,
                retain_graph=False,
                allow_unused=True,
            )
            grad_p_from_attn, attn_param_grads = grads[0], grads[1:]
            for prm, g in zip(attn_params, attn_param_grads):
                if g is None:
                    continue
                key = id(prm)
                param_grads[key] = g if key not in param_grads else param_grads[key] + g

            with torch.no_grad():
                q_old = q - attn_val  # q_new - attn_part(p_old)

            # grad wrt p_old accumulates: direct (via p_new = p_old + ...)
            # PLUS via attn_part(p_old) feeding q_new.
            grad_p_old = grad_p + grad_p_from_attn
            grad_q_old = grad_q_new_total  # q_old feeds forward unchanged into q_new

            p, q = p_old, q_old
            grad_p, grad_q = grad_p_old, grad_q_old

        # Loop ends with (p, q) == (p0, p0); total grad wrt the shared input
        # is the sum of both channels.
        grad_p0 = grad_p + grad_q
        param_grad_list = [param_grads.get(id(prm)) for prm in ctx.param_refs]
        return (grad_p0, None, None, *param_grad_list)


class HamiltonianReversibleStack(nn.Module):
    """Wraps a sequence of (attn_part, mlp_part) callable pairs into one
    reversible stack evaluated via the Hamiltonian coupling of Eq. (2.8-2.9).
    Returns only the position stream p (the value fed into ln_f/lm_head),
    matching how model.py consumes the other stacks' single-tensor output."""

    def __init__(self, fns: Sequence[tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]]):
        super().__init__()
        self.fns = list(fns)
        owners = []
        for attn_part, mlp_part in self.fns:
            owners.append(getattr(attn_part, "__self__", None))
            owners.append(getattr(mlp_part, "__self__", None))
        self._param_refs = _collect_param_refs(owners)

    def forward(self, p0: torch.Tensor) -> torch.Tensor:
        p, _q = HamiltonianReversibleFunction.apply(p0, self.fns, self._param_refs, *self._param_refs)
        return p
