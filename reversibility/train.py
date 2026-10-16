"""Train the ~20M-param GPT (Dense or MoE FFN) on the prepared token stream,
for a fixed token budget, in one of four reversible block types.

Core entry point is train_one_run(cfg, batch_size, max_tokens, ...), which
pipeline.py calls directly (in-process, no subprocess) to run the full
euler/midpoint/leapfrog/hamiltonian comparison and the max-batch-size
re-run. train.py also exposes a standalone CLI for running one config at a
time.

Usage (single run, from the repo root):

    python -m reversibility.train --ffn_type dense --block_type euler \
        --batch_size 32 --max_tokens 50000000 --data_dir data --out_dir runs

    python -m reversibility.train --ffn_type moe --block_type hamiltonian \
        --batch_size 32 --max_tokens 50000000 --data_dir data --out_dir runs

    python -m reversibility.train --ffn_type dense --block_type midpoint \
        --auto_max_batch --batch_size 32 --max_tokens 50000000 \
        --data_dir data --out_dir runs
        # --batch_size here is the known-good floor the binary search starts
        # from (see find_max_batch_size), not the batch size actually used.

Logs per-step loss/throughput/GPU-telemetry to <out_dir>/<run_name>.jsonl
and a summary to <out_dir>/<run_name>_summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from .gpu_metrics import gpu_name, gpu_snapshot, nvml_available
from .model import GPT, GPTConfig

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
BLOCK_TYPES = ["euler", "midpoint", "leapfrog", "hamiltonian"]


def load_bin(path: str) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data: np.memmap, batch_size: int, block_size: int, device: str):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def reset_memory_stats(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def build_optimizer(model: GPT, lr: float, device: str) -> torch.optim.Optimizer:
    use_fused = device == "cuda"
    try:
        return torch.optim.AdamW(
            model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1, fused=use_fused
        )
    except (TypeError, RuntimeError):
        # Older torch build, or fused kernel unavailable for this dtype/device.
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)


def compile_submodules(model: GPT, enabled: bool) -> None:
    """Applies torch.compile to each block's attention and FFN sub-modules
    individually, leaving the reversible autograd Functions' manual
    forward/backward orchestration (the no_grad loop, the explicit
    torch.autograd.grad calls) running in eager mode -- compiling around
    that custom control flow is unsupported/unreliable, but the actual
    heavy compute inside each f_theta call still benefits from fusion."""
    if not enabled or not hasattr(torch, "compile"):
        return
    for block in model.blocks:
        try:
            block.attn = torch.compile(block.attn)
            block.mlp = torch.compile(block.mlp)
        except Exception as e:  # noqa: BLE001 - never let compile failure abort training
            print(f"[compile] skipping sub-module compile ({e})")
            return


@torch.no_grad()
def estimate_val_loss(model: GPT, val_data: np.memmap, block_size: int, batch_size: int, device: str, autocast_dtype, iters: int = 50) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(val_data, batch_size, block_size, device)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype) if device == "cuda" else _nullcontext():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def find_max_batch_size(
    cfg: GPTConfig,
    block_size: int,
    device: str,
    start: int,
    cap: int = 4096,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Doubling search followed by a bisection, for the largest batch size
    that completes one forward+backward step without OOM. `start` should be
    a batch size already known to fit (e.g. the fixed batch size used in the
    variant-comparison runs), so the search only explores upward from a safe
    floor instead of re-verifying small sizes."""

    def fits(bs: int) -> bool:
        try:
            reset_memory_stats(device)
            model = GPT(cfg).to(device)
            opt = build_optimizer(model, 1e-3, device)
            x = torch.randint(0, cfg.vocab_size, (bs, block_size), device=device)
            y = torch.randint(0, cfg.vocab_size, (bs, block_size), device=device)
            # Must match train_one_run: forward under autocast on CUDA, else
            # the forced-flash attention sees fp32 q/k/v and fails with
            # "No available kernel" (flash requires fp16/bf16).
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            loss.backward()
            opt.step()
            del model, opt, x, y, loss
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            return True
        except torch.OutOfMemoryError:  # type: ignore[attr-defined]
            if device == "cuda":
                torch.cuda.empty_cache()
            return False
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if device == "cuda":
                    torch.cuda.empty_cache()
                return False
            raise

    if not fits(start):
        raise RuntimeError(
            f"batch size {start} (the known-good floor) does not fit; "
            "cannot search upward from an invalid starting point"
        )

    # Doubling phase: lo is always known to fit; hi is the first size tried
    # that does NOT fit (or `cap` if we hit the cap while still fitting).
    lo = start
    hi = start * 2
    while hi <= cap and fits(hi):
        lo = hi
        hi *= 2
    if hi > cap:
        hi = cap
        if fits(hi):
            return hi

    # Bisection phase: lo fits, hi does not (or hi == cap and doesn't fit).
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo


def train_one_run(
    cfg: GPTConfig,
    batch_size: int,
    max_tokens: int,
    run_name: str,
    out_dir: str,
    data_dir: str = "data",
    lr: float = 3e-4,
    warmup_steps: int = 20,
    log_every: int = 50,
    use_compile: bool = True,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Runs one training config for `max_tokens` tokens, logs per-step
    metrics, and returns the summary dict (also written to
    <out_dir>/<run_name>_summary.json)."""

    device = DEVICE
    print(f"[{run_name}] device: {device}  ffn_type: {cfg.ffn_type}  block_type: {cfg.block_type}  batch_size: {batch_size}")

    train_data = load_bin(os.path.join(data_dir, "train.bin"))
    val_data = load_bin(os.path.join(data_dir, "val.bin"))

    reset_memory_stats(device)
    model = GPT(cfg).to(device)
    compile_submodules(model, enabled=use_compile and device == "cuda")
    print(f"[{run_name}] model params: {model.num_params():,}")

    opt = build_optimizer(model, lr, device)

    tokens_per_step = batch_size * cfg.block_size
    max_steps = max_tokens // tokens_per_step
    if max_steps < 1:
        raise ValueError(
            f"[{run_name}] batch_size={batch_size} x block_size={cfg.block_size} = "
            f"{tokens_per_step:,} tokens/step exceeds max_tokens={max_tokens:,}; "
            "would run 0 training steps. Increase --max_tokens or reduce batch_size/block_size."
        )
    print(f"[{run_name}] tokens/step: {tokens_per_step:,}  steps for {max_tokens:,} tokens: {max_steps:,}")

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"{run_name}.jsonl")
    log_file = open(log_path, "w")

    tokens_seen = 0
    t_start = time.time()
    t_warm = None
    tokens_at_warm = 0
    running_loss = 0.0
    peak_gpu_snapshot: dict[str, float] = {}

    autocast_ctx = (
        (lambda: torch.autocast(device_type="cuda", dtype=autocast_dtype))
        if device == "cuda"
        else (lambda: _nullcontext())
    )

    for step in range(1, max_steps + 1):
        x, y = get_batch(train_data, batch_size, cfg.block_size, device)
        with autocast_ctx():
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        tokens_seen += tokens_per_step
        running_loss = loss.item() if step == 1 else 0.98 * running_loss + 0.02 * loss.item()

        if step == warmup_steps:
            t_warm = time.time()
            tokens_at_warm = tokens_seen

        if step % log_every == 0 or step == max_steps:
            elapsed = time.time() - t_start
            if t_warm is not None:
                steady_tok_s = (tokens_seen - tokens_at_warm) / max(time.time() - t_warm, 1e-9)
            else:
                steady_tok_s = tokens_seen / elapsed

            snap = gpu_snapshot()
            for k, v in snap.items():
                if v == v:  # not NaN
                    peak_gpu_snapshot[k] = max(peak_gpu_snapshot.get(k, v), v)

            rec = {
                "step": step,
                "tokens_seen": tokens_seen,
                "loss": loss.item(),
                "running_loss": running_loss,
                "elapsed_s": elapsed,
                "tokens_per_s": steady_tok_s,
                **{f"gpu_{k}": v for k, v in snap.items()},
            }
            log_file.write(json.dumps(rec) + "\n")
            log_file.flush()
            print(
                f"[{run_name}] step {step}/{max_steps} tokens {tokens_seen:,} loss {loss.item():.4f} "
                f"tok/s {steady_tok_s:,.0f} gpu_util {snap['gpu_util_pct']:.0f}% "
                f"mem_alloc {snap['cuda_mem_allocated_mb']:.0f}MB power {snap['power_draw_w']:.0f}W"
            )

    total_elapsed = time.time() - t_start
    val_loss = estimate_val_loss(model, val_data, cfg.block_size, min(batch_size, 32), device, autocast_dtype)
    final_tok_s = (tokens_seen - tokens_at_warm) / max(time.time() - t_warm, 1e-9) if t_warm else tokens_seen / total_elapsed

    final_snap = gpu_snapshot()
    for k, v in final_snap.items():
        if v == v:
            peak_gpu_snapshot[k] = max(peak_gpu_snapshot.get(k, v), v)

    summary = {
        "run_name": run_name,
        "ffn_type": cfg.ffn_type,
        "block_type": cfg.block_type,
        "batch_size": batch_size,
        "block_size": cfg.block_size,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "n_experts": cfg.n_experts if cfg.ffn_type == "moe" else None,
        "moe_top_k": cfg.moe_top_k if cfg.ffn_type == "moe" else None,
        "step_h": cfg.step_h if cfg.block_type in ("midpoint", "leapfrog") else None,
        "model_params": model.num_params(),
        "max_tokens": max_tokens,
        "tokens_seen": tokens_seen,
        "steps": max_steps,
        "final_train_loss": running_loss,
        "final_val_loss": val_loss,
        "tokens_per_s_steady": final_tok_s,
        "total_wallclock_s": total_elapsed,
        "device": device,
        "gpu_name": gpu_name() if nvml_available() else device,
        "peak_gpu_metrics": peak_gpu_snapshot,
    }
    summary_path = os.path.join(out_dir, f"{run_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log_file.close()

    print(f"[{run_name}] === SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"[{run_name}] written to {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ffn_type", choices=["dense", "moe"], default="dense")
    p.add_argument("--block_type", choices=BLOCK_TYPES, default="euler")
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="fixed batch size; with --auto_max_batch, used as the known-good floor for the search",
    )
    p.add_argument("--auto_max_batch", action="store_true", help="binary-search the max batch size, starting from --batch_size")
    p.add_argument("--max_tokens", type=int, default=50_000_000)
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--run_name", type=str, default=None)

    p.add_argument("--vocab_size", type=int, default=50257)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--step_h", type=float, default=0.5)

    p.add_argument("--n_experts", type=int, default=8)
    p.add_argument("--expert_hidden_mult", type=float, default=0.5)
    p.add_argument("--moe_top_k", type=int, default=2)
    p.add_argument("--moe_aux_loss_coef", type=float, default=0.01)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--force_flash_attention", action="store_true", default=True)

    args = p.parse_args()
    if args.run_name is None:
        args.run_name = f"{args.ffn_type}_{args.block_type}"
    return args


def main() -> None:
    args = parse_args()
    cfg = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,
        block_type=args.block_type,
        step_h=args.step_h,
        ffn_type=args.ffn_type,
        n_experts=args.n_experts,
        expert_hidden_mult=args.expert_hidden_mult,
        moe_top_k=args.moe_top_k,
        moe_aux_loss_coef=args.moe_aux_loss_coef,
        force_flash_attention=args.force_flash_attention,
    )

    batch_size = args.batch_size
    if args.auto_max_batch:
        print("searching for max batch size...")
        batch_size = find_max_batch_size(cfg, args.block_size, DEVICE, start=args.batch_size)
        print(f"max batch size found: {batch_size}")

    train_one_run(
        cfg=cfg,
        batch_size=batch_size,
        max_tokens=args.max_tokens,
        run_name=args.run_name,
        out_dir=args.out_dir,
        data_dir=args.data_dir,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        use_compile=not args.no_compile,
    )


if __name__ == "__main__":
    main()
