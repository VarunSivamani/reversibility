"""Orchestrates the full comparison for one architecture (dense or moe):

  1. Train all 4 block types (euler, midpoint, leapfrog, hamiltonian) at a
     fixed batch size, for the same token budget.
  2. Pick the winner: lowest final validation loss.
  3. Re-run the winning block type with an auto-maximized batch size
     (binary search floored at the fixed batch size from step 1, since
     that size is already known to fit).
  4. Write a combined pipeline_summary.json with all 5 runs plus the
     comparison table.

Usage (from the repo root):

    python -m reversibility.pipeline --arch dense --batch_size 32 --max_tokens 50000000 --out_dir runs/dense
    python -m reversibility.pipeline --arch moe   --batch_size 32 --max_tokens 50000000 --out_dir runs/moe

Run both, then use --compare to print a dense-vs-moe table from the two
resulting pipeline_summary.json files:

    python -m reversibility.pipeline --compare runs/dense/pipeline_summary.json runs/moe/pipeline_summary.json
"""

from __future__ import annotations

import argparse
import json
import os

from .model import GPTConfig
from .train import BLOCK_TYPES, DEVICE, find_max_batch_size, train_one_run


def run_pipeline(
    arch: str,
    batch_size: int,
    max_tokens: int,
    out_dir: str,
    data_dir: str,
    vocab_size: int,
    block_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    step_h: float,
    n_experts: int,
    expert_hidden_mult: float,
    moe_top_k: int,
    moe_aux_loss_coef: float,
    lr: float,
    warmup_steps: int,
    log_every: int,
    use_compile: bool,
) -> dict:
    ffn_type = "moe" if arch == "moe" else "dense"
    os.makedirs(out_dir, exist_ok=True)

    def make_cfg(block_type: str) -> GPTConfig:
        return GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=0.0,
            block_type=block_type,
            step_h=step_h,
            ffn_type=ffn_type,
            n_experts=n_experts,
            expert_hidden_mult=expert_hidden_mult,
            moe_top_k=moe_top_k,
            moe_aux_loss_coef=moe_aux_loss_coef,
        )

    variant_summaries: dict[str, dict] = {}
    for block_type in BLOCK_TYPES:
        run_name = f"{arch}_{block_type}"
        print(f"\n=== [{arch}] running block_type={block_type} at fixed batch_size={batch_size} ===")
        cfg = make_cfg(block_type)
        summary = train_one_run(
            cfg=cfg,
            batch_size=batch_size,
            max_tokens=max_tokens,
            run_name=run_name,
            out_dir=out_dir,
            data_dir=data_dir,
            lr=lr,
            warmup_steps=warmup_steps,
            log_every=log_every,
            use_compile=use_compile,
        )
        variant_summaries[block_type] = summary

    winner = min(variant_summaries, key=lambda bt: variant_summaries[bt]["final_val_loss"])
    print(f"\n=== [{arch}] winner: {winner} (val loss {variant_summaries[winner]['final_val_loss']:.4f}) ===")

    print(f"\n=== [{arch}] searching max batch size for winner={winner}, floor={batch_size} ===")
    winner_cfg = make_cfg(winner)
    max_batch_size = find_max_batch_size(winner_cfg, block_size, DEVICE, start=batch_size)
    print(f"[{arch}] max batch size found: {max_batch_size}")

    max_batch_run_name = f"{arch}_{winner}_maxbatch"
    max_batch_summary = train_one_run(
        cfg=winner_cfg,
        batch_size=max_batch_size,
        max_tokens=max_tokens,
        run_name=max_batch_run_name,
        out_dir=out_dir,
        data_dir=data_dir,
        lr=lr,
        warmup_steps=warmup_steps,
        log_every=log_every,
        use_compile=use_compile,
    )

    pipeline_summary = {
        "arch": arch,
        "ffn_type": ffn_type,
        "fixed_batch_size": batch_size,
        "max_tokens": max_tokens,
        "variant_summaries": variant_summaries,
        "winner_block_type": winner,
        "winner_val_loss_fixed_batch": variant_summaries[winner]["final_val_loss"],
        "max_batch_size_found": max_batch_size,
        "max_batch_summary": max_batch_summary,
        "comparison_table": {
            bt: {
                "final_val_loss": s["final_val_loss"],
                "final_train_loss": s["final_train_loss"],
                "tokens_per_s_steady": s["tokens_per_s_steady"],
                "peak_gpu_metrics": s["peak_gpu_metrics"],
            }
            for bt, s in variant_summaries.items()
        },
    }
    summary_path = os.path.join(out_dir, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(pipeline_summary, f, indent=2)
    print(f"\n[{arch}] pipeline summary written to {summary_path}")
    return pipeline_summary


def print_dense_vs_moe(dense_summary_path: str, moe_summary_path: str) -> None:
    with open(dense_summary_path) as f:
        dense = json.load(f)
    with open(moe_summary_path) as f:
        moe = json.load(f)

    print("\n=== Dense vs MoE comparison ===")
    print(f"{'metric':30s} {'dense (winner=' + dense['winner_block_type'] + ')':30s} {'moe (winner=' + moe['winner_block_type'] + ')':30s}")
    rows = [
        ("fixed-batch val loss", dense["winner_val_loss_fixed_batch"], moe["winner_val_loss_fixed_batch"]),
        ("max batch size found", dense["max_batch_size_found"], moe["max_batch_size_found"]),
        ("max-batch val loss", dense["max_batch_summary"]["final_val_loss"], moe["max_batch_summary"]["final_val_loss"]),
        ("max-batch tokens/s", dense["max_batch_summary"]["tokens_per_s_steady"], moe["max_batch_summary"]["tokens_per_s_steady"]),
        (
            "max-batch peak mem (MB)",
            dense["max_batch_summary"]["peak_gpu_metrics"].get("cuda_max_mem_allocated_mb"),
            moe["max_batch_summary"]["peak_gpu_metrics"].get("cuda_max_mem_allocated_mb"),
        ),
    ]
    for name, d_val, m_val in rows:
        print(f"{name:30s} {str(d_val):30s} {str(m_val):30s}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["dense", "moe"], help="required unless --compare is given")
    p.add_argument("--compare", nargs=2, metavar=("DENSE_SUMMARY_JSON", "MOE_SUMMARY_JSON"))

    p.add_argument("--batch_size", type=int, default=32, help="fixed batch size for the 4-variant comparison; also the search floor for the max-batch run")
    p.add_argument("--max_tokens", type=int, default=50_000_000)
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--out_dir", type=str, default="runs")

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

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.compare:
        print_dense_vs_moe(*args.compare)
        return

    if not args.arch:
        raise SystemExit("either --arch dense|moe or --compare DENSE_JSON MOE_JSON is required")

    run_pipeline(
        arch=args.arch,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        out_dir=args.out_dir,
        data_dir=args.data_dir,
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        step_h=args.step_h,
        n_experts=args.n_experts,
        expert_hidden_mult=args.expert_hidden_mult,
        moe_top_k=args.moe_top_k,
        moe_aux_loss_coef=args.moe_aux_loss_coef,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        use_compile=not args.no_compile,
    )


if __name__ == "__main__":
    main()
