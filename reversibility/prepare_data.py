"""Build a fixed 50M-token (+ small val split) training corpus from a
FineWeb-edu sample, tokenized with GPT-2 BPE (tiktoken) and packed into a
nanoGPT-style flat uint16 .bin file, so all three training runs (baseline,
reversible fixed-batch, reversible max-batch) see exactly the same data.

Usage (from the repo root):
    python -m reversibility.prepare_data --tokens 50_000_000 --out_dir data
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import tiktoken
from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=50_000_000)
    parser.add_argument("--val_fraction", type=float, default=0.005)
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="HF dataset id to stream from (falls back to openwebtext on failure).",
    )
    parser.add_argument("--dataset_config", type=str, default="sample-10BT")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    n_val = int(args.tokens * args.val_fraction)
    n_train = args.tokens - n_val
    total = n_train + n_val

    print(f"target tokens: train={n_train:,} val={n_val:,} total={total:,}")

    try:
        ds = load_dataset(args.dataset, name=args.dataset_config, split="train", streaming=True)
        print(f"streaming from {args.dataset} ({args.dataset_config})")
    except Exception as e:  # noqa: BLE001 - fall back cleanly for a Colab/offline-flaky dataset id
        print(f"primary dataset failed ({e}); falling back to openwebtext")
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=True)

    all_ids = np.empty(total, dtype=np.uint16)
    filled = 0
    for example in ds:
        text = example.get("text")
        if not text:
            continue
        ids = enc.encode_ordinary(text)
        ids.append(eot)
        n = len(ids)
        if filled + n > total:
            n = total - filled
            ids = ids[:n]
        all_ids[filled : filled + n] = np.array(ids, dtype=np.uint16)
        filled += n
        if filled >= total:
            break
        if filled % 5_000_000 < n:
            print(f"  tokenized {filled:,}/{total:,}")

    if filled < total:
        raise RuntimeError(f"dataset exhausted before reaching {total:,} tokens (got {filled:,})")

    train_ids = all_ids[:n_train]
    val_ids = all_ids[n_train:total]

    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    print(f"wrote {train_path} ({train_ids.nbytes / 1e6:.1f} MB, {len(train_ids):,} tokens)")
    print(f"wrote {val_path} ({val_ids.nbytes / 1e6:.1f} MB, {len(val_ids):,} tokens)")


if __name__ == "__main__":
    main()
