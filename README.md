# Reversible Transformers: Dense vs. MoE

A from-scratch comparison of four transformer update rules — one standard
residual baseline and three analytically reversible variants — trained on
two ~20M-parameter architectures (Dense and Mixture-of-Experts) for 50M
tokens each on an NVIDIA H200 NVL. The reversible variants use a custom
`torch.autograd.Function` that reconstructs activations during the backward
pass instead of caching them, decoupling activation memory from model depth.

Full interactive results, charts, and discussion: **[`docs/index.html`](docs/index.html)**.

## Results at a glance

| | Dense | MoE |
|---|---|---|
| Parameters | 19,292,672 | 19,833,344 |
| Winning variant (fixed batch) | **Midpoint** | **Leapfrog** |
| Fixed-batch (32) validation loss | 5.0426 | 4.9733 |
| Fixed-batch throughput | ~507K tok/s | ~127–149K tok/s |
| Max batch size found | 383 | 383 |
| Max-batch validation loss | 6.1510 | 6.1744 |
| Max-batch throughput | ~577K tok/s | ~285K tok/s |
| Max-batch peak GPU memory | 116.3 GB | 116.3 GB |
| Hardware | NVIDIA H200 NVL | NVIDIA H200 NVL |

Every reversible variant matched or beat the non-reversible baseline on
final validation loss, on both architectures, at equal token budget. MoE
reached the lowest loss overall but at roughly a third of Dense's
throughput on this scale — see [Findings](#findings) for why, and for why
the max-batch runs converge worse rather than better.

## Architecture

Two backbones, same depth and width, differing only in the feed-forward
layer:

| | `n_layer` | `n_head` | `n_embd` | `block_size` | Vocabulary | FFN |
|---|---|---|---|---|---|---|
| **Dense** | 8 | 8 | 256 | 512 | 50,257 (GPT-2 BPE) | one MLP per block |
| **MoE** | 8 | 8 | 256 | 512 | 50,257 (GPT-2 BPE) | 8 routed experts (top-2) + 1 shared expert |

The MoE feed-forward layer follows the DeepSeek-V3/V4 pattern: a linear
router selects the top-2 of 8 small experts per token, renormalizes their
weights, and adds one always-on shared expert unconditionally. A
Switch-Transformer-style load-balancing loss is folded into the LM loss
automatically. Expert hidden size is deliberately smaller than Dense's 4×
expansion (`expert_hidden_mult=0.5`), since all 8 experts' parameters count
toward total model size even though only 2 + shared are active per token —
this keeps the two architectures within 3% of each other in parameter
count for a fair comparison.

## The four update rules

| Variant | Update | Reversible |
|---|---|---|
| **Euler** | `p_{l+1} = p_l + f_l(p_l)` | No — ordinary residual stack, full activation caching. The non-reversible baseline. |
| **Midpoint** | `p_{l+1} = p_{l-1} + 2h·f_l(p_l)` | Yes — first-order dynamics. |
| **Leapfrog** | `p_{l+1} = 2p_l - p_{l-1} + h²·f_l(p_l)` | Yes — second-order (wave-equation) dynamics; more stable under oscillatory Jacobians than Midpoint. |
| **Hamiltonian** | `q_l = q_{l-1} + Attn(LN1(p_{l-1}))`, `p_l = p_{l-1} + MLP(LN2(q_l))` | Yes — two coupled streams (position/momentum), conserves a discrete energy analogue. |

Each reversible variant is implemented as its own hand-written
`torch.autograd.Function` (`reversibility/reversible.py`): the forward pass runs under
`torch.no_grad()` and saves only the boundary state(s) that enter and leave
the stack; the backward pass recomputes each layer's local graph on demand
via `torch.autograd.grad` and algebraically inverts the update to
reconstruct the previous state. No per-layer activation is ever cached —
memory for the reversible stack is therefore independent of `n_layer`. This
is distinct from `torch.utils.checkpoint`, which still stores each
checkpointed segment's input and reruns it forward; here nothing is stored
between the two boundary tensors.

Dropout is forced to zero whenever a reversible variant is selected — the
backward pass depends on recomputing each layer deterministically, and a
random dropout mask would make that reconstruction diverge from the
original forward pass.

Correctness of all three reversible Functions was verified before any GPU
time was spent (`tests/test_reversible.py`, run with
`python -m tests.test_reversible`): each custom backward was checked
against a plain-autograd reference implementing the identical recursion,
matching forward output, input gradient, and every per-layer parameter
gradient to within ~1e-8–1e-9.

## Pipeline

`pipeline.py` runs, per architecture:

1. Train all four block types (`euler`, `midpoint`, `leapfrog`,
   `hamiltonian`) at a fixed batch size for the full token budget.
2. Select the winner by lowest final validation loss.
3. Binary-search the largest batch size that fits in GPU memory
   (`find_max_batch_size`), floored at the fixed batch size already known
   to fit.
4. Retrain the winning variant at that batch size.
5. Write a consolidated `pipeline_summary.json` with all five runs, the
   winner, and a comparison table.

```bash
pip install -r requirements.txt
# install torch matching your CUDA version, e.g.:
#   pip install torch --index-url https://download.pytorch.org/whl/cu124

python -m reversibility.prepare_data --tokens 50000000 --out_dir data

python -m reversibility.pipeline --arch dense --batch_size 32 --max_tokens 50000000 --out_dir runs/dense
python -m reversibility.pipeline --arch moe   --batch_size 32 --max_tokens 50000000 --out_dir runs/moe

python -m reversibility.pipeline --compare runs/dense/pipeline_summary.json runs/moe/pipeline_summary.json
```

A single configuration can also be run directly (all commands run from the repo root):

```bash
python -m reversibility.train --ffn_type moe --block_type hamiltonian \
    --batch_size 32 --max_tokens 50000000 --data_dir data --out_dir runs

python -m reversibility.train --ffn_type dense --block_type midpoint --auto_max_batch \
    --batch_size 32 --max_tokens 50000000 --data_dir data --out_dir runs
```

## Throughput engineering

- **FlashAttention forced and verified.** `CausalSelfAttention` wraps
  `F.scaled_dot_product_attention` in
  `torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION])` rather than
  relying on automatic backend selection, so a shape or dtype mismatch
  fails loudly instead of silently falling back to a slower kernel. Flash
  requires fp16/bf16 query/key/value tensors; this is what the bf16
  autocast context below guarantees.
- **bf16 mixed precision.** The forward pass (including the reversible
  backward's recomputation) runs under
  `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. No
  `GradScaler` is needed with bf16.
- **Fused AdamW** (`fused=True`) on CUDA.
- **`torch.compile`** is applied to each block's attention and
  feed-forward sub-modules individually — never to the reversible stacks'
  autograd Functions themselves. Compiling around their hand-written
  `torch.no_grad()` / `torch.autograd.grad` control flow is unsupported;
  keeping that path eager preserves the verified-correct backward while
  still letting the actual matmul/attention/expert compute benefit from
  kernel fusion.

## Telemetry

Every run logs, every 50 steps and in the final summary: loss, steady-state
tokens/second, and — via NVML (`gpu_metrics.py`) — GPU utilization,
memory-controller utilization, power draw, temperature, SM/memory clocks,
plus `torch.cuda` allocated/reserved/peak memory. Per-step records live in
`runs/<arch>/<arch>_<variant>.jsonl`; final summaries in
`runs/<arch>/<arch>_<variant>_summary.json` and the consolidated
`runs/<arch>/pipeline_summary.json`.

## Findings

**Reversibility did not cost accuracy.** On Dense, Midpoint (5.0426) edged
out Euler (5.0508); on MoE, Leapfrog (4.9733) beat Euler (5.0034) by a
wider margin. Neither architecture saw the reversible variants lag the
non-reversible baseline at equal token budget.

**MoE reached a lower loss than Dense, at roughly a third of the
throughput.** MoE's best variant (4.9733) beat Dense's best (5.0426), but
Dense trained at ~507–510K tokens/s against MoE's ~127–149K tokens/s on
identical hardware. At this scale — 8 experts of hidden size 128 — there
isn't enough matmul work per expert to hide routing/gather/scatter
overhead in a from-scratch (non-fused) MoE FFN. A production grouped-GEMM
MoE kernel would close much of this gap; the quality gain here comes with
a real, current throughput cost.

**The max-batch runs converged worse, not better — expected, not a
defect.** Both architectures found an identical max batch size (383), and
both saw validation loss rise at that size (Dense: 5.04 → 6.15; MoE:
4.97 → 6.17). The cause is the fixed 50M-token budget, not the update rule:
batch 32 completes 3,051 optimizer steps to cover it; batch 383 completes
only 254. Twelve times the batch size with roughly a twelfth of the weight
updates, under an unchanged learning-rate schedule, under-trains by
construction. A production large-batch run would pair the larger batch
with a scaled learning rate and/or a proportionally larger token budget.

**Two real autograd bugs were caught before spending GPU time**, both the
same class of error — a custom backward's `torch.autograd.grad` call
assumed every parameter it queried was actually used in that call's graph:

- *Hamiltonian*: `attn_part` and `mlp_part` are bound methods of the same
  `Block` instance, so naively collecting
  `mlp_part.__self__.parameters()` pulled in attention weights the
  MLP-half's graph never touches.
- *Midpoint / Leapfrog, with MoE*: top-k routing means some experts
  legitimately receive zero tokens in a given batch, so their weights have
  no gradient path through that layer's call.

Both fixed with `allow_unused=True`, filtering `None` before accumulating
— a parameter with no gradient this step is a legitimate zero, not an
error.

## Files

| File | Purpose |
|---|---|
| `reversibility/model.py` | `GPT` model; `GPTConfig.block_type` selects the update rule, `GPTConfig.ffn_type` selects Dense or MoE. Forces the FlashAttention SDPA backend. |
| `reversibility/reversible.py` | Custom `torch.autograd.Function` per reversible variant, plus the non-reversible `EulerReversibleStack`. |
| `reversibility/train.py` | `train_one_run(cfg, batch_size, max_tokens, ...)` — the training loop, plus a standalone CLI and `find_max_batch_size`. |
| `reversibility/pipeline.py` | Orchestrates the 4-variant comparison and max-batch re-run per architecture; `--compare` prints the Dense-vs-MoE table. |
| `reversibility/gpu_metrics.py` | NVML telemetry wrapper. |
| `reversibility/prepare_data.py` | Streams FineWeb-edu (falls back to OpenWebText), tokenizes with GPT-2 BPE, writes `train.bin` / `val.bin`. |
| `tests/test_reversible.py` | Gradient-correctness harness for the three reversible Functions (`python -m tests.test_reversible`). |
| `requirements.txt` | Dependencies (`torch` installed separately, matched to your CUDA version). |
| `data/` | Tokenized corpus (`train.bin` / `val.bin`) shared by every run reported here. |
| `runs/` | Per-step logs and summary JSONs for every run (`dense/`, `moe/`); `results/comparison.txt` is the plain-text Dense-vs-MoE table. |
| `docs/index.html` | Interactive results report — charts, tables, and the full write-up this README summarizes. |
