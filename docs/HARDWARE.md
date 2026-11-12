# Hardware: what fits where, measured

## Training memory for a 27B

| approach | memory | verdict |
| --- | --- | --- |
| full fine-tune, BF16 + AdamW | weights 54 + grads 54 + **AdamW 324** (12 B/param: fp32 master, momentum, variance) + activations 20–60 ≈ **450–490 GB** | not on two 121 GB nodes; with offload, a throughput disaster |
| **BF16 LoRA** (default) | weights 54.7 GB + LoRA/optimizer ~1.5 GB + activations → **71.5–72 GB peak at 4,096 tokens** (measured) | fits one GB10 node with headroom |
| QLoRA (4-bit base) | ~30–50 GB | needs bitsandbytes for your platform; untested on GB10 |

"Full fine-tune ≈ 3–4× the weights" only holds for 8-bit optimizers or
Adafactor. With AdamW it's ~9×. Check this arithmetic before planning
hardware.

## NVIDIA GB10 (DGX Spark), measured on lkml-reviewer

- 121 GiB **unified** memory per node, shared by GPU, CPU and page cache.
  **Cap the allocator** (`[train] mem_fraction = 0.75`) and set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Uncapped, the reserved
  pool plus fragmentation froze a node with only 72 GB *allocated*. With the cap,
  42–55 GB stayed available for the whole run.
- BF16 LoRA on a 27B: **~163 tok/s** on one node, 6 min model load.
- **Two nodes, data parallel** (`train/run-chain-2node.sh`, torchrun):
  **258–296 tok/s, 1.6–1.7×**. NCCL over plain TCP on the 200 Gb link is
  enough: LoRA syncs only ~435 MB of gradients once per optimizer step (no RDMA
  device needed in the container).
- **When to use two nodes:** when a run would take more than ~half a day on
  one. Below that, the second node's setup (model copy, container, NCCL) isn't
  worth it. Keep the effective batch fixed across 1 and 2 nodes
  (`[train] effective_batch`) so runs stay comparable.
- Generation with HF `generate` is bandwidth-bound: batching 16 sequences
  costs about the same per step as 8. Cap new tokens at what the task needs.

## Apple silicon (the shipping target)

See `docs/CONVERSION.md` §1 and `ft27b size`. Summary for a 27B on a 64 GB
M4 Pro: 8-bit weights ~28.6 GB, ~30 GB in use at 4k context, ~8.5 tok/s;
macOS lets the GPU wire ~75% of RAM by default (raise with
`sudo sysctl iogpu.wired_limit_mb=<MB>`).

## Before you pick a different base model

1. `ft27b size <hf-model-dir>`: does the 8-bit conversion fit the Mac?
2. Check that mlx-lm supports its architecture (`mlx_lm/models/<model_type>.py`).
3. `ft27b targets`: list its Linear layers, then write `[model] lora_targets`.
4. Check its chat template for a thinking mode, and set `enable_thinking`.
