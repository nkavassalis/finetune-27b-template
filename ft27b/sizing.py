"""Will it fit on the Mac? Memory for an MLX-quantized model, from its HF config.

Stdlib only: reads config.json and model.safetensors.index.json, nothing else.
Run before training anything you mean to ship to Apple silicon.

    weights   = text params × (bits/8 + 4/group)     MLX keeps an fp16 scale+bias per group
    KV cache  = full-attention layers × 2 × KV heads × head_dim × 2 bytes, per token
    state     = linear-attention layers × V heads × K dim × V dim × 4 bytes  (constant)

Text params = checkpoint bytes / 2 (BF16), minus the vision tower and any MTP
(speculative-decoding) layers, which mlx-lm drops. Calibrated on Qwen3.8-27B:
predicts 28.6 GB at 8-bit; the converted model measured 29.0 GB (27 GiB) and
peaked at 30.2 GB in use with a 838-token prompt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

WORKING_GB = 3.0  # prefill activations, sampler, framework overhead (measured ~1.5-3)


@dataclass
class Shape:
    text_params: float
    full_layers: int
    linear_layers: int
    kv_heads: int
    head_dim: int
    linear_state_bytes: int

    @property
    def kv_bytes_per_token(self) -> int:
        return self.full_layers * 2 * self.kv_heads * self.head_dim * 2


def _vision_params(v: dict) -> float:
    if not v:
        return 0.0
    h, inter, depth = v.get("hidden_size", 0), v.get("intermediate_size", 0), v.get("depth", 0)
    merge = v.get("spatial_merge_size", 1) ** 2
    out = v.get("out_hidden_size", h)
    blocks = depth * (4 * h * h + 2 * h * inter)
    merger = (h * merge) ** 2 + h * merge * out
    return float(blocks + merger)


def _decoder_layer_params(t: dict) -> float:
    h, inter = t["hidden_size"], t.get("intermediate_size", 4 * t["hidden_size"])
    heads, kv, hd = t["num_attention_heads"], t.get("num_key_value_heads"), t.get("head_dim")
    hd = hd or h // heads
    kv = kv or heads
    attn = h * heads * hd * 2 + 2 * h * kv * hd   # q (with gate) + o, k + v
    return float(attn + 3 * h * inter)


def shape(model_dir: str | Path) -> Shape:
    d = Path(model_dir)
    cfg = json.loads((d / "config.json").read_text())
    t = cfg.get("text_config", cfg)
    index = d / "model.safetensors.index.json"
    total = json.loads(index.read_text())["metadata"]["total_size"] if index.exists() else None
    if total is None:
        raise ValueError(f"{d}: need model.safetensors.index.json with metadata.total_size")
    params = float(total) / 2  # BF16 checkpoint
    params -= _vision_params(cfg.get("vision_config", {}))
    mtp = t.get("mtp_num_hidden_layers") or t.get("num_nextn_predict_layers") or 0
    params -= mtp * (_decoder_layer_params(t) + 2 * t["hidden_size"] ** 2)

    n = t["num_hidden_layers"]
    types = t.get("layer_types") or ["full_attention"] * n
    full = sum(x == "full_attention" for x in types)
    linear = n - full
    state = 0
    if linear:
        state = (linear * t.get("linear_num_value_heads", 0) * t.get("linear_key_head_dim", 0)
                 * t.get("linear_value_head_dim", 0) * 4)
    heads = t["num_attention_heads"]
    return Shape(text_params=params, full_layers=full, linear_layers=linear,
                 kv_heads=t.get("num_key_value_heads") or heads,
                 head_dim=t.get("head_dim") or t["hidden_size"] // heads,
                 linear_state_bytes=state)


def weights_gb(s: Shape, bits: int, group: int = 64) -> float:
    return s.text_params * (bits / 8 + 4 / group) / 1e9


def total_gb(s: Shape, bits: int, context: int, group: int = 64) -> float:
    return (weights_gb(s, bits, group) + s.kv_bytes_per_token * context / 1e9
            + s.linear_state_bytes / 1e9 + WORKING_GB)


def table(s: Shape, *, limit_gb: float, bits_list=(4, 6, 8),
          contexts=(4096, 32768, 131072, 262144), group: int = 64) -> str:
    lines = [f"text params {s.text_params / 1e9:.2f}B | {s.full_layers} full-attention + "
             f"{s.linear_layers} linear layers | KV {s.kv_bytes_per_token / 1024:.0f} KB/token | "
             f"recurrent state {s.linear_state_bytes / 1e9:.2f} GB",
             f"budget: {limit_gb:.0f} GB the GPU may wire (macOS: ~75% of RAM by default; "
             "raise with sudo sysctl iogpu.wired_limit_mb=<MB>)", ""]
    head = "bits | weights | " + " | ".join(f"{c // 1024}k ctx" for c in contexts)
    lines += [head, "-" * len(head)]
    for b in bits_list:
        cells = []
        for c in contexts:
            gb = total_gb(s, b, c, group)
            cells.append(f"{gb:5.1f} GB {'ok' if gb <= limit_gb else 'NO'}")
        lines.append(f"{b:>4} | {weights_gb(s, b, group):5.1f} GB | " + " | ".join(cells))
    return "\n".join(lines)
