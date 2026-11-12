"""Generation backends. Heavy imports happen inside each backend, so the rest of
ft27b (and its tests) need only the stdlib.

  cuda  base model + PEFT adapter, transformers, NVIDIA GPU (BF16, or NF4)
  mlx   a merged + quantized MLX model (scripts/convert-mlx.sh), Apple silicon

Both render prompts with ft27b.template and sample with the same settings from
project.toml [eval]. Greedy decoding is not offered: on lkml-reviewer it
looped on 44% of outputs.
"""
from __future__ import annotations

import sys

from .config import Config
from .template import prompt_text

BF16_NEEDS_GB = 60
NF4_NEEDS_GB = 22


class BackendError(RuntimeError):
    """A setup problem with an actionable message."""


def sampling(cfg: Config) -> dict:
    e = cfg["eval"]
    return {"temperature": e.get("temperature", 0.7), "top_p": e.get("top_p", 0.95),
            "repetition_penalty": e.get("repetition_penalty", 1.1)}


class CudaBackend:
    def __init__(self, cfg: Config, adapter: str | None, *, load_in_4bit: bool = False,
                 seed: int | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoTokenizer
        except ImportError as e:
            raise BackendError(f"missing {e.name}: pip install 'ft27b[cuda]'") from e
        if not torch.cuda.is_available():
            raise BackendError("no CUDA GPU visible to torch (on a Mac, use --backend mlx)")
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        need = NF4_NEEDS_GB if load_in_4bit else BF16_NEEDS_GB
        if gb < need:
            raise BackendError(f"GPU has {gb:.0f} GB; this needs ~{need} GB"
                               + ("" if load_in_4bit else " (try --load-in-4bit)"))
        kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0}}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        self.torch, self.cfg, self.seed = torch, cfg, seed
        self.tok = AutoTokenizer.from_pretrained(cfg.base, padding_side="left")
        model = AutoModelForImageTextToText.from_pretrained(cfg.base, **kwargs)
        if adapter:
            from peft import PeftModel
            try:
                model = PeftModel.from_pretrained(model, adapter)
            except (OSError, ValueError) as e:
                raise BackendError(f"could not load adapter {adapter!r}: {e}") from e
        self.model = model.eval()

    def generate(self, users: list[str], *, batch: int = 4, max_new_tokens: int = 1024) -> list[str]:
        torch, out = self.torch, []
        for i in range(0, len(users), batch):
            prompts = [prompt_text(self.tok, self.cfg.system, u,
                                   enable_thinking=self.cfg.enable_thinking)
                       for u in users[i : i + batch]]
            enc = self.tok(prompts, return_tensors="pt", padding=True,
                           add_special_tokens=False).to("cuda")
            if self.seed is not None:
                torch.manual_seed(self.seed + i)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                          pad_token_id=self.tok.pad_token_id, **sampling(self.cfg))
            out += self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return [t.strip() for t in out]


class MlxBackend:
    def __init__(self, cfg: Config, model: str, *, seed: int | None = None, **_) -> None:
        try:
            import mlx.core as mx
            from mlx_lm import load
            from mlx_lm.sample_utils import make_logits_processors, make_sampler
        except ImportError as e:
            raise BackendError(f"missing {e.name}: pip install 'ft27b[mlx]' (Apple silicon)") from e
        self.mx, self.cfg, self.seed, self.stats = mx, cfg, seed, []
        try:
            self.model, self.tok = load(model)
        except Exception as e:  # noqa: BLE001  (bad path, bad id, unsupported architecture)
            raise BackendError(f"could not load MLX model {model!r}: {e}") from e
        s = sampling(cfg)
        self.sampler = make_sampler(temp=s["temperature"], top_p=s["top_p"])
        self.processors = make_logits_processors(repetition_penalty=s["repetition_penalty"])

    def generate(self, users: list[str], *, max_new_tokens: int = 1024, **_) -> list[str]:
        from mlx_lm import stream_generate

        out = []
        for i, u in enumerate(users):
            if self.seed is not None:
                self.mx.random.seed(self.seed + i)
            prompt = prompt_text(self.tok, self.cfg.system, u, enable_thinking=self.cfg.enable_thinking)
            text, last = "", None
            for last in stream_generate(self.model, self.tok, prompt, max_tokens=max_new_tokens,
                                        sampler=self.sampler, logits_processors=self.processors):
                text += last.text
            if last is not None:
                self.stats.append({"prompt_tokens": last.prompt_tokens, "prompt_tps": last.prompt_tps,
                                   "generation_tokens": last.generation_tokens,
                                   "generation_tps": last.generation_tps,
                                   "peak_memory_gb": last.peak_memory})
            out.append(text.strip())
        return out


def load_backend(name: str, cfg: Config, *, adapter: str | None = None, model: str | None = None,
                 **kwargs):
    if name == "cuda":
        return CudaBackend(cfg, adapter, **kwargs)
    if name == "mlx":
        if not model:
            raise BackendError("--backend mlx needs --model (a converted MLX model dir or HF id)")
        return MlxBackend(cfg, model, **kwargs)
    raise BackendError(f"unknown backend {name!r}")


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)
