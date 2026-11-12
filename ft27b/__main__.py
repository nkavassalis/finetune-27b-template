"""ft27b: the project CLI.

    ft27b validate data/sft.jsonl          # find boilerplate, duplicates, empties
    ft27b sample data/sft.jsonl -n 10      # read examples. Really read them.
    ft27b split data/all.jsonl             # held-out split by [data] split_key
    ft27b size path/to/hf-model            # will it fit on the Mac, per bits/context
    ft27b prompt "some input"              # the exact rendered prompt (needs a tokenizer)
    ft27b infer --backend mlx --model DIR --input-file x.txt
    ft27b targets                          # Linear module names, to write lora_targets

Everything reads project.toml (found in this directory or a parent).
Exit codes: 0 ok, 1 problems found, 2 bad input, 3 backend problem.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config as config_mod
from . import data, sizing


def cmd_validate(args, cfg) -> int:
    rep = data.validate(data.read(args.file))
    for k, v in rep.stats.items():
        print(f"  {k:<26} {v:.0%}" if isinstance(v, float) and v <= 1 else f"  {k:<26} {v}")
    for w in rep.warnings:
        print(f"warning: {w}")
    for p in rep.problems:
        print(f"PROBLEM: {p}")
    print("ok" if rep.ok else "problems found")
    print("Now read some rows by eye: ft27b sample", args.file)
    return 0 if rep.ok else 1


def cmd_sample(args, cfg) -> int:
    for r in data.sample(data.read(args.file), args.n, args.seed):
        print("=" * 80)
        print("--- input ---\n" + r["input"][: args.chars])
        print("--- output ---\n" + r["output"][: args.chars])
    return 0


def cmd_split(args, cfg) -> int:
    d = cfg["data"]
    key, groups = d.get("split_key", "group"), d.get("eval_groups", [])
    if not groups:
        print("set [data] eval_groups in project.toml first", file=sys.stderr)
        return 2
    train, held, missing = data.split(data.read(args.file), key, groups)
    data.write(d["train"], train)
    data.write(d["eval"], held)
    print(f"train {len(train)} -> {d['train']}\neval  {len(held)} -> {d['eval']} (groups {groups})")
    if missing:
        print(f"WARNING: {len(missing)} rows have no '{key}' and went to neither side")
    return 0


def cmd_size(args, cfg) -> int:
    limit = args.limit or cfg["convert"].get("mac_wired_limit_gb", 48)
    print(sizing.table(sizing.shape(args.model_dir), limit_gb=limit,
                       group=cfg["convert"].get("group_size", 64)))
    return 0


def cmd_prompt(args, cfg) -> int:
    from transformers import AutoTokenizer

    from .template import prompt_text
    tok = AutoTokenizer.from_pretrained(cfg.base)
    print(prompt_text(tok, cfg.system, args.text, enable_thinking=cfg.enable_thinking))
    return 0


def cmd_infer(args, cfg) -> int:
    from .infer import BackendError, load_backend
    text = sys.stdin.read() if args.input_file in (None, "-") else Path(args.input_file).read_text()
    try:
        be = load_backend(args.backend, cfg, adapter=args.adapter, model=args.model, seed=args.seed)
        out = be.generate([text], max_new_tokens=args.max_new_tokens)[0]
    except BackendError as e:
        print(f"ft27b: {e}", file=sys.stderr)
        return 3
    print(out)
    for st in getattr(be, "stats", []):
        print(json.dumps(st), file=sys.stderr)
    return 0


def cmd_targets(args, cfg) -> int:
    import collections

    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForImageTextToText

    with init_empty_weights():
        m = AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(cfg.base))
    names = collections.Counter(
        ".".join(n.split(".")[:3]) + " … " + n.split(".")[-1]
        for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear))
    for k, v in sorted(names.items()):
        print(f"{v:4d} × {k}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ft27b", description=__doc__.split("\n")[0])
    ap.add_argument("--config", help="path to project.toml (default: search upwards)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate").add_argument("file")
    s = sub.add_parser("sample")
    s.add_argument("file")
    s.add_argument("-n", type=int, default=10)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--chars", type=int, default=1500)
    sub.add_parser("split").add_argument("file")
    sz = sub.add_parser("size")
    sz.add_argument("model_dir")
    sz.add_argument("--limit", type=float, help="GB the Mac GPU may wire")
    sub.add_parser("prompt").add_argument("text")
    i = sub.add_parser("infer")
    i.add_argument("--backend", choices=["cuda", "mlx"], default="cuda")
    i.add_argument("--adapter")
    i.add_argument("--model")
    i.add_argument("--input-file")
    i.add_argument("--seed", type=int)
    i.add_argument("--max-new-tokens", type=int, default=1024)
    sub.add_parser("targets")
    args = ap.parse_args(argv)
    try:
        cfg = config_mod.load(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"ft27b: {e}", file=sys.stderr)
        return 2
    return {"validate": cmd_validate, "sample": cmd_sample, "split": cmd_split, "size": cmd_size,
            "prompt": cmd_prompt, "infer": cmd_infer, "targets": cmd_targets}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
