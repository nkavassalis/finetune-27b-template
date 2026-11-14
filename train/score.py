"""Cheap, deterministic metrics over compare.jsonl. Stdlib only.

These are behaviour proxies, not quality. They show whether a model *acts*
like the reference (grounded in its input, plain, not looping), not whether
it's right. Read the side-by-sides, and use judge.py for that.

    python3 train/score.py runs/run-1/compare.jsonl

Add project-specific metrics to EXTRA_METRICS (name -> function(text, row) -> bool).
lkml-reviewer added an abuse-filter rate there, for example.
"""
from __future__ import annotations

import json
import re
import statistics
from collections.abc import Callable

WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
ASSISTANT_RE = re.compile(
    r"(^\s*[-*]\s|\*\*|^#+\s|\bgreat question\b|\bI hope (this|that) helps\b|\blet me know\b|"
    r"\bcertainly[,!]|\bin summary\b|\bas an ai\b)", re.I | re.M)
REFUSAL_RE = re.compile(r"\b(I can'?t help|I cannot (help|assist)|I'?m not able to|as an ai)\b", re.I)
COMMON = {"this", "that", "with", "from", "have", "should", "would", "there", "which", "what"}

EXTRA_METRICS: dict[str, Callable[[str, dict], bool]] = {}


def loops(text: str) -> bool:
    """The same non-trivial line 3+ times: decoding stuck in a loop."""
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 20]
    return any(lines.count(ln) >= 3 for ln in set(lines))


def grounded(text: str, row: dict) -> bool:
    """Reuses a distinctive word (4+ chars) from the input."""
    inp = {w.lower() for w in WORD_RE.findall(row["input"])}
    out = {w.lower() for w in WORD_RE.findall(text)}
    return bool((inp & out) - COMMON)


def metrics(rows: list[dict], key: str) -> dict[str, float]:
    n = len(rows)
    out = [r[key] for r in rows]
    m = {
        "median length (chars)": statistics.median(len(o) for o in out),
        "reuses a word from the input": sum(grounded(o, r) for o, r in zip(out, rows, strict=True)) / n,
        "assistant-style formatting/phrases": sum(bool(ASSISTANT_RE.search(o)) for o in out) / n,
        "refusal phrasing": sum(bool(REFUSAL_RE.search(o)) for o in out) / n,
        "repeats a line 3+ times (loop)": sum(loops(o) for o in out) / n,
    }
    for name, fn in EXTRA_METRICS.items():
        m[name] = sum(bool(fn(o, r)) for o, r in zip(out, rows, strict=True)) / n
    return m


def read_rows(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def targets_column(path: str) -> dict[str, float]:
    """The same metrics on the TRAINING targets: the rate the data teaches.
    A model rarely does something more often than its targets do, so this is
    the ceiling to read every model column against."""
    return metrics(read_rows(path), "output")


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="proxy metrics per arm; the training targets' rate alongside")
    ap.add_argument("compare", nargs="?", help="compare.py output (held-out rows, one column per arm)")
    ap.add_argument("--train", help="training file: adds a 'training targets' column (the data's rate)")
    ap.add_argument("--targets", help="training file alone: the metrics on its targets, nothing else")
    args = ap.parse_args(argv)
    if args.targets:
        col = targets_column(args.targets)
        print(f"training targets, n = {len(read_rows(args.targets))}\n\n| metric | training targets |\n|---|---|")
        for m, v in col.items():
            print(f"| {m} | {v:.0f} |" if m.startswith("median") else f"| {m} | {v:.0%} |")
        return
    rows = read_rows(args.compare)
    keys = ["reference"] + [k for k in rows[0] if k not in ("input", "reference", "eval_index")]
    cols = {k: metrics(rows, k) for k in keys}
    if args.train:   # the data's rate, next to the reference's and the models'
        keys.insert(1, "training targets")
        cols["training targets"] = targets_column(args.train)
    print(f"n = {len(rows)} held-out rows\n")
    print("| metric | " + " | ".join(k.replace("_sampled", ", sampled") for k in keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for m in cols["reference"]:
        cells = [f"{cols[k][m]:.0f}" if m.startswith("median") else f"{cols[k][m]:.0%}" for k in keys]
        print(f"| {m} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
