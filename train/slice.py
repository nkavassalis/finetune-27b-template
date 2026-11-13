"""Metrics per slice of the held-out set, per model arm. Stdlib only.

Aggregate scores hide where a model fails. lkml-reviewer v3 looked only
slightly less specific than v2 overall; sliced, it named something from the
*patch* 8% of the time when the input carried discussion context (v2 24%,
the reference 26%), and 31% when it didn't. The extra context had pulled it
off the primary artifact. Slice before you pick the next lever.

    python3 train/slice.py runs/run-1/compare.jsonl \
        --primary-marker '--- The patch ---' --context-marker 'Earlier in this thread'

Slices: inputs with / without the context marker, and short / long inputs
(split at the median). "grounded in primary" = the output reuses an
identifier-like token from the part of the input after --primary-marker
(the whole input when the marker is absent).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics

# Strict by default: code identifiers only (snake_case names, calls like foo()).
# A looser "any 6+ letter word" version counts words like "kernel" that appear
# in almost every input and output, and it hid the effect on lkml-reviewer
# (47% vs 54% loose, against 8% vs 31% strict). Pick the definition that
# measures "talks about *this* artifact", and say which one you used.
STRICT = re.compile(r"\b(?:[a-z_][a-z0-9]*_[a-z0-9_]+|[A-Za-z_]\w*\(\))")
LOOSE = re.compile(r"\b(?:[a-z_][a-z0-9]*_[a-z0-9_]+|[A-Za-z_]\w*\(\))|\b[A-Za-z]\w{5,}\b")
IDENT = STRICT


def tokens(text: str) -> set[str]:
    return {m.group(0) for m in IDENT.finditer(text)}


def primary(inp: str, marker: str | None) -> str:
    return inp.split(marker, 1)[-1] if marker and marker in inp else inp


def report(rows: list[dict], arms: list[str], primary_marker: str | None,
           context_marker: str | None) -> list[dict]:
    lens = sorted(len(r["input"]) for r in rows)
    cut = lens[len(lens) // 2]
    slices = {"all": lambda r: True}
    if context_marker:
        slices[f"with '{context_marker}'"] = lambda r: context_marker in r["input"]
        slices[f"without '{context_marker}'"] = lambda r: context_marker not in r["input"]
    slices[f"input <= {cut} chars"] = lambda r: len(r["input"]) <= cut
    slices[f"input > {cut} chars"] = lambda r: len(r["input"]) > cut
    out = []
    for name, keep in slices.items():
        s = [r for r in rows if keep(r)]
        if not s:
            continue
        row = {"slice": name, "n": len(s)}
        for arm in arms:
            row[f"{arm}: grounded in primary"] = sum(
                bool(tokens(r[arm]) & tokens(primary(r["input"], primary_marker))) for r in s) / len(s)
            row[f"{arm}: median length"] = statistics.median(len(r[arm]) for r in s)
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("compare_jsonl")
    ap.add_argument("--primary-marker", help="text that precedes the primary artifact in the input")
    ap.add_argument("--context-marker", help="text that marks extra context in the input")
    ap.add_argument("--loose", action="store_true",
                    help="count any 6+ letter word as grounding (noisier; see IDENT)")
    args = ap.parse_args()
    global IDENT
    IDENT = LOOSE if args.loose else STRICT
    with open(args.compare_jsonl) as f:
        rows = [json.loads(line) for line in f]
    # Arms are the text fields; skip metadata such as the Mac's gen_tps / peak_gb.
    arms = ["reference"] + [k for k, v in rows[0].items()
                            if k not in ("input", "reference", "eval_index") and isinstance(v, str)]
    table = report(rows, arms, args.primary_marker, args.context_marker)
    print("| slice | n | " + " | ".join(f"{a} grounded / length" for a in arms) + " |")
    print("|---" * (len(arms) + 2) + "|")
    for t in table:
        cells = [f"{t[f'{a}: grounded in primary']:.0%} / {t[f'{a}: median length']:.0f}" for a in arms]
        print(f"| {t['slice']} | {t['n']} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
