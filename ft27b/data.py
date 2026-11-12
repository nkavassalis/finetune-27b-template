"""Dataset contract, validation and splitting. Stdlib only.

A dataset is JSONL, one example per line:

    {"input": "...", "output": "...", "group": "2024", "id": "optional"}

`input` is the full user turn, `output` the response to train on. `group` is
whatever you hold out by (year, repo, author: see project.toml [data]).

`validate` looks for the bugs that actually hurt lkml-reviewer, all of them
boilerplate a model learns to copy, none of them caught by any count:

- a line that **ends** a large share of outputs (list footers, signatures);
- a line that **opens** a large share of outputs (quote headers, greetings);
- exact duplicates; empty fields; extreme lengths;
- inputs with no indentation at all where code is expected (flattened diffs).
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

BOILERPLATE_SHARE = 0.05   # a line ending/opening >5% of outputs is suspicious
_DIGITS = re.compile(r"\d+")


def read(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: not JSON ({e.msg})") from e
    return rows


def write(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _shape(line: str) -> str:
    """Normalise a line so 'On 3 Jan, Al wrote:' and 'On 9 Feb, Bo wrote:' match."""
    line = _DIGITS.sub("N", line.strip())
    words = line.split()
    return " ".join(words[:2] + ["…"] + words[-2:]) if len(words) > 6 else line


@dataclass
class Report:
    n: int = 0
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems


def validate(rows: list[dict], *, share: float = BOILERPLATE_SHARE) -> Report:
    rep = Report(n=len(rows))
    if not rows:
        rep.problems.append("no rows")
        return rep
    bad = [i for i, r in enumerate(rows)
           if not isinstance(r.get("input"), str) or not isinstance(r.get("output"), str)]
    if bad:
        rep.problems.append(f"{len(bad)} rows lack string 'input'/'output' (first: row {bad[0]})")
        rows = [r for i, r in enumerate(rows) if i not in set(bad)]
    empty = sum(1 for r in rows if not r["input"].strip() or not r["output"].strip())
    if empty:
        rep.problems.append(f"{empty} rows have an empty input or output")
    dups = sum(c - 1 for c in Counter((r["input"], r["output"]) for r in rows).values() if c > 1)
    if dups:
        rep.warnings.append(f"{dups} exact duplicate rows")

    outs = [r["output"].strip() for r in rows if r["output"].strip()]
    for where, pick in (("ends", lambda t: t.splitlines()[-1]), ("opens", lambda t: t.splitlines()[0])):
        shapes = Counter(_shape(pick(t)) for t in outs)
        for shape, c in shapes.most_common(5):
            if c / len(outs) > share and len(shape) > 1:
                rep.warnings.append(
                    f"{c / len(outs):.0%} of outputs {where} with a line like {shape!r} "
                    "(boilerplate the model will learn to copy?)")

    lens = sorted(len(r["output"]) for r in rows)
    ilens = sorted(len(r["input"]) for r in rows)
    rep.stats = {
        "rows": len(rows),
        "output_chars_median": lens[len(lens) // 2],
        "output_chars_p99": lens[int(len(lens) * 0.99)],
        "input_chars_median": ilens[len(ilens) // 2],
        "input_chars_p99": ilens[int(len(ilens) * 0.99)],
        "inputs_with_indentation": sum(bool(re.search(r"\n[ \t]+\S", r["input"]))
                                       for r in rows) / len(rows),
    }
    return rep


def split(rows: list[dict], key: str, eval_groups: list[str]) -> tuple[list, list, list]:
    """(train, eval, ungrouped). Ungrouped rows go to neither side, loudly."""
    groups = {str(g) for g in eval_groups}
    train, held, missing = [], [], []
    for r in rows:
        if key not in r:
            missing.append(r)
        elif str(r[key]) in groups:
            held.append(r)
        else:
            train.append(r)
    return train, held, missing


def sample(rows: list[dict], n: int, seed: int = 7) -> list[dict]:
    return random.Random(seed).sample(rows, min(n, len(rows)))
