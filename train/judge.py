"""Blind pairwise judge: base vs fine-tuned, on any OpenAI-compatible endpoint.

Stdlib only (urllib). Every comparison is asked twice with the candidates
swapped, so a judge that prefers "the first one" scores a tie, not a win.
The reference response is given; the judge is asked which candidate makes
the same substantive points with fewer errors, not which one sounds more
like the reference.

Caveat for every report: if the judge is from the same model family as the
base, it may share its taste. Say so, and validate it with judge_check.py.

    JUDGE_URL=http://<vllm-host>:8000/v1/chat/completions \
        python3 train/judge.py data/train/run-1/compare.jsonl data/train/run-1/judge.jsonl
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# Any OpenAI-compatible endpoint. Set JUDGE_URL to your server; nothing about
# the network it lives on belongs in the repo.
URL = os.environ.get("JUDGE_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("JUDGE_MODEL", "qwen3.8-flash-next")
# The task description and criteria come from project.toml [judge]; these are
# the defaults. Keep accuracy first: on lkml-reviewer the judge's most useful
# verdicts were "A makes false claims, B doesn't".
_J = {}
try:
    import tomllib
    from pathlib import Path as _P
    for _d in (_P.cwd(), *_P.cwd().parents):
        if (_d / "project.toml").exists():
            _J = tomllib.loads((_d / "project.toml").read_text()).get("judge", {})
            break
except Exception:  # noqa: BLE001  (judge works without a config)
    _J = {}
TASK = _J.get("task", "You are evaluating responses to the input below.")
CRITERIA = _J.get("criteria", [
    "accuracy: no false claims about the input",
    "overlap with the substantive points the reference makes",
    "specificity to this input rather than generic advice",
])
IGNORE = _J.get("ignore", "tone, formatting and length")
PROMPT = TASK + """

INPUT (may be truncated):
<<<
{patch}
>>>

REFERENCE: the response written by the person the model learns from:
<<<
{reference}
>>>

CANDIDATE A:
<<<
{a}
>>>

CANDIDATE B:
<<<
{b}
>>>

Which candidate is the better response to THIS input? Judge on, in order:
""" + "\n".join(f"{i}. {c};" for i, c in enumerate(CRITERIA, 1)) + """
Ignore """ + IGNORE + """. A response that repeats itself or invents facts
should lose.

Answer with one JSON object on the last line:
{{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}}"""


def ask(patch: str, ref: str, a: str, b: str, max_tokens: int = 24000) -> dict:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(
            patch=patch[:6000], reference=ref[:3000], a=a[:5000], b=b[:5000])}],
        # The judge is a reasoning model: at 6000 it ran out of budget mid-thought
        # on 60/156 calls and returned empty content. Empty is never a verdict.
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        text = json.load(resp)["choices"][0]["message"]["content"] or ""
    found = re.findall(r"\{[^{}]*\"winner\"[^{}]*\}", text)
    try:
        return json.loads(found[-1])
    except (IndexError, json.JSONDecodeError):
        return {"winner": "invalid", "reason": text[-200:]}


def resolve(row: dict, v1: dict, v2: dict) -> dict:
    m1 = {"A": "base", "B": "tuned"}.get(v1.get("winner"), v1.get("winner"))
    m2 = {"A": "tuned", "B": "base"}.get(v2.get("winner"), v2.get("winner"))
    if m1 == m2 and m1 in ("base", "tuned"):
        final = m1
    elif "invalid" in (m1, m2):
        final = "invalid"
    else:
        final = "tie"  # the judge disagreed with itself across orderings
    return {"subject": row["input"].split("\n")[0], "order1": v1, "order2": v2,
            "final": final}


# Which columns of compare.jsonl to pit against each other; `--sampled` switches
# to the temperature-0.7 arms. Results are still reported as base/tuned.
BASE, TUNED = "base", "tuned"


def judge(row: dict) -> dict:
    # Order 1: A = base, B = tuned.  Order 2: A = tuned, B = base.
    v1 = ask(row["input"], row["reference"], row[BASE], row[TUNED])
    v2 = ask(row["input"], row["reference"], row[TUNED], row[BASE])
    return resolve(row, v1, v2)


def retry(pair: tuple[dict, dict]) -> dict:
    """Re-ask only the orderings that came back invalid; keep valid verdicts."""
    row, prev = pair
    v1, v2 = prev["order1"], prev["order2"]
    if v1.get("winner") == "invalid":
        v1 = ask(row["input"], row["reference"], row[BASE], row[TUNED])
    if v2.get("winner") == "invalid":
        v2 = ask(row["input"], row["reference"], row[TUNED], row[BASE])
    return resolve(row, v1, v2)


def _progress(pool, fn, items) -> list:
    """pool.map with a progress line every 5 rows, results in input order.
    The first version printed nothing until the end, so its ETA was a guess."""
    futures = [pool.submit(fn, it) for it in items]
    for done, _ in enumerate(as_completed(futures), 1):
        if done % 5 == 0 or done == len(futures):
            print(f"[{time.strftime('%H:%M:%S')}] {done}/{len(futures)} rows judged",
                  flush=True)
    return [f.result() for f in futures]


def main(src: str, dst: str, mode: str = "") -> None:
    with open(src) as f:
        rows = [json.loads(line) for line in f]
    with ThreadPoolExecutor(max_workers=4) as pool:
        if mode == "--retry-invalid":
            with open(dst) as f:
                prev = [json.loads(line) for line in f]
            results = _progress(pool, retry, list(zip(rows, prev, strict=True)))
        else:
            results = _progress(pool, judge, rows)
    with open(dst, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    tally = {k: sum(r["final"] == k for r in results) for k in ("tuned", "base", "tie", "invalid")}
    print(f"n = {len(results)}  " + "  ".join(f"{k} {v}" for k, v in tally.items()))


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--sampled" in argv:
        argv.remove("--sampled")
        BASE, TUNED = "base_sampled", "tuned_sampled"
    if "--pair" in argv:  # --pair base_sampled,v3_sampled: any two arms
        i = argv.index("--pair")
        BASE, TUNED = argv[i + 1].split(",")
        del argv[i : i + 2]
    main(*argv)
