"""train/score.py --targets / --train: the data's rate, next to the models'."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(*args: str) -> str:
    return subprocess.run([sys.executable, str(ROOT / "train" / "score.py"), *args],
                          capture_output=True, text=True, check=True).stdout


def write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_targets_and_the_training_column(tmp_path):
    inp = "def frobnicate_widget(w): ..."
    train = write(tmp_path / "sft.jsonl", [{"input": inp, "output": "frobnicate_widget leaks"},
                                           {"input": inp, "output": "I dislike everything."},
                                           {"input": inp, "output": "No. Just no."}])
    assert "| reuses a word from the input | 33% |" in run("--targets", train)
    cmp = write(tmp_path / "cmp.jsonl", [{"eval_index": 0, "input": inp, "reference": "frobnicate_widget bad",
                                          "v2_sampled": "meh"}])
    header = next(x for x in run(cmp, "--train", train).splitlines() if x.startswith("| metric"))
    assert header.split("|")[2:4] == [" reference ", " training targets "]
