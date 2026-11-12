"""Hermetic tests: no network, no GPU, no torch. Each one pins a lesson."""
from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import pytest

from ft27b import config, data, order, sizing
from ft27b.__main__ import main
from ft27b.template import chat_messages, prompt_text

ROOT = Path(__file__).resolve().parent.parent
QWEN = ROOT / "tests" / "fixtures" / "qwen3.8-27b"


# --- config -------------------------------------------------------------------

def test_shipped_project_toml_loads():
    cfg = config.load(ROOT / "project.toml")
    assert cfg.base and cfg.system and cfg.enable_thinking is False
    assert re.compile(cfg["model"]["lora_targets"])  # a valid regex


def test_missing_required_keys_are_rejected(tmp_path):
    p = tmp_path / "project.toml"
    p.write_text('[model]\nbase = "x"\n')
    with pytest.raises(ValueError, match="system"):
        config.load(p)


# --- one prompt format --------------------------------------------------------

class FakeTok:
    def apply_chat_template(self, msgs, **kw):
        self.kw = kw
        return json.dumps(msgs)


def test_prompt_uses_system_and_thinking_flag():
    tok = FakeTok()
    out = json.loads(prompt_text(tok, "SYS", "hi", enable_thinking=False))
    assert out == chat_messages("SYS", "hi")
    assert tok.kw == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}


@pytest.mark.parametrize("path", ["train/sft_lora.py", "train/compare.py", "ft27b/infer.py"])
def test_every_prompt_goes_through_the_shared_template(path):
    """lkml-reviewer's training script kept its own copy of the system prompt.
    Here nothing may: training, eval and inference all import ft27b.template."""
    src = (ROOT / path).read_text()
    assert "ft27b.template" in src or "from .template" in src
    assert not re.search(r"^SYSTEM\s*=", src, re.M)
    assert "apply_chat_template" not in src


# --- data -------------------------------------------------------------------

def rows_with(outputs, inputs=None):
    inputs = inputs or [f"input {i}" for i in range(len(outputs))]
    return [{"input": i, "output": o} for i, o in zip(inputs, outputs, strict=True)]


def test_validate_flags_a_footer_on_many_outputs():
    footer = "--\nTo unsubscribe from this list: send the line unsubscribe"
    outs = [f"real answer {i}\n{footer}" for i in range(40)] + [f"clean {i}" for i in range(40)]
    rep = data.validate(rows_with(outs))
    assert any("end" in w and "unsubscribe" in w for w in rep.warnings)


def test_validate_flags_an_invented_opener():
    outs = [f"On {d} Jan 2024, Person {d} wrote:\nanswer {d}" for d in range(30)]
    rep = data.validate(rows_with(outs))
    assert any("open" in w for w in rep.warnings)


def test_validate_problems_and_duplicates():
    rows = rows_with(["a", "", "b"]) + [{"input": "x"}] + rows_with(["b"], ["input 2"])
    rep = data.validate(rows)
    assert not rep.ok
    assert any("empty" in p for p in rep.problems) and any("lack" in p for p in rep.problems)
    assert any("duplicate" in w for w in rep.warnings)


def test_clean_data_passes():
    import random
    words = ["lock", "race", "buffer", "queue", "kernel", "page", "cache", "inode", "dentry", "mutex", "barrier", "flag"]
    rng = random.Random(1)
    outs = [" ".join(rng.sample(words, 6)) + "." for _ in range(50)]
    rep = data.validate(rows_with(outs))
    assert rep.ok and not rep.warnings


def test_outputs_that_differ_only_by_numbers_are_boilerplate():
    """Digits are normalised on purpose: that's how dated headers get caught."""
    rep = data.validate(rows_with([f"answer number {i} is different" for i in range(50)]))
    assert rep.warnings


def test_split_by_group_and_loud_ungrouped():
    rows = [{"input": "a", "output": "b", "group": g} for g in ("2023", "2024", 2024)]
    rows.append({"input": "a", "output": "b"})
    train, held, missing = data.split(rows, "group", ["2024"])
    assert (len(train), len(held), len(missing)) == (1, 2, 1)


def test_sample_is_deterministic():
    rows = rows_with([str(i) for i in range(100)])
    assert data.sample(rows, 5, seed=3) == data.sample(rows, 5, seed=3)


# --- order / resume -------------------------------------------------------------

def test_ranks_are_disjoint_and_cover_an_epoch():
    n, world = 5110, 2
    shards = [list(itertools.islice(order.examples(n, world, r, 7), n // world)) for r in range(world)]
    assert not set(shards[0]) & set(shards[1])
    assert len(set(shards[0]) | set(shards[1])) == n


@pytest.mark.parametrize("rank", [0, 1])
def test_resume_replays_exactly_the_uninterrupted_order(rank):
    n, world, ga, total, at = 5110, 2, 4, 639, 100
    full = list(itertools.islice(order.examples(n, world, rank, 7), total * ga))
    stream = order.resumed(n, world, rank, 7, step=at, grad_accum=ga)
    assert list(itertools.islice(stream, (total - at) * ga)) == full[at * ga:]


# --- sizing (calibrated on a real conversion) ------------------------------------

def test_sizing_matches_the_measured_qwen_conversion():
    s = sizing.shape(QWEN)
    assert s.text_params == pytest.approx(26.9e9, rel=0.01)
    assert (s.full_layers, s.linear_layers) == (16, 48)
    assert s.kv_bytes_per_token == 64 * 1024
    # converted 8-bit model measured 29.0 GB on disk; predicted within 2%
    assert sizing.weights_gb(s, 8) == pytest.approx(29.0, rel=0.02)
    # measured 30.2 GB peak in use with an 838-token prompt: the 4k estimate must not undershoot it
    assert 30.2 <= sizing.total_gb(s, 8, 4096) <= 34


def test_sizing_table_marks_what_does_not_fit():
    t = sizing.table(sizing.shape(QWEN), limit_gb=48)
    row8 = next(line for line in t.splitlines() if line.strip().startswith("8 |"))
    assert row8.count("ok") == 3 and row8.endswith("NO")


# --- CLI on the tiny example ------------------------------------------------------

def test_cli_validate_split_size(tmp_path, monkeypatch, capsys):
    proj = tmp_path / "project.toml"
    proj.write_text((ROOT / "project.toml").read_text().replace("eval_groups = []", 'eval_groups = ["2024"]'))
    monkeypatch.chdir(tmp_path)
    src = str(ROOT / "examples" / "tiny" / "all.jsonl")
    assert main(["validate", src]) == 0
    assert main(["split", src]) == 0
    assert len((tmp_path / "data" / "sft_eval.jsonl").read_text().splitlines()) == 3
    assert main(["size", str(QWEN)]) == 0
    assert "28.6 GB" in capsys.readouterr().out


def test_cli_without_a_project_toml_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["validate", "x.jsonl"]) == 2
    assert "project.toml" in capsys.readouterr().err
