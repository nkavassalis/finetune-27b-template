"""merge_lora.py on a tiny fake checkpoint. Needs torch + safetensors, so it
skips on machines without them (the gate still runs everything else)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")

ROOT = Path(__file__).resolve().parent.parent
KEY = "model.language_model.layers.0.self_attn.q_proj"


def make(tmp: Path, *, r=2, alpha=4, cfg_extra=None):
    base, adapter = tmp / "base", tmp / "adapter"
    base.mkdir()
    adapter.mkdir()
    torch.manual_seed(0)
    w = torch.randn(6, 5, dtype=torch.bfloat16)
    other = torch.randn(3, 3, dtype=torch.bfloat16)
    st.save_file({f"{KEY}.weight": w}, str(base / "a.safetensors"))
    st.save_file({"model.visual.x.weight": other}, str(base / "b.safetensors"))
    (base / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 1},
        "weight_map": {f"{KEY}.weight": "a.safetensors", "model.visual.x.weight": "b.safetensors"}}))
    (base / "config.json").write_text("{}")
    a, b = torch.randn(r, 5), torch.randn(6, r)
    st.save_file({f"base_model.model.{KEY}.lora_A.weight": a,
                  f"base_model.model.{KEY}.lora_B.weight": b}, str(adapter / "adapter_model.safetensors"))
    (adapter / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": alpha, **(cfg_extra or {})}))
    return base, adapter, w, a, b


def run(base, adapter, out):
    return subprocess.run([sys.executable, str(ROOT / "train" / "merge_lora.py"), "--base", str(base),
                           "--adapter", str(adapter), "--out", str(out)], capture_output=True, text=True)


def test_merge_is_w_plus_scaled_ba(tmp_path):
    base, adapter, w, a, b = make(tmp_path, r=2, alpha=4)
    r = run(base, adapter, tmp_path / "out")
    assert r.returncode == 0, r.stderr
    got = st.load_file(str(tmp_path / "out" / "a.safetensors"))[f"{KEY}.weight"]
    want = (w.float() + 2.0 * (b @ a)).to(torch.bfloat16)
    assert torch.equal(got, want)
    untouched = st.load_file(str(tmp_path / "out" / "b.safetensors"))["model.visual.x.weight"]
    assert torch.equal(untouched, st.load_file(str(base / "b.safetensors"))["model.visual.x.weight"])
    info = json.loads((tmp_path / "out" / "MERGE_INFO.json").read_text())
    assert info["modules_merged"] == 1 and info["scale"] == 2.0
    # MERGE_INFO ships with published models: no local filesystem paths in it
    assert str(tmp_path) not in json.dumps(info)


def test_rslora_uses_alpha_over_sqrt_r(tmp_path):
    base, adapter, w, a, b = make(tmp_path, r=4, alpha=4, cfg_extra={"use_rslora": True})
    assert run(base, adapter, tmp_path / "out").returncode == 0
    info = json.loads((tmp_path / "out" / "MERGE_INFO.json").read_text())
    assert info["scale"] == pytest.approx(2.0)


@pytest.mark.parametrize("extra", [{"use_dora": True}, {"fan_in_fan_out": True}, {"bias": "all"}])
def test_unsupported_adapters_are_refused(tmp_path, extra):
    base, adapter, *_ = make(tmp_path, cfg_extra=extra)
    r = run(base, adapter, tmp_path / "out")
    assert r.returncode != 0 and "refusing" in (r.stderr + r.stdout)
