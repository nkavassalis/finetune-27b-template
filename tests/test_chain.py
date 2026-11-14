"""train/run-chain-2node.sh with docker/ssh/scp replaced by recorders: what it
runs, what it copies to the peer, and that serving is restored."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

CHAIN = Path(__file__).resolve().parent.parent / "train" / "run-chain-2node.sh"


def run_chain(tmp: Path, **env) -> tuple[int, list[str], str]:
    bin_ = tmp / "bin"
    bin_.mkdir(exist_ok=True)
    calls = tmp / "calls.log"
    for tool in ("docker", "ssh", "scp"):
        (bin_ / tool).write_text(f'#!/bin/bash\necho "{tool} $*" >> {calls}\n')
        (bin_ / tool).chmod(0o755)
    work = tmp / "work"
    (work / "runs").mkdir(parents=True, exist_ok=True)
    e = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "WORK": str(work), "RUN": "r1",
         "PEER": "peer", "MASTER_ADDR": "192.0.2.1", "IFACE": "eth9", **env}
    p = subprocess.run(["bash", str(CHAIN)], env=e, capture_output=True, text=True)
    logs = list(work.rglob("chain-r1.log"))
    return p.returncode, calls.read_text().splitlines() if calls.exists() else [], \
        logs[0].read_text() if logs else ""


def test_trains_on_both_nodes_and_restores_serving(tmp_path):
    rc, calls, _ = run_chain(tmp_path)
    assert rc == 0 and sum("torchrun" in c for c in calls) == 2
    assert sum(c.startswith("docker start") or ("ssh" in c and "docker start" in c) for c in calls) == 2


def test_a_resumed_run_copies_the_checkpoint_to_the_peer_first(tmp_path):
    ck = tmp_path / "work" / "runs" / "r1" / "adapter-latest"
    ck.mkdir(parents=True)
    (ck / "STEP").write_text("50")
    rc, calls, log = run_chain(tmp_path)
    first_train = next(i for i, c in enumerate(calls) if "torchrun" in c)
    copy = next(i for i, c in enumerate(calls) if c.startswith("scp") and "r1/adapter-latest" in c)
    assert rc == 0 and copy < first_train
    assert "resuming: copied runs/r1/adapter-latest (step 50)" in log
