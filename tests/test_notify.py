"""scripts/notify.sh: pluggable sender, local log, and .status.env as
defaults only (so a test can never pick up a real sender)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def project(tmp: Path, status_env: str = "") -> Path:
    (tmp / "scripts").mkdir()
    (tmp / "runs").mkdir()
    shutil.copy(ROOT / "scripts" / "notify.sh", tmp / "scripts" / "notify.sh")
    if status_env:
        (tmp / ".status.env").write_text(status_env)
    return tmp


def notify(root: Path, subject: str, body: str, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(root / "scripts" / "notify.sh"), subject], input=body,
                          text=True, capture_output=True, env={"PATH": "/usr/bin:/bin", **(env or {})})


def fake_sender(tmp: Path) -> Path:
    s = tmp / "fake-send.sh"
    s.write_text('#!/bin/bash\necho "subject=${@: -1}" >> "$(dirname "$0")/sent.log"\n'
                 'cat >> "$(dirname "$0")/sent.log"\n')
    s.chmod(0o755)
    return s


def test_without_a_sender_it_logs_locally(tmp_path):
    root = project(tmp_path)
    assert notify(root, "Night report", "all good").returncode == 0
    log = (root / "runs" / "notifications.log").read_text()
    assert "Night report" in log and "all good" in log


def test_sender_gets_subject_as_last_arg_and_body_on_stdin(tmp_path):
    root = project(tmp_path, f'NOTIFY_CMD="{fake_sender(tmp_path)} --subject"\n')
    assert notify(root, "PROBLEM: x", "details here").returncode == 0
    sent = (tmp_path / "sent.log").read_text()
    assert "subject=PROBLEM: x" in sent and "details here" in sent


def test_environment_wins_over_status_env(tmp_path):
    """.status.env names a real sender; a test's NOTIFY_CMD must override it."""
    real = tmp_path / "REAL-SENDER-MUST-NOT-RUN"
    root = project(tmp_path, f'NOTIFY_CMD="touch {real}"\n')
    assert notify(root, "s", "b", env={"NOTIFY_CMD": f"{fake_sender(tmp_path)} --subject"}).returncode == 0
    assert not real.exists() and "subject=s" in (tmp_path / "sent.log").read_text()


def test_a_failing_sender_is_reported_and_still_logged(tmp_path):
    root = project(tmp_path, 'NOTIFY_CMD="false"\n')
    r = notify(root, "s", "b")
    assert r.returncode == 1 and "failed" in r.stderr
    assert "s" in (root / "runs" / "notifications.log").read_text()
