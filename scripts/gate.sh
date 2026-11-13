#!/usr/bin/env bash
# The gate: tests, lint, syntax (Python and shell), docs drift, secret scan. The pre-push
# hook runs it; run it yourself before committing. Every step must pass, and a
# step that can't run SKIPs loudly instead of passing silently.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=python3
fail=0
step() { printf '\n=== %s ===\n' "$*"; }

step "tests (hermetic: no network, no GPU)"
"$PY" -m pytest -q || fail=1

step "lint"
if [ -x .venv/bin/ruff ]; then .venv/bin/ruff check . || fail=1
elif command -v ruff >/dev/null; then ruff check . || fail=1
else echo "SKIP — ruff not installed (pip install -e '.[dev]')"; fi

step "syntax: python + shell"
"$PY" -m compileall -q -x '\.venv|runs|data' . || fail=1
for f in scripts/*.sh train/*.sh .githooks/*; do bash -n "$f" || { echo "syntax error: $f"; fail=1; }; done

step "docs (drift check: dead paths, stale counts, plan status, handoff date, learnings format)"
"$PY" scripts/docs_check.py || fail=1

step "secret scan (staged + branch vs origin/main)"
scripts/secret-scan.sh --staged || fail=1
git rev-parse --verify -q origin/main >/dev/null && { scripts/secret-scan.sh --range origin/main..HEAD || fail=1; }

printf '\n=== gate: %s ===\n' "$([ $fail = 0 ] && echo PASS || echo FAIL)"
exit $fail
