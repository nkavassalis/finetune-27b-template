# AGENTS.md

Rules for anyone (person or agent) working in a project made from this
template. Read `docs/METHOD.md` and `docs/OPERATIONS.md` before changing
data or launching a run. Record where work stopped in `HANDOFF.md`.

## Purpose

<one sentence: what this model learns, and what it must not learn>

## Invariants

- **One config.** Settings live in `project.toml`; scripts read it. Flags
  override for one run, never permanently.
- **One prompt format.** Everything renders prompts through
  `ft27b/template.py`. `tests/test_core.py` fails if a script builds its
  own prompt or defines its own system prompt.
- **Read the data by eye** (`ft27b sample`) after every change to the data
  or the export, before training. Don't trust a percentage you haven't sampled.
- **Hold out by group**, never at random. Report the eval-set size with every score.
- **Sampled generations for evaluation**, never greedy. Validate any judge
  (`train/judge_check.py`) before reporting its verdicts.
- **Data, runs and weights never enter git** (`data/`, `runs/`, `*.safetensors`).
- **No addresses, hostnames or home paths in tracked files or commit
  messages.** They go in gitignored `.status.env` / `.secret-watchlist`. The
  pre-push and commit-msg hooks scan for them; enable them with
  `git config core.hooksPath .githooks`.
- **Anything you stop on a shared machine, you restore**, via the chain
  scripts' EXIT trap. Long runs get the watcher and an hourly status run.

## How to test

```bash
pip install -e '.[dev]'
python -m pytest            # hermetic; test_merge.py runs only where torch is installed
scripts/gate.sh             # tests + lint + python/shell syntax + secret scan
```

## Measured numbers

Keep a table of your funnel and runs here (rows at each data stage, held-out
loss, tok/s, peak memory) so a regression shows up as a changed number.

## Do not touch

- `tests/fixtures/`: captured truth (real model configs). Regenerating them
  silently launders regressions into passing tests.
- The allocator cap and restore traps in the chain scripts: they exist
  because a node froze without them.
