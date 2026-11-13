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
scripts/gate.sh             # tests + lint + python/shell syntax + docs drift + secret scan
```

## Measured numbers

Keep a table of your funnel and runs here (rows at each data stage, held-out
loss, tok/s, peak memory) so a regression shows up as a changed number.

## Documenting as we go

What you learn is as much the output as the model. Docs drift quietly (in
lkml-reviewer: a README calling an old version current, a stale handoff, a plan
contradicting the code), so these are rules, and `scripts/docs_check.py` (in
the gate) checks the mechanical part.

**Every fact has one home; everything else links to it.**

| fact | home |
| --- | --- |
| the story, two lines per version | README "The journey so far" table, near the top |
| measured results per version | the report (`docs/REPORT-template.md` to start) |
| what a version is for, its criteria (set before training), decisions | `docs/PLAN-v<N>.md`, **Status:** line at the top |
| what we believed, what we found, what changed | `docs/LEARNINGS.md`, append-only, dated |
| where things stand right now | `HANDOFF.md` Status, rewritten, never appended |
| design, invariants, how to run | this file |
| running long jobs | `docs/OPERATIONS.md` |

**On each event, in the same commit as the change:**

- **A version's results arrive:** report section, README journey row, PLAN
  status → done with the outcome against its criteria, LEARNINGS entry for
  any surprise, HANDOFF Status, model card if published.
- **You find you were wrong** (bug, misdiagnosis, a number that didn't hold
  up): LEARNINGS entry (Believed / Evidence / Changed / Generalizes) next to
  the fix and its regression test.
- **A decision is made:** dated in the PLAN; HANDOFF if it changes what runs next.
- **Code stops matching a doc:** fix the doc in the same commit.
- **A lesson applies beyond this project:** send it back to the template
  (METHOD.md / OPERATIONS.md) and say so under **Generalizes**.
- **End of a session:** rewrite HANDOFF Status with today's date.

Don't quote numbers that change every commit (test counts) unless dated.

## Do not touch

- `tests/fixtures/`: captured truth (real model configs). Regenerating them
  silently launders regressions into passing tests.
- The allocator cap and restore traps in the chain scripts: they exist
  because a node froze without them.
