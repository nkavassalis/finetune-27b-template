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

- **The data selection changes** (a gate loosened, a filter added, new
  sources): run `train/score.py --targets` on the new training file and put
  each target metric's rate in the PLAN's **Data rate:** line, in the same
  commit. (lkml-reviewer's v3 loosened its gate; two-thirds of the new pairs
  weren't code review, and nobody measured it for two versions.)
- **A model falls short on a metric:** measure it on the training targets
  first. If the data scores the same, change the data, not the training.
- **Any result arrives:** check every queued decision it bears on
  ("Automatic decisions", 5) before reporting it.
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

## Automatic decisions

Some steps choose for us: which model the next version starts from, whether a
result clears a bar. On lkml-reviewer a rule set beforehand would have started
the next version from the model the blind judge (the only correctness
signal) rated worse, for a step meant to improve correctness. A person
caught it, not the tooling. So:

1. **Write a `## Decision:` section in the PLAN before the data**, with
   **Optimizes:** (what the step after the decision is for) and **Measured
   by:** (which signal in the rule measures that). `docs_check.py` requires
   both lines. A blank "Measured by" is the warning.
2. **Pre-mortem:** in the same section, the results that would make the rule's
   choice wrong. Then look for them when the results come in.
3. **Automate only when the evidence agrees.** After the rule picks,
   `train/decide.py` checks the choice against every result available and
   exits 76 (the queue stops and emails "Needs you") on a significant
   conflict, or if a `--target` signal (the metric the next step is for)
   leans the other way at all or is missing. A significance test alone
   would have let the 45–28 judge result through (p = 0.06).
   **The target is the next step's own scoring function**, applied to the
   candidates (`train/rubric.py`): for preference training, the rubric that
   will score its pairs. A different judge measures a different question
   (a pairwise judge without "saying little is not a virtue" rewards
   caution).
4. **A person at every fork that commits nights of compute.** Mechanical steps
   run unattended; branches that are expensive to undo get a human gate.
5. **When any result arrives, check what's queued behind it.** For each
   pending decision in `QUEUE.toml` and each PLAN Decision section: does this
   change it? Every status report says what a new result means for queued
   decisions, not only what the number is.

## Do not touch

- `tests/fixtures/`: captured truth (real model configs). Regenerating them
  silently launders regressions into passing tests.
- The allocator cap and restore traps in the chain scripts: they exist
  because a node froze without them.
