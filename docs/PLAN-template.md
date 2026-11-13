# Plan: v<N>

**Status:** draft (YYYY-MM-DD). One of: draft, proposal, queued, running, done,
superseded, abandoned, with a date and what changed. `scripts/docs_check.py`
requires it, and a README journey row for every PLAN-v<N>.md.

Write this **before** training, and don't edit the criteria after the
results are in. That's what keeps "better" from being redefined to fit.

## What v<N-1> showed

Funnel, the main comparison table, and the diagnostics (`ft27b shift`,
`train/slice.py`, the judge's reasons tallied by type). Each with its sample
size, and "suggestive" vs "established" said plainly.

## Changes in v<N>

At most two or three, each tied to a finding above. Everything else stays
the same (LoRA config, schedule, eval subset and seed), so the comparison can
tell which change mattered.

| finding | change |
|---|---|

## Criteria

**Gates (no regression):**
**Targets (vs v<N-1> and base):**
**Eval set:** same as v<N-1>? If not, how loss stays comparable.

## Cost

Training tokens and time (1 or 2 nodes), downtime on shared machines, and
who has agreed to it.

## Process

Smoke-test the whole chain (training, copies, evaluation) at tiny sizes;
watcher plus hourly status; read 20 side-by-sides before believing any table.
