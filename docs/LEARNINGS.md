# Learnings

Append-only. One entry per thing you believed and found to be wrong (or found
out the hard way), in date order, newest at the bottom. Never edit an old
entry's facts; if a later finding overturns one, add a new entry that says so.
`scripts/docs_check.py` (in the gate) checks the format. The README's journey
table is the two-line summary per version, the report has the numbers, and
this file is the reasoning between them.

Entry format (headings must be `## YYYY-MM-DD: title`):

    ## 2026-11-11: The smoke-test model found a data bug

    **Believed:** patch coverage was 88.2%.
    **Evidence:** a smoke-test generation said "this is a block layer patch,
    not an eventfs patch"; patch texts were keyed by a per-month counter that
    repeats every month. Real coverage 76.1%.
    **Changed:** unique keys + a regression test; coverage reported as 76.1%.
    **Generalizes:** yes: read a small model's outputs before the big run
    (METHOD.md §1).

lkml-reviewer's log (github.com/nkavassalis/lkml-reviewer, docs/LEARNINGS.md)
is a worked example; most of this template came out of it.
