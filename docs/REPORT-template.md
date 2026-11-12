# <model name>: <one-line purpose>

**Status:** <one paragraph: what works, what doesn't, how to run it safely>

**What we're teaching:** <the one sentence from project.toml>

## 1. What was built

| | |
|---|---|
| base model | |
| method | BF16 LoRA r=…, α=…, targets … |
| data | N train / M held out (held out by …) |
| schedule | epochs, lr, effective batch, max tokens |
| hardware | nodes, tok/s, peak memory, wall time |

## 2. Data, and the bugs that mattered

Funnel table (raw → filtered → train/eval). Every data bug found, how it was
found, and what changed. Bugs found by reading count as findings, not as
embarrassments.

## 3. Runs

| run | data | epochs | held-out loss before → after |
|---|---|---|---|

## 4. Evaluation method

Subset size and how it was chosen, decoding settings (sampled), judge model
and its validation result, any human ratings.

## 5. Results

Proxy-metric table (reference / base / tuned), judge tally with the number
of invalid and tie verdicts, and three side-by-side examples, including one
where the tuned model is worse.

## 6. What we learned

## 7. Limits of this evaluation

Eval set size, judge family, no human raters (if so), single reference.

## 8. Using it responsibly

## 9. Next steps

## Reproduce

Commit hash, project.toml, exact commands.
