# The method

How to take a dataset to a fine-tuned 27B model you can trust and ship. Each
step is here because skipping it cost something real on lkml-reviewer, the
project this template was extracted from. The cost is noted where it matters.

## 0. Decide what you're teaching, in one sentence

lkml-reviewer's was *"train on his judgement, not his swearing."* Every
curation decision traces back to that sentence. Write yours at the top of
`project.toml`'s description before you touch data. When two choices both
look reasonable, it decides.

## 1. Data: the step that decides everything

**Contract:** JSONL rows `{"input", "output", "group"}`; see `ft27b/data.py`.
`input` is the full user turn the model will see at inference time; `output`
is exactly what it should learn to say.

**Read it by eye, every time it changes.** `ft27b sample data/sft.jsonl -n 20`.
On lkml-reviewer, five separate data bugs passed every test and produced
plausible counts. Every one was found by reading rows or a model's output:

| bug | symptom | how it was found |
| --- | --- | --- |
| a whitespace cleanup collapsed all indentation | every diff in the training set flattened | reading the export |
| mailing-list footers kept in targets | 52% of outputs ended "To unsubscribe…" | reading the export |
| an over-broad filter word | "Looks sane to me" cut from 86 reviews | reading the filter's decisions |
| records keyed by a counter that repeats per month | 2024 reviews paired with 2010 patches | **a 3-step smoke-test model said "this is a block-layer patch, not an eventfs patch"** |
| quote headers kept in targets | the model opened 94% of replies with an invented "On <date>, <name> wrote:" | reading generations |

`ft27b validate` now catches the generic shape of most of these (a line that
ends or opens many outputs, duplicates, empty fields, flattened inputs), but
it is not a substitute for reading. **A percentage you haven't sampled by eye
isn't a metric.**

**Strip what isn't the target behaviour, at export, reversibly.** Footers,
signatures, quote headers, mangled addresses: remove them in the export step,
never in the raw corpus, so every transform can be audited and undone. And
never let a transform touch code or formatting you want learned.

**Make every target predictable from its input.** A reply that answers
something the model can't see (an earlier message, a private conversation,
"applied, thanks") teaches vague, generic replies. Either put the missing
context in the input (lkml-reviewer v3 added the earlier thread) or drop the
row.

**Hold out by group, never at random.** Rows near each other in time or
source leak into each other and flatter every score. Split by year, repo or
author (`[data] split_key`), make the held-out set big enough to rank models
(lkml-reviewer went from 78 to 774 rows), and report its size next to every
number.

**Moderation drops, never rewrites.** If the data contains things you won't
teach, cut whole sentences or drop the row; never paraphrase. Rewriting puts
your words in the model. Then read the filter's decisions: a vocabulary filter
will hit surnames and approvals.

## 2. Training

The defaults in `project.toml` are the ones that worked:

- **BF16 LoRA, r=16, α=32, on every linear layer of the language model.**
  Full fine-tuning a 27B with AdamW needs ~450–490 GB (weights 54 + grads 54 +
  optimizer 324 + activations), not the "3–4× the weights" rule of thumb, which
  is for 8-bit optimizers. LoRA fits one 121 GB node. See `docs/HARDWARE.md`.
- **Loss on the response only.** Prompt tokens are masked.
- **One epoch.** A second overfit on ~1.2k rows (train loss kept falling,
  held-out loss rose).
- **Thinking off** (`enable_thinking = false`) for style tasks: train and
  infer with an empty `<think></think>`.
- **Truncate the input's tail, never the response.** Responses longer than half
  the budget are skipped, not cut.
- **Print the rendered template in the training log.** Template drift between
  training and inference is a silent quality loss. This template has one
  prompt module (`ft27b/template.py`) and a test that fails if anything
  renders prompts on its own.

**Smoke test first, and read its sample.** `TRAIN_EXTRA="--limit 40 --max-steps 3"`
takes minutes and exercises load, memory, loss, save and generation. The
wrong-patch bug above was found by reading that 3-step model's one sample.

## 3. Evaluation

Settle the success criteria **before** training (lkml-reviewer's v3 plan did),
so "better" can't be redefined after the fact. Then gather four kinds of
evidence, from cheap to strong:

1. **Held-out loss before and after** (`sft_lora.py` logs both). Only
   comparable between runs with identical targets: stripping boilerplate
   changes what's being predicted.
2. **Generations on a fixed held-out subset** (`train/compare.py --subset 100`):
   base and tuned from one model load (the adapter toggled), identical prompts,
   seeded. **Sampled, not greedy:** greedy looped on 44% of lkml-reviewer's
   outputs and 0% sampled. Evaluate what users will run. Give both arms
   enough tokens to finish (a 400-token cap cut 50/78 base replies mid-sentence
   and biased the judge).
3. **Proxy metrics** (`train/score.py`): length, grounding in the input,
   chatbot-isms, refusals, loops, plus your own (`EXTRA_METRICS`). Behaviour,
   not correctness.
4. **A blind pairwise judge** (`train/judge.py`): both orderings, a win only if
   it wins both, the reference given, accuracy first.
   - **Validate it before you trust it:** `train/judge_check.py` pits a real
     response against a fluent response to a *different* input; it must pass
     ~100%.
   - **Give a reasoning judge room to think:** at 6,000 tokens it returned
     empty content on 60/156 calls. Empty is never a verdict.
   - **Read its reasons.** On lkml-reviewer, the tuned model "won" largely by
     saying less, which beats confident fabrication on accuracy, but isn't
     the same as having more substance.
   - Say in the report if the judge shares the base model's family.
5. **Humans.** A few dozen blind ratings by someone who knows the domain
   outrank everything above.

Write it up with `docs/REPORT-template.md`, including what didn't work.

## 4. Iterate: from one version's results to the next

Don't change the recipe because a number looks bad. Find out *where* the model
fails, then pick the one change that addresses that. Three kinds of criteria,
written into the next version's plan (`docs/PLAN-template.md`) **before**
training:

- **Gates: no regression allowed.** Things the model must never start
  doing again (signing as a real person, leaking emails, abuse above the
  source's own rate, looping, a Mac build that loses to the CUDA one).
- **Targets: move toward the reference.** The judge's win rate against the
  previous version and the base, specificity, length, like-for-like loss,
  human ratings.
- **Diagnostics: tell you which lever to pull.** They are not scores.

The diagnostics that found lkml-reviewer v3's two real problems, both now
tools here:

1. **Do the training targets look like the held-out targets?**
   `ft27b shift`. v3 trained on replies with a median of 620 chars while the
   held-out replies had 906, and the model wrote short replies. Then **find out
   why before you fix it.** lkml-reviewer first blamed the data selection and
   planned a length filter. Two checks overturned that:
   - Measured *after* cleaning (the raw text was padded with footers and
     headers), the author simply writes longer now: ~580 chars per reply
     through 2005–2019, ~780 in 2020–2024. The held-out years were a different
     era. A time-based split guards against leakage, and it also builds in
     this drift. Fix with **recency weighting**, not filtering.
   - Reading the rows a length filter would drop: about half were real terse
     technical replies. The filter would have taught the model that short
     means wrong.

   Measure on the same transformation the model trains on, and read what a
   filter removes before adopting it.
2. **Is the output grounded in the primary input or in extra context?**
   `train/slice.py --primary-marker … --context-marker …`. Adding
   discussion context to v3's inputs made it name something from the patch
   8% of the time on inputs with context vs 31% without (the reference:
   26% vs 62%). The extra context pulled it off the artifact it was meant to
   review. Overall averages hid this. **Slice before you decide.**
   - How you define "grounded" decides whether the effect shows at all. A
     loose definition (any long word) put the same two slices at 47% vs
     54%. Use the definition that measures what you care about, and report it.

Then map each finding to one lever, and change as few things at once as you
can, so the next comparison can tell you which change mattered:

| finding | lever |
| --- | --- |
| targets shorter / thinner than the reference | rebalance or filter the data until `ft27b shift` is clean |
| answers the context, not the artifact | artifact first, each section with its own token budget; trim context, never the artifact |
| truncated inputs score worse | smarter truncation (keep every section's head) or longer context, if memory allows |
| more false claims than before | preference training (DPO) on judge-ranked samples against the reference |
| the Mac build loses to CUDA | a conversion change (group size, mixed precision), not retraining |
| a behaviour leaks (signatures, names) | strip it at export *everywhere it occurs*, not just at the end; then measure the rate in sampled generations |

## 5. Ship

Run `docs/CONVERSION.md`: size check → merge in BF16 → verify the merge → MLX
8-bit → smoke test → quality gate against the CUDA model → publish
(privately first). On a 64 GB M4 Pro, a 27B at 8-bit measured 30.2 GB peak at
8.5–8.8 tok/s.
