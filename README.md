# finetune-27b-template

A template for fine-tuning a **27B model with LoRA on NVIDIA DGX Spark
(GB10)** and shipping it as an **MLX 8-bit model on a 64 GB Apple-silicon
Mac**. You bring a dataset; the template brings the training, evaluation,
conversion and operations tooling, plus the method, which is the part that
actually makes the model good.

Extracted from [lkml-reviewer](https://github.com/nkavassalis/lkml-reviewer)
(a Qwen3.8-27B fine-tuned on Linus Torvalds' patch reviews). Every default
here was measured there.

**Use this template** on GitHub, then edit `project.toml`.

## The pipeline

```
  your data ─► 1 validate + read ─► 2 split ─► 3 smoke test ─► 4 train ─► 5 evaluate ─► 6 convert ─► 7 ship
               ft27b validate      ft27b      run-chain.sh    run-chain   compare.py    convert-    MLX 8-bit
               ft27b sample        split      (3 steps)       (1/2 node)  score, judge  mlx.sh      on the Mac
```

```bash
pip install -e '.[dev]'                         # core is stdlib-only; this adds pytest/ruff
git config core.hooksPath .githooks             # gate + secret scan before every push

# 1. data: JSONL {"input", "output", "group"}; see ft27b/data.py
ft27b validate data/all.jsonl                   # boilerplate, duplicates, empties (try examples/tiny/boilerplate.jsonl)
ft27b sample data/all.jsonl -n 20               # READ them. Most bugs are found here.
ft27b split data/all.jsonl                      # held out by [data] split_key / eval_groups
ft27b shift                                     # do train targets look like held-out ones?

# before training: will the result fit on the Mac?
ft27b size ~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/<rev>

# 3-4. train on the GPU node(s): see docs/OPERATIONS.md first
scripts/sync-to-node.sh user@node               # code + config + data to the node
RUN=smoke TRAIN_EXTRA="--limit 40 --max-steps 3" train/run-chain.sh   # on the node; read its sample
RUN=run-1 train/run-chain.sh                    # one node
RUN=run-1 PEER=... MASTER_ADDR=... IFACE=... train/run-chain-2node.sh  # two nodes
train/watch-run.sh                              # off-cluster alarm (and scripts/status.sh hourly)
PLAN=runs/night-plan.sh setsid nohup train/night-run.sh &   # optional: run inside a nightly maintenance window
GO_NOW=1 MAX_HOURS=6 PLAN=... setsid nohup train/night-run.sh &   # ...or now ("go for it"), same reporting
scripts/night-reporter.sh start                 # optional: email PROBLEM / Report / FINISHED (NOTIFY_CMD)
scripts/queue-runner.sh start                   # optional: work through QUEUE.toml, step after step

# 5. evaluate
python3 train/score.py runs/run-1/compare.jsonl
python3 train/slice.py runs/run-1/compare.jsonl --primary-marker '...' --context-marker '...'  # where it fails
python3 train/paired_stats.py A.jsonl:v1 B.jsonl:v2 --primary-marker '...'  # is the difference bigger than the noise?
JUDGE_URL=... python3 train/judge_check.py runs/run-1/compare.jsonl 20 runs/run-1/judge_check.jsonl
JUDGE_URL=... python3 train/judge.py runs/run-1/compare.jsonl runs/run-1/judge.jsonl --pair base_sampled,tuned_sampled

# 6-7. convert on the Mac, then try it
PYTHON=... scripts/convert-mlx.sh --adapter runs/run-1/adapter-final --out ~/models/my-model --bits 8
ft27b infer --backend mlx --model ~/models/my-model/mlx-8bit --input-file example.txt
```

## Layout

| path | what |
| --- | --- |
| `project.toml` | **the one config**: base model, system prompt, LoRA, data, training, eval, conversion |
| `ft27b/` | stdlib-only core: `config`, `template` (the one prompt format), `data` (validate / split / sample), `sizing` (Mac fit calculator), `order` (resumable data order), `infer` (cuda / mlx), CLI |
| `train/` | `sft_lora.py` (BF16 LoRA, 1–2 nodes, memory cap, resume), `compare.py`, `score.py`, `slice.py` (metrics per input slice), `paired_stats.py` (paired CIs and the noise floor), `judge.py`, `judge_check.py`, `merge_lora.py`, `run-chain*.sh`, `watch-run.sh`, `night-run.sh` + `window-lib.sh` (optional overnight maintenance window), `workqueue.py` (the improvement queue) |
| `scripts/` | `gate.sh`, `secret-scan.sh`, `status.sh` (hourly status), `night-reporter.sh` + `notify.sh` (optional email reports for unattended runs), `queue-runner.sh` (runs `QUEUE.toml`), `sync-to-node.sh`, `convert-mlx.sh`; `.status.env.example` |
| `docs/` | `METHOD.md` (start here; §4 is how to go from one version to the next), `OPERATIONS.md`, `CONVERSION.md`, `HARDWARE.md`, `PLAN-template.md`, `REPORT-template.md` |
| `tests/` | hermetic tests pinning each lesson; real Qwen3.8-27B config fixtures |

## Measured defaults (from lkml-reviewer)

| | |
|---|---|
| training, 1× GB10 | ~163 tok/s, 71.5 GB peak at 4,096 tokens |
| training, 2× GB10 | 258–296 tok/s (1.6–1.7×), NCCL over TCP |
| MLX 8-bit on a 64 GB M4 Pro | 27 GB on disk, 30.2 GB peak, 8.5–8.8 tok/s |
| adapter → Mac model | under 4 minutes once the base is downloaded |

## The five things that matter most

1. **Read the data by eye** after every change. Counts don't find bugs.
2. **One prompt format** for training and inference (`ft27b/template.py`).
3. **Hold out by group**, evaluate **sampled** generations, and **validate the judge**.
4. **Restore what you stop**, cap memory, checkpoint for resume, watch from
   elsewhere (`docs/OPERATIONS.md`).
5. **Merge in BF16, then quantize**, and gate the Mac model against the CUDA one.
6. **Slice before you change anything.** Aggregates hide where a model fails;
   check train-vs-held-out distributions (`ft27b shift`) and per-slice
   grounding (`train/slice.py`), then change one thing per finding.
7. **Measure the noise floor, and probe at inference first.** Several samples
   per input, paired CIs (`train/paired_stats.py`); try a lever without
   training before spending a night on it.

## Licence

MIT. Base models carry their own licences (Qwen3.8-27B: Apache-2.0).
Datasets and weights never go in the repo (`.gitignore`).
