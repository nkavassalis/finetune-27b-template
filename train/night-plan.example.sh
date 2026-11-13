#!/usr/bin/env bash
# An example plan for train/night-run.sh. Copy it to runs/night-plan.sh, edit,
# then from the work dir on the master node:
#
#   PLAN=runs/night-plan.sh PEER=user@peer WAIT_FOR_NEXT_WINDOW=1 \
#     setsid nohup train/night-run.sh > /dev/null 2>&1 < /dev/null &
#
# Steps run in order. Each one checks it fits before the window's hard stop;
# a step that can't fit is skipped, never started. Chains restore serving
# themselves; night-run.sh double-checks at the end.
set -u
cd "$(dirname "$0")/.."
. train/window-lib.sh

common=(PEER="${PEER:?}" MASTER_ADDR="${MASTER_ADDR:?}" IFACE="${IFACE:?}" SERVE_CTR="${SERVE_CTR:-}")

# 1. Smoke test the whole chain at tiny sizes (~20 min). Stop the night if it fails:
#    no point spending the window on a run that will break at the copy step.
if need_minutes 30 "smoke test"; then
  env "${common[@]}" RUN=smoke-$(date +%F) TRAIN_EXTRA="--limit 160 --eval-limit 16 --max-steps 4 --n-samples 1" \
    EVAL_ARMS_MASTER=tuned EVAL_ADAPTERS_MASTER="tuned=runs/smoke-$(date +%F)/adapter-final" EVAL_SUBSET=2 \
    train/run-chain-2node.sh || { echo "smoke test failed; stopping the night"; exit 1; }
fi

# 2. The real run. --resume-auto continues from the last checkpoint if an earlier
#    night was cut off, so a run longer than one window finishes over several.
if need_minutes 60 "training run-1 (resumes if partial)"; then
  env "${common[@]}" RUN=run-1 TRAIN_EXTRA="--resume-auto" train/run-chain-2node.sh
fi

# 3. Evaluation, only once training has finished (adapter-final exists).
if [ -d runs/run-1/adapter-final ] && need_minutes 60 "evaluation"; then
  env "${common[@]}" RUN=run-1 EVAL_ONLY=1 EVAL_ARMS_MASTER=base EVAL_ARMS_PEER=tuned \
    EVAL_ADAPTERS_MASTER="tuned=runs/run-1/adapter-final" EVAL_ADAPTERS_PEER="tuned=runs/run-1-adapter-final" \
    train/run-chain-2node.sh
fi
