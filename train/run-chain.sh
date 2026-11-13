#!/bin/bash
# One-node chain: stop the serving container -> train -> compare -> restore.
# The restore runs in an EXIT trap, so a crash, OOM or kill can't leave the
# service down. Run on the training node from the synced work dir, detached:
#
#   RUN=run-1 setsid nohup train/run-chain.sh > /dev/null 2>&1 < /dev/null &
#
# Environment:
#   RUN          run name; outputs go to runs/$RUN (required)
#   TRAIN_EXTRA  extra sft_lora.py args (e.g. "--limit 40 --max-steps 3" for a smoke test)
#   EVAL_ARMS    compare.py arms, default "base,tuned"; empty skips evaluation
#   SERVE_CTR    serving container to stop/start (default: none; nothing is stopped)
#   TRAIN_CTR    training container (default ft27b-train)
set -u
: "${RUN:?set RUN, e.g. RUN=run-1}"
TRAIN_EXTRA=${TRAIN_EXTRA:-}
EVAL_ARMS=${EVAL_ARMS-base,tuned}
SERVE_CTR=${SERVE_CTR:-}
TRAIN_CTR=${TRAIN_CTR:-ft27b-train}
WORK=$(pwd)
mkdir -p "$WORK/runs"
LOG=$WORK/runs/chain-$RUN.log
say() { echo "[$(date +%T)] $*" >> "$LOG"; }

restore() {
  local status=$?
  say "exit status $status"
  docker exec "$TRAIN_CTR" pkill -f "sft_lora.py|compare.py" >> "$LOG" 2>&1 || true
  if [ -n "$SERVE_CTR" ]; then
    sleep 5
    docker start "$SERVE_CTR" >> "$LOG" 2>&1
    say "restore of $SERVE_CTR issued"
  fi
  say "chain finished"
}
trap restore EXIT
# A stop signal (the window's hard stop, a kill) becomes a normal exit, so the
# EXIT trap above runs and serving is restored.
trap 'exit 143' TERM INT
set -e

if [ -n "$SERVE_CTR" ]; then
  say "stopping $SERVE_CTR"
  docker stop -t 60 "$SERVE_CTR" >> "$LOG" 2>&1
fi

say "training $RUN"
docker exec -w /work "$TRAIN_CTR" python3 train/sft_lora.py --out "runs/$RUN" $TRAIN_EXTRA \
  > "$WORK/runs/$RUN.out" 2>&1

if [ -n "$EVAL_ARMS" ]; then
  say "comparing [$EVAL_ARMS]"
  docker exec -w /work "$TRAIN_CTR" python3 train/compare.py --eval data/sft_eval.jsonl \
    --subset 100 --modes sampled --arms "$EVAL_ARMS" \
    --adapters "tuned=runs/$RUN/adapter-final" --out "runs/$RUN/compare.jsonl" \
    > "$WORK/runs/$RUN.compare.out" 2>&1
fi
say "CHAIN DONE"
