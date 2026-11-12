#!/usr/bin/env bash
# Copy the repo (code, config, data/sft*.jsonl) to a GPU node's work directory,
# which is mounted at /work in the training container. Never copies runs/,
# .git or local secrets. Run before every launch so the node has exactly the
# code and config you mean to train with.
#
#   scripts/sync-to-node.sh user@node [workdir]      (default workdir: ft27b-train)
set -euo pipefail
node=${1:?usage: sync-to-node.sh user@node [workdir]}; work=${2:-ft27b-train}
root="$(cd "$(dirname "$0")/.." && pwd)"
ssh -o BatchMode=yes "$node" "mkdir -p ~/$work/data ~/$work/runs"
rsync -a --delete --exclude .git --exclude runs/ --exclude data/ --exclude '.status.env' \
  --exclude '.secret-watchlist' --exclude '__pycache__' "$root/" "$node:$work/"
rsync -a "$root"/data/sft*.jsonl "$node:$work/data/"
ssh -o BatchMode=yes "$node" "cd ~/$work && md5sum project.toml data/sft*.jsonl train/sft_lora.py" \
  | sed "s#^#$node: #"
