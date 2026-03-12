#!/bin/bash

# Run two smallrun training jobs in parallel, each on 2 GPUs.
# Usage:
#   bash runs/smallrun_parallel.sh <commit1> <tag1> <commit2> <tag2>
# Example:
#   bash runs/smallrun_parallel.sh 39a5f83 d12-relu-x3 2939823 d12-relu-x5

set -euo pipefail

if [ $# -ne 4 ]; then
    echo "Usage: $0 <commit1> <tag1> <commit2> <tag2>"
    exit 1
fi

COMMIT1=$1; TAG1=$2
COMMIT2=$3; TAG2=$4

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

# Activate venv
source .venv/bin/activate

DEPTH=${DEPTH:-12}

# Create separate worktrees for each commit
WORKTREE1=$(mktemp -u)
WORKTREE2=$(mktemp -u)
echo "Worktree 1: $WORKTREE1 (commit $COMMIT1, tag $TAG1)"
echo "Worktree 2: $WORKTREE2 (commit $COMMIT2, tag $TAG2)"

git worktree add "$WORKTREE1" "$COMMIT1"
git worktree add "$WORKTREE2" "$COMMIT2"

# Run training in worktree with specified GPUs
run_train() {
    local worktree=$1 gpus=$2 tag=$3 master_port=$4 logfile=$5
    (
        cd "$worktree"
        CUDA_VISIBLE_DEVICES=$gpus \
        NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR" \
        OMP_NUM_THREADS=1 \
        torchrun --standalone --nproc_per_node=2 --master_port=$master_port \
            -m scripts.base_train -- \
            --depth=$DEPTH \
            --target-param-data-ratio=9.5 \
            --device-batch-size=16 \
            --fp8 \
            --run=$tag \
            --window-pattern L \
            --save-every=1000 \
            --attention-type=linear \
            --model-tag=$tag
    ) > "$logfile" 2>&1
}

LOG1="runs/${TAG1}.log"
LOG2="runs/${TAG2}.log"

echo "Starting $TAG1 on GPUs 0,1..."
run_train "$WORKTREE1" "0,1" "$TAG1" 29500 "$LOG1" &
PID1=$!

echo "Starting $TAG2 on GPUs 2,3..."
run_train "$WORKTREE2" "2,3" "$TAG2" 29501 "$LOG2" &
PID2=$!

echo "Training PIDs: $TAG1=$PID1, $TAG2=$PID2"
echo "Logs: tail -f $LOG1 | tail -f $LOG2"

# Wait for both
FAIL=0
wait $PID1 || { echo "FAILED: $TAG1 (see $LOG1)"; FAIL=1; }
wait $PID2 || { echo "FAILED: $TAG2 (see $LOG2)"; FAIL=1; }

# Cleanup worktrees
git worktree remove "$WORKTREE1" --force 2>/dev/null || true
git worktree remove "$WORKTREE2" --force 2>/dev/null || true

if [ $FAIL -ne 0 ]; then
    echo "One or more training runs failed. Check logs."
    exit 1
fi

echo "Both training runs complete!"
echo "Run evals with: python experiments/eval.py --bench --sample"
