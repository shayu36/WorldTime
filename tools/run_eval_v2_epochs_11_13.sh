#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

CONFIG="configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-future-adapter-v2.py"
WORK_DIR="work_dirs/sparseworld-traj-memory-future-adapter-v2-native-q48"

for E in 11 12 13; do
    CKPT="$WORK_DIR/epoch_${E}.pth"
    OUT="$WORK_DIR/eval_epoch_${E}"
    test -f "$CKPT"
    mkdir -p "$OUT"

    "$REPO_ROOT/env/bin/torchrun" \
        --nproc_per_node=2 \
        --master_port=29540 \
        tools/test_temporal.py \
        "$CONFIG" "$CKPT" \
        --launcher pytorch \
        --dump-only \
        --dump_dir "$OUT" \
        --cfg-options \
        data.test_dataloader.samples_per_gpu=1 \
        data.test_dataloader.workers_per_gpu=2 \
        model.query_memory_cfg.future_memory_adapter_v2.query_chunk_size=48 \
        2>&1 | tee -a "$OUT/console.log"

    "$REPO_ROOT/env/bin/python" tools/eval_dumped_temporal.py \
        "$CONFIG" "$OUT" \
        --output "$OUT/metrics.json"
done
