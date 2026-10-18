#!/usr/bin/env bash
# Evaluate A+B local epochs 1/2/3 sequentially with bounded result memory.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

CONFIG=configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-future-adapter-v2-ab.py
AB_DIR=work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-from-occfix11
PYTHON="$REPO_ROOT/env/bin/python"
TORCHRUN="$REPO_ROOT/env/bin/torchrun"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --check ) ]]; then
    echo "Usage: bash tools/run_eval_future_memory_ab.sh [--check]" >&2
    exit 2
fi

test -x "$PYTHON"
test -x "$TORCHRUN"
test -f "$CONFIG"
test -f tools/eval_dumped_temporal.py
for E in 1 2 3; do
    test -f "$AB_DIR/epoch_${E}.pth" || {
        echo "Missing checkpoint: $AB_DIR/epoch_${E}.pth" >&2
        exit 1
    }
done

# Use the A+B model config, not the original shared-gate V2 config.
"$PYTHON" -c 'from mmcv import Config; import sys; c=Config.fromfile(sys.argv[1]); a=c.model.query_memory_cfg.future_memory_adapter_v2; assert a.gate_mode == "task_split" and a.horizon_gate_bias is True; assert a.query_chunk_size == 48; print("A+B config and epoch_1/2/3 checkpoints: PASS")' "$CONFIG"
if [[ "${1:-}" == --check ]]; then
    exit 0
fi

for E in 1 2 3; do
    CKPT="$AB_DIR/epoch_${E}.pth"
    OUT="$AB_DIR/eval_epoch_${E}"
    if [[ -s "$OUT/metrics.json" ]]; then
        echo "Existing metrics, skipping epoch $E: $OUT/metrics.json"
        continue
    fi
    mkdir -p "$OUT"
    if [[ ! -f "$OUT/.dump_complete" ]]; then
        echo "Starting A+B epoch $E prediction dump"
        "$TORCHRUN" \
            --nproc_per_node=2 --master_port=29543 \
            tools/test_temporal.py "$CONFIG" "$CKPT" \
            --launcher pytorch --seed 0 \
            --dump-only --dump_dir "$OUT" \
            --cfg-options \
            data.test_dataloader.samples_per_gpu=1 \
            data.test_dataloader.workers_per_gpu=2 \
            model.query_memory_cfg.future_memory_adapter_v2.query_chunk_size=48 \
            2>&1 | tee -a "$OUT/console.log"
        touch "$OUT/.dump_complete"
    fi
    echo "Computing A+B epoch $E validation metrics"
    "$PYTHON" tools/eval_dumped_temporal.py "$CONFIG" "$OUT" \
        --output "$OUT/metrics.pending.json" \
        2>&1 | tee -a "$OUT/metrics-console.log"
    mv "$OUT/metrics.pending.json" "$OUT/metrics.json"
done

"$PYTHON" - "$AB_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print('checkpoint  Future_IoU  Future_mIoU  delta_IoU  delta_mIoU')
baseline_path = root.parent / 'sparseworld-traj-memory-future-adapter-v2-occfix-q48/eval_epoch_11/metrics.json'
with baseline_path.open() as stream:
    baseline = json.load(stream)
reference = [sum(baseline[key][1:4]) / 3 for key in ('IoU', 'mIoU')]
for epoch in (1, 2, 3):
    with (root / f'eval_epoch_{epoch}/metrics.json').open() as stream:
        result = json.load(stream)
    means = [sum(result[key][1:4]) / 3 for key in ('IoU', 'mIoU')]
    print(f'epoch_{epoch}    {means[0]:.4f}     {means[1]:.4f}      '
          f'{means[0]-reference[0]:+.4f}    {means[1]-reference[1]:+.4f}')
    print('  IoU [0s,1s,2s,3s]:', result['IoU'])
    print(' mIoU [0s,1s,2s,3s]:', result['mIoU'])
PY
