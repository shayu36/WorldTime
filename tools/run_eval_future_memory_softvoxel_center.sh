#!/usr/bin/env bash
# Full validation for the correctly resumed, one-epoch SoftVoxel experiment.
set -euo pipefail
cd /data/jxy/projects
export PYTHONPATH="/data/jxy/projects${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

CONFIG=configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center.py
RUN=work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-from-ab1
CHECKPOINT="$RUN/epoch_2.pth"
OUT="$RUN/eval_epoch_2"
REFERENCE=work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-from-occfix11/eval_epoch_2/metrics.json

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --check ) ]]; then
    echo 'Usage: bash tools/run_eval_future_memory_softvoxel_center.sh [--check]' >&2
    exit 2
fi
test -s "$CHECKPOINT"
test -s "$REFERENCE"
test -f "$CONFIG"
test ! -e "$OUT"
/data/jxy/projects/env/bin/python -c \
    'from mmcv import Config; import sys; c=Config.fromfile(sys.argv[1]); v=c.model.query_memory_cfg.future_memory_adapter_v2; assert v.gate_mode=="task_split" and v.soft_voxel_center_offset==0.5 and v.soft_voxel_renormalize_valid_corners is True; print("SoftVoxel evaluation config: PASS")' \
    "$CONFIG"
if [[ "${1:-}" == --check ]]; then
    exit 0
fi

mkdir -p "$OUT"
/data/jxy/projects/env/bin/torchrun \
    --nproc_per_node=2 --master_port=29554 \
    tools/test_temporal.py "$CONFIG" "$CHECKPOINT" \
    --launcher pytorch --seed 0 --dump-only --dump_dir "$OUT" \
    --cfg-options data.test_dataloader.samples_per_gpu=1 \
    data.test_dataloader.workers_per_gpu=2 \
    model.query_memory_cfg.future_memory_adapter_v2.query_chunk_size=48 \
    2>&1 | tee "$OUT/console.log"
touch "$OUT/.dump_complete"
/data/jxy/projects/env/bin/python tools/eval_dumped_temporal.py \
    "$CONFIG" "$OUT" --output "$OUT/metrics.pending.json" \
    2>&1 | tee "$OUT/metrics-console.log"
mv "$OUT/metrics.pending.json" "$OUT/metrics.json"
/data/jxy/projects/env/bin/python - "$REFERENCE" "$OUT/metrics.json" <<'PY'
import json
import sys
for label, path in zip(('A+B epoch 2', 'SoftVoxel epoch 2'), sys.argv[1:]):
    with open(path) as stream:
        metrics = json.load(stream)
    print(label)
    for name in ('IoU', 'mIoU'):
        values = metrics[name]
        print(f'  {name}: {values}; Future mean={sum(values[1:4])/3:.4f}')
PY
