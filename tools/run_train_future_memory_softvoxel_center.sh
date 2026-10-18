#!/usr/bin/env bash
set -euo pipefail
cd /data/jxy/projects
export PYTHONPATH="/data/jxy/projects${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

CONFIG=configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center.py
SOURCE=work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-from-occfix11/epoch_1.pth
EXPERIMENT_DIR=work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-from-ab1
test -f "$SOURCE"
test -f "$CONFIG"
test ! -e "$EXPERIMENT_DIR"

/data/jxy/projects/env/bin/python -c \
  'from mmcv import Config; import sys; c=Config.fromfile(sys.argv[1]); v=c.model.query_memory_cfg.future_memory_adapter_v2; assert c.resume_from==sys.argv[2] and c.load_from is None; assert v.gate_mode=="task_split" and v.soft_voxel_center_offset==0.5 and v.soft_voxel_renormalize_valid_corners is True; assert c.runner.max_epochs==2; print("SoftVoxel centre experiment config: PASS")' \
  "$CONFIG" "$SOURCE"
/data/jxy/projects/env/bin/python -c \
  'import sys, torch; c=torch.load(sys.argv[1], map_location="cpu"); assert c["meta"]["epoch"]==1, c["meta"].get("epoch"); assert c["meta"]["iter"]>0; assert "optimizer" in c; print("A+B epoch 1 model/optimizer checkpoint: PASS")' \
  "$SOURCE"
if [[ "${1:-}" == --check ]]; then
  exit 0
fi
if [[ $# -gt 0 ]]; then
  echo 'Usage: bash tools/run_train_future_memory_softvoxel_center.sh [--check]' >&2
  exit 2
fi
used=$(/usr/bin/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
while IFS= read -r mib; do
  if (( mib > 1500 )); then
    echo "GPU is busy (${mib} MiB used); wait until both selected GPUs are free" >&2
    exit 1
  fi
done <<< "$used"
mkdir -p "$EXPERIMENT_DIR"

/data/jxy/projects/env/bin/torchrun \
  --nproc_per_node=2 --master_port=29552 \
  tools/train.py "$CONFIG" \
  --work-dir "$EXPERIMENT_DIR" --resume-from "$SOURCE" \
  --launcher pytorch --seed 17 \
  2>&1 | tee "$EXPERIMENT_DIR/console.log"
