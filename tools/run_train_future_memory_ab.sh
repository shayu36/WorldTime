#!/usr/bin/env bash
set -euo pipefail
cd /data/jxy/projects
export PYTHONPATH="/data/jxy/projects${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
# Native allocator avoids the cudaMallocAsync record_stream warning flood.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
AB_DIR=work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-from-occfix11
test -f "$AB_DIR/init_ab_from_occfix_epoch11.pth"
mkdir -p "$AB_DIR"
# Fail rather than silently restarting over an existing experiment.
test ! -e "$AB_DIR/console.log"
/data/jxy/projects/env/bin/torchrun \
  --nproc_per_node=2 --master_port=29542 \
  tools/train.py \
  configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-future-adapter-v2-ab.py \
  --work-dir "$AB_DIR" --launcher pytorch --seed 17 \
  2>&1 | tee "$AB_DIR/console.log"
