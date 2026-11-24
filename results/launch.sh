#!/usr/bin/env bash
set -uo pipefail
cd /data/jxy/projects
export PYTHONPATH=/data/jxy/projects
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=backend:native
/data/jxy/projects/env/bin/torchrun --nproc_per_node=2 --master_port=29547 tools/train.py configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-planning-fusion.py --work-dir /data/jxy/projects/work_dirs/memory-planning-fusion-from-t1-e3 --launcher pytorch --seed 17 --cfg-options runner.max_epochs=1 data.samples_per_gpu=1 data.workers_per_gpu=1 log_config.interval=10 2>&1 | tee -a /data/jxy/projects/work_dirs/memory-planning-fusion-from-t1-e3/console.log
run_status=${PIPESTATUS[0]}
printf "%s\n" "$run_status" > /data/jxy/projects/work_dirs/memory-planning-fusion-from-t1-e3/exit_code
exit "$run_status"
