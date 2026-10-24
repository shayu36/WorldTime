#!/usr/bin/env bash
# Manual launcher; never stops other GPU jobs or overwrites old outputs.
set -euo pipefail
cd /data/jxy/projects
export PYTHONPATH=/data/jxy/projects
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=backend:native,max_split_size_mb:128
PY=/data/jxy/projects/env/bin/python
NAME=sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center
RUN="work_dirs/${NAME}-t1-from-center2"
OUT="$RUN/planning_epoch_3"
ARGS=(--config "$RUN/${NAME}-t1.py" --checkpoint "$RUN/epoch_3.pth"
      --occupancy-provenance "$RUN/eval_epoch_3/provenance.json" --output-dir "$OUT")
case "${1:-run}" in
  --check) CUDA_VISIBLE_DEVICES='' "$PY" tools/eval_future_memory_planning.py check "${ARGS[@]}"; exit ;;
  --metrics-only) CUDA_VISIBLE_DEVICES='' "$PY" tools/eval_future_memory_planning.py metrics "${ARGS[@]}" 2>&1 | tee -a "$RUN/planning_epoch_3-metrics.log"; exit ;;
  run) ;;
  *) echo 'Usage: bash tools/run_eval_future_memory_t1_planning.sh [--check|--metrics-only]' >&2; exit 2 ;;
esac
CUDA_VISIBLE_DEVICES='' "$PY" tools/eval_future_memory_planning.py check "${ARGS[@]}"
if [[ -e "$OUT" ]]; then
  echo "Output exists; refusing to overwrite: $OUT" >&2; exit 1
fi
used=$(nvidia-smi --id=0,1 --query-gpu=memory.used --format=csv,noheader,nounits)
while IFS= read -r mib; do
  if (( mib > 1500 )); then
    echo "GPU busy (${mib} MiB); no evaluation started. Run later when both GPUs are free." >&2; exit 1
  fi
done <<< "$used"
/data/jxy/projects/env/bin/torchrun --nproc_per_node=2 --master_port=29558 \
  tools/eval_future_memory_planning.py infer "${ARGS[@]}" 2>&1 | tee -a "$RUN/planning_epoch_3-inference.log"
CUDA_VISIBLE_DEVICES='' "$PY" tools/eval_future_memory_planning.py metrics "${ARGS[@]}" \
  2>&1 | tee -a "$RUN/planning_epoch_3-metrics.log"
