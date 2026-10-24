#!/usr/bin/env bash
# Manual actions only. No polling, auto-training, pkill, or old-output overwrite.
set -euo pipefail
cd /data/jxy/projects
export PYTHONPATH=/data/jxy/projects
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=backend:native,max_split_size_mb:128
PY=/data/jxy/projects/env/bin/python
CFG=configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1-memory-causal.py
RUN=work_dirs/t1-memory-causal-from-best3
PREP=work_dirs/t1-memory-causal-preparation

idle_gpu() {
  local used
  used=$(nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits)
  while IFS= read -r mib; do
    if (( mib > 1500 )); then
      echo "GPU $1 busy (${mib} MiB); no process was stopped." >&2; exit 1
    fi
  done <<< "$used"
}

case "${1:-check}" in
  check)
    CUDA_VISIBLE_DEVICES='' "$PY" tools/prepare_t1_memory_causal.py
    ;;
  cache)
    CUDA_VISIBLE_DEVICES='' "$PY" tools/prepare_t1_memory_causal.py
    idle_gpu 0
    # One GPU, serial train then val; generation uses source-time causal ego
    # only. Resumption validates every existing record, never trusts old cache.
    for SPLIT in train test; do
      OUT=data/query_memory/t1_memory_causal_v1/train
      if [[ "$SPLIT" == test ]]; then OUT=data/query_memory/t1_memory_causal_v1/val; fi
      CUDA_VISIBLE_DEVICES=0 "$PY" tools/query_memory/precompute_query_memory.py \
        --config "$PREP/cache_config.py" --checkpoint ckpts/epoch_56.pth \
        --split "$SPLIT" --output-dir "$OUT" --workers-per-gpu 1 --skip-existing \
        2>&1 | tee -a "$PREP/cache-${SPLIT}.log"
    done
    CUDA_VISIBLE_DEVICES='' "$PY" tools/prepare_t1_memory_causal.py --require-cache
    ;;
  train)
    CUDA_VISIBLE_DEVICES='' "$PY" tools/prepare_t1_memory_causal.py --require-cache
    if [[ -e "$RUN" ]]; then
      echo "Output already exists: $RUN; refusing to overwrite. Resume explicitly after review." >&2; exit 1
    fi
    idle_gpu 0,1
    mkdir "$RUN"
    cp "$PREP/last_check.json" "$RUN/initialization_provenance.json"
    CUDA_VISIBLE_DEVICES=0,1 /data/jxy/projects/env/bin/torchrun \
      --nproc_per_node=2 --master_port=29561 tools/train.py "$CFG" \
      --work-dir "$RUN" --launcher pytorch --seed 17 \
      2>&1 | tee "$RUN/console.log"
    ;;
  *) echo 'Usage: bash tools/run_t1_memory_causal.sh check|cache|train' >&2; exit 2 ;;
esac
