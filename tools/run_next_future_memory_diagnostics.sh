#!/usr/bin/env bash
# One-line entry point for paired output interventions on fixed checkpoints.
set -euo pipefail
cd /data/jxy/projects
export PYTHONPATH="/data/jxy/projects${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

MODE="${1:-}"
CONFIG=configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-future-adapter-v2-ab.py
CHECKPOINT=work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-from-occfix11/epoch_1.pth
case "$MODE" in
  scan)
    EXPERIMENT=scan
    EXTRA=(--limit 256)
    OUTPUT=work_dirs/diagnostics-ab-miou-20261016/scan256
    ;;
  position-full)
    EXPERIMENT=position_full
    EXTRA=(--full)
    OUTPUT=work_dirs/diagnostics-ab-miou-20261016/position_full4219
    ;;
  center-position-full)
    EXPERIMENT=position_full
    EXTRA=(--full)
    CONFIG=configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center.py
    CHECKPOINT=work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-from-ab1/epoch_2.pth
    OUTPUT=work_dirs/diagnostics-center-position-20261018/full4219
    ;;
  *)
    echo 'Usage: bash tools/run_next_future_memory_diagnostics.sh scan|position-full|center-position-full [--check]' >&2
    exit 2
    ;;
esac

# Keep the default artifact immutable, while allowing a deliberate rerun in
# a new directory without editing this script or overwriting prior evidence.
OUTPUT="${FUTURE_MEMORY_DIAG_OUTPUT_DIR:-$OUTPUT}"

test -f "$CONFIG"
test -f "$CHECKPOINT"
if [[ "$MODE" == scan ]]; then
  test -f work_dirs/diagnostics-ab-miou-20261015/paired_subset256.json
fi
test ! -e "$OUTPUT"

/data/jxy/projects/env/bin/python -c \
  'from mmcv import Config; import sys, torch; c=Config.fromfile(sys.argv[1]); a=c.model.query_memory_cfg.future_memory_adapter_v2; assert a.gate_mode=="task_split" and a.horizon_gate_bias is True; checkpoint=torch.load(sys.argv[2], map_location="cpu"); assert checkpoint["meta"]["epoch"]==int(sys.argv[3]); assert (a.soft_voxel_center_offset==0.5 and a.soft_voxel_renormalize_valid_corners is True) if int(sys.argv[3])==2 else True; print("diagnostic config/checkpoint/output: PASS")' \
  "$CONFIG" "$CHECKPOINT" "$([[ "$MODE" == center-position-full ]] && echo 2 || echo 1)"
if [[ "${2:-}" == --check ]]; then
  exit 0
fi
if [[ $# -gt 2 ]]; then
  echo 'Too many arguments' >&2
  exit 2
fi

/data/jxy/projects/env/bin/python -u tools/diagnose_future_memory_branches.py \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --experiment "$EXPERIMENT" "${EXTRA[@]}" --gpu-id 0 --workers 1 \
  --output-dir "$OUTPUT"

/data/jxy/projects/env/bin/python tools/summarize_future_memory_branches.py \
  "$OUTPUT" --output-name summary.json
if [[ "$MODE" == scan ]]; then
  /data/jxy/projects/env/bin/python tools/summarize_future_memory_scan.py "$OUTPUT"
fi
