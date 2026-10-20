#!/usr/bin/env bash
# Manual native-checkpoint evaluation. No training or branch interventions.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 || ( $# -eq 2 && "$2" != --check ) ]]; then
  echo 'Usage: bash tools/run_eval_future_memory_occ_t0_t1.sh t0|t1 [--check]' >&2
  exit 2
fi
EXPERIMENT="$1"
case "$EXPERIMENT" in
  t0|t1) ;;
  *) echo 'Choose t0 or t1' >&2; exit 2 ;;
esac

cd /data/jxy/projects
export PYTHONPATH=/data/jxy/projects
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=backend:native,max_split_size_mb:128
PY=/data/jxy/projects/env/bin/python
NAME=sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center
CONFIG_NAME="${NAME}.py"
if [[ "$EXPERIMENT" == t1 ]]; then
  CONFIG_NAME="${NAME}-t1.py"
fi
RUN="work_dirs/${NAME}-${EXPERIMENT}-from-center2"
# Use the effective config written by tools/train.py for this run. MMCV
# BaseRunner.resume replaces self.meta with the source checkpoint metadata;
# meta.config can therefore still describe the ancestral, pre-center A+B run.
CONFIG="$RUN/$CONFIG_NAME"
CHECKPOINT="$RUN/epoch_3.pth"
OUT="$RUN/eval_epoch_3"
REFERENCE="work_dirs/${NAME}-from-ab1/eval_epoch_2/metrics.json"
for required in "$CONFIG" "$CHECKPOINT" "$REFERENCE"; do
  if [[ ! -s "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done
if [[ -e "$OUT" ]]; then
  echo "Output already exists; refusing to overwrite: $OUT" >&2
  echo 'Inspect existing metrics/dumps before attempting another evaluation.' >&2
  exit 1
fi

# --check is CPU-only, performs no data/model iteration and creates no output.
"$PY" - "$CONFIG" "$CHECKPOINT" "$EXPERIMENT" <<'PY'
import sys
import torch
from mmcv import Config

config_path, checkpoint_path, experiment = sys.argv[1:]
cfg = Config.fromfile(config_path)
q = cfg.model.query_memory_cfg
v = q.future_memory_adapter_v2
assert q.future_memory_adapter_version == 'v2' and q.source == 'cache'
assert v.gate_mode == 'task_split' and v.horizon_gate_bias is True
assert v.soft_voxel_center_offset == .5 and v.soft_voxel_renormalize_valid_corners is True
checkpoint = torch.load(checkpoint_path, map_location='cpu')
assert checkpoint['meta']['epoch'] == 3, checkpoint['meta']['epoch']
assert checkpoint['meta']['iter'] > 0
expected_beta = {'1s': 0., '2s': 0., '3s': .25} if experiment == 't1' else {}
assert v.get('occupancy_positive_class_balance_beta', {}) == expected_beta
assert q.freeze_base_model is True and cfg.runner.max_epochs == 3
print(experiment.upper(), 'native epoch 3 checkpoint / config: PASS')
print('Using saved effective run config:', config_path)
print('Epoch:', checkpoint['meta']['epoch'], 'iter:', checkpoint['meta']['iter'])
print('No semantic, occupancy, position or gate intervention; 4219-frame evaluator.')
PY
if [[ "${2:-}" == --check ]]; then
  exit 0
fi

# Do not share the GPUs with other training/evaluation jobs or stop them.
used=$(nvidia-smi --id=0,1 --query-gpu=memory.used --format=csv,noheader,nounits)
while IFS= read -r mib; do
  if (( mib > 1500 )); then
    echo "GPU busy (${mib} MiB); wait until both selected GPUs are free." >&2
    exit 1
  fi
done <<< "$used"
mkdir "$OUT"

"$PY" - "$CONFIG" "$CHECKPOINT" "$OUT" "$EXPERIMENT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
from mmcv import Config

config, checkpoint, output, experiment = sys.argv[1:]
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
paths = [config, checkpoint, 'tools/test_temporal.py', 'tools/eval_dumped_temporal.py',
         'mmdet3d/models/sparsedetectors/query_memory.py',
         'mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py',
         'mmdet3d/models/sparsedetectors/opus_head.py']
root = Path(output)
record = dict(experiment=experiment, checkpoint=checkpoint, seed=0,
              mode='native checkpoint; dump-only then offline evaluation',
              overrides=dict(samples_per_gpu=1, workers_per_gpu=2, query_chunk_size=48),
              sha256={p: sha256(p) for p in paths})
(root / 'provenance.json').write_text(json.dumps(record, indent=2))
Config.fromfile(config).dump(str(root / 'eval_config.py'))
PY

/data/jxy/projects/env/bin/torchrun \
  --nproc_per_node=2 --master_port=29556 \
  tools/test_temporal.py "$CONFIG" "$CHECKPOINT" \
  --launcher pytorch --seed 0 --dump-only --dump_dir "$OUT" \
  --cfg-options data.test_dataloader.samples_per_gpu=1 \
  data.test_dataloader.workers_per_gpu=2 \
  model.query_memory_cfg.future_memory_adapter_v2.query_chunk_size=48 \
  2>&1 | tee "$OUT/console.log"
touch "$OUT/.dump_complete"

"$PY" tools/eval_dumped_temporal.py \
  "$CONFIG" "$OUT" --output "$OUT/metrics.pending.json" \
  2>&1 | tee "$OUT/metrics-console.log"
mv "$OUT/metrics.pending.json" "$OUT/metrics.json"

"$PY" - "$NAME" <<'PY' | tee "$OUT/comparison.log"
import json
from pathlib import Path
import sys

name = sys.argv[1]
root = Path('work_dirs')
reference = root / (name + '-from-ab1') / 'eval_epoch_2/metrics.json'
paths = [('Center epoch 2', reference)]
paths += [(f'{e.upper()} epoch 3', root / (name + f'-{e}-from-center2') /
           'eval_epoch_3/metrics.json') for e in ('t0', 't1')]
baseline = json.loads(reference.read_text())
for label, path in paths:
    if not path.is_file():
        print(label, 'not evaluated yet')
        continue
    metrics = json.loads(path.read_text())
    print(label, path)
    for key in ('IoU', 'mIoU'):
        values = metrics[key]
        assert len(values) == 4
        mean = sum(values[1:4]) / 3.
        delta = mean - sum(baseline[key][1:4]) / 3.
        print(f'  {key}: {values}; Future Mean={mean:.4f}; delta vs center2={delta:+.4f} pp')
PY
printf '\nCompleted native evaluation: %s/metrics.json\n' "$OUT"
