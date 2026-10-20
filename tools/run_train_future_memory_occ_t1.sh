#!/usr/bin/env bash
# Manual launcher only. Never starts T0, stops another job, or overwrites a run.
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --check ) ]]; then
  echo 'Usage: bash tools/run_train_future_memory_occ_t1.sh [--check]' >&2
  exit 2
fi
cd /data/jxy/projects
export PYTHONPATH=/data/jxy/projects
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=backend:native,max_split_size_mb:128

PY=/data/jxy/projects/env/bin/python
NAME=sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center
CONFIG="configs/sparseworld/nuscenes-temporal/${NAME}-t1.py"
SOURCE="work_dirs/${NAME}-from-ab1/epoch_2.pth"
EXPERIMENT_DIR="work_dirs/${NAME}-t1-from-center2"
test -s "$CONFIG"
test -s "$SOURCE"
if [[ -e "$EXPERIMENT_DIR" ]]; then
  echo "Output already exists; refusing to overwrite: $EXPERIMENT_DIR" >&2
  exit 1
fi

"$PY" - "$CONFIG" "$SOURCE" "$EXPERIMENT_DIR" <<'PY'
import copy
import importlib.util
import sys
import torch
from mmcv import Config

path, source, output = sys.argv[1:]
cfg = Config.fromfile(path)
reference = Config.fromfile(path.replace('-center-t1.py', '-center.py'))
v = cfg.model.query_memory_cfg.future_memory_adapter_v2
expected = {'1s': 0.0, '2s': 0.0, '3s': 0.25}
assert v.occupancy_positive_class_balance_beta == expected
assert v.occupancy_class_balance_diagnostics is True
matched_model = copy.deepcopy(cfg.model)
del matched_model.query_memory_cfg.future_memory_adapter_v2.occupancy_positive_class_balance_beta
del matched_model.query_memory_cfg.future_memory_adapter_v2.occupancy_class_balance_diagnostics
assert matched_model == reference.model, 'Unexpected model/loss changes beyond T1'
assert cfg.resume_from == source and cfg.load_from is None and not cfg.auto_resume
assert cfg.work_dir == output and cfg.runner.max_epochs == 3
for key in ('optimizer', 'optimizer_config', 'lr_config', 'data', 'fp16', 'workflow'):
    assert cfg.get(key) == reference.get(key), key
assert cfg.model.query_memory_cfg.freeze_base_model is True
checkpoint = torch.load(source, map_location='cpu')
assert checkpoint['meta']['epoch'] == 2
assert checkpoint['meta']['iter'] > 0
assert checkpoint['optimizer']['state'], 'Missing optimizer moments: cannot do matched resume'
assert checkpoint['optimizer']['param_groups'], 'Missing optimizer parameter groups'
spec = importlib.util.spec_from_file_location(
    't1_query_memory', 'mmdet3d/models/sparsedetectors/query_memory.py')
qm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qm)
qm.validate_occupancy_class_balance_beta(v.occupancy_positive_class_balance_beta)
adapter = qm.FutureMemoryAdapterV2(**dict(v))
prefix = 'future_memory_adapter_v2.'
state = {k.removeprefix('module.'): value for k, value in checkpoint['state_dict'].items()}
adapter.load_state_dict({k[len(prefix):]: value for k, value in state.items()
                         if k.startswith(prefix)}, strict=True)
# The existing TrainableOnlyOptimizerConstructor creates one group per
# trainable parameter. Recreate that layout for a CPU-only state-shape check;
# do not instantiate the Baseline (OPUSHead.__init__ unconditionally uses CUDA).
optimizer = torch.optim.AdamW([{'params': [p]} for p in adapter.parameters()],
                             lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
optimizer.load_state_dict(checkpoint['optimizer'])
for param, moments in optimizer.state.items():
    for key in ('exp_avg', 'exp_avg_sq'):
        assert moments[key].shape == param.shape, key
assert all(p.grad is None for p in adapter.parameters())
print('T1 config / center epoch 2 / optimizer / matched settings: PASS')
print('CPU Adapter strict checkpoint load / optimizer group and moment shapes: PASS')
print('Resume epoch:', checkpoint['meta']['epoch'], 'iter:', checkpoint['meta']['iter'])
print('Only 3s positive occupancy BCE is changed; one additional epoch to epoch 3.')
print('Full model/GPU execution is NOT checked by --check.')
PY
if [[ "${1:-}" == --check ]]; then
  exit 0
fi

# Do not overlap T0 or evaluation on either GPU. --check above is CPU-only.
used=$(nvidia-smi --id=0,1 --query-gpu=memory.used --format=csv,noheader,nounits)
while IFS= read -r mib; do
  if (( mib > 1500 )); then
    echo "GPU busy (${mib} MiB); wait for T0/evaluation to finish." >&2
    exit 1
  fi
done <<< "$used"
mkdir "$EXPERIMENT_DIR"

# Record the exact source/checkpoint hashes, not environment variables or secrets.
"$PY" - "$CONFIG" "$SOURCE" "$EXPERIMENT_DIR" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from mmcv import Config

config, source, output = sys.argv[1:]
def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
paths = [config, source, 'mmdet3d/models/sparsedetectors/query_memory.py',
         'mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py',
         'tools/run_train_future_memory_occ_t1.sh']
record = dict(experiment='T1', source_checkpoint=source, seed=17,
              resume_optimizer=True, source_epoch=2, max_epochs=3,
              beta_by_horizon={'1s': 0., '2s': 0., '3s': .25},
              git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              sha256={p: digest(p) for p in paths})
root = Path(output)
(root / 'launch_provenance.json').write_text(json.dumps(record, indent=2))
Config.fromfile(config).dump(str(root / 'launch_config.py'))
PY

/data/jxy/projects/env/bin/torchrun \
  --nproc_per_node=2 --master_port=29555 \
  tools/train.py "$CONFIG" \
  --work-dir "$EXPERIMENT_DIR" --resume-from "$SOURCE" \
  --launcher pytorch --seed 17 \
  2>&1 | tee "$EXPERIMENT_DIR/console.log"
