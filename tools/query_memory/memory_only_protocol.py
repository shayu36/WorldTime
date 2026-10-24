"""T1 Memory-only repair. Current-frame Baseline inputs remain LEGACY.

No module is trained or dataset read on import. Used by the independent
config, cache preparation, and CPU preflight checks.
"""
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARENT = 'configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1.py'
CONFIG = PARENT.replace('-t1.py', '-t1-memory-causal.py')
SOURCE = 'work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1-from-center2/epoch_3.pth'
SOURCE_SHA256 = 'bad9713741aa31373102e9769513ee52832359388990bfb9b365e9e9da40057c'
CACHE_SOURCE = 'ckpts/epoch_56.pth'
CACHE_SOURCE_SHA256 = '376df7a785b76d72f2aefb9c75fea8e1120dfebb49f3b7db2eaad84dd36c3015'
EGO_INPUT = 'data/ego_causal_v1/ego_inputs.pkl'
EGO_SHA256 = 'a059d6affc5ccd8e5e6182c375155af5d6bc2a721ccef81d7edc0c9b6006dea3'
CACHE_ROOT = 'data/query_memory/t1_memory_causal_v1'
WORK_DIR = 'work_dirs/t1-memory-causal-from-best3'


def transforms(pipeline):
    for item in pipeline:
        yield item
        yield from transforms(item.get('transforms', []))


def memory_only_data():
    from mmcv import Config
    data = deepcopy(Config.fromfile(str(ROOT / PARENT)).data)
    for split in ('train', 'val', 'test'):
        ds = data[split]
        # Do NOT set ego_gt_path or causal_v1 on the online model dataset.
        ds.ego_input_mode = 'legacy'
        ds.query_memory_require_causal_inputs = True
        cache = CACHE_ROOT + ('/train' if split == 'train' else '/val')
        ds.query_memory_cache_root = cache
        ds.query_memory_strict = True
        found = 0
        for item in transforms(ds.pipeline):
            if item['type'] == 'LoadQueryMemoryFromFiles':
                item.update(cache_root=cache, strict=True, require_causal_inputs=True,
                            expected_ego_input_sha256=EGO_SHA256,
                            expected_source_checkpoint_sha256=CACHE_SOURCE_SHA256)
                found += 1
        if found != 1:
            raise ValueError('Expected exactly one Memory loader in ' + split)
    return data


def cache_generation_config():
    """Separate offline source dataset; never reused for online train/eval."""
    from mmcv import Config
    cfg = Config.fromfile(str(ROOT / PARENT))
    # Use deterministic test image transforms for BOTH source splits.
    # GT may be read by the dataset for bookkeeping, but RAP consumes only
    # source images and temporal_ego_states[0], never supervision dictionaries.
    image_pipeline = deepcopy(cfg.data.test.pipeline)
    for item in transforms(image_pipeline):
        if item['type'] == 'LoadMultiViewImageFromMultiSweeps':
            item['test_mode'] = True
        if item['type'] == 'RandomTransformImage':
            item['training'] = False
    for split in ('train', 'val', 'test'):
        ds = cfg.data[split]
        ds.pipeline = deepcopy(image_pipeline)
        ds.ego_input_mode = 'causal_v1'
        ds.ego_gt_path = str(ROOT / EGO_INPUT)
        ds.ego_input_sha256 = EGO_SHA256
        ds.query_memory_require_causal_inputs = True
    cfg.memory_generation_only = True
    cfg.load_from = CACHE_SOURCE
    cfg.resume_from = None
    return cfg


def assert_memory_only_config(cfg):
    """Fail if a future edit silently changes the frozen Baseline protocol."""
    from mmcv import Config
    parent = Config.fromfile(str(ROOT / PARENT))
    if cfg.model != parent.model:
        raise ValueError('Memory-only repair must not change model/loss/threshold configuration')
    if cfg.data != memory_only_data():
        raise ValueError('Unexpected online data change; current ego must stay legacy')
    if (cfg.load_from != SOURCE or cfg.resume_from is not None or cfg.auto_resume
            or cfg.runner.max_epochs != 3):
        raise ValueError('Expected fresh optimizer, best epoch 3 initialization, three local epochs')
    if not cfg.model.query_memory_cfg.freeze_base_model:
        raise ValueError('Baseline must stay frozen')
