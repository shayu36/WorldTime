"""Prepare/check Memory-only T1 adaptation; never runs training or GPU inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.query_memory.memory_only_protocol import (
    CONFIG, SOURCE, SOURCE_SHA256, CACHE_SOURCE, CACHE_SOURCE_SHA256,
    EGO_INPUT, EGO_SHA256, CACHE_ROOT, WORK_DIR,
    cache_generation_config, assert_memory_only_config)

PREPARATION = ROOT / 'work_dirs/t1-memory-causal-preparation'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def verify_cache(split):
    root = ROOT / CACHE_ROOT / split
    complete = json.loads((root / 'complete.json').read_text())
    manifest = json.loads((root / 'generation.json').read_text())
    if complete['generation_sha256'] != sha256(root / 'generation.json'):
        raise ValueError('Cache completion/generation identity mismatch')
    if (not complete['full_dataset'] or complete['count'] != complete['dataset_count']
            or complete['count'] != manifest['dataset_count']
            or len(set(complete['files'])) != complete['count']):
        raise ValueError('Cache incomplete: diagnostic subsets are NOT training cache')
    for field, expected in [('ego_input_sha256', EGO_SHA256),
                            ('source_checkpoint_sha256', CACHE_SOURCE_SHA256),
                            ('split', 'train' if split == 'train' else 'test')]:
        if manifest[field] != expected: raise ValueError('Cache source mismatch: ' + field)
    if manifest['generator_sha256'] != sha256(ROOT / 'tools/query_memory/precompute_query_memory.py'):
        raise ValueError('Cache generator changed since generation')
    for path, expected in manifest['source_sha256'].items():
        if sha256(ROOT / path) != expected:
            raise ValueError('Cache source code changed: ' + path)
    for name in complete['files']:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('Missing/empty cache file: ' + str(path))
    return dict(count=complete['count'], manifest_sha256=complete['generation_sha256'])


def prepare(require_cache=False):
    import torch
    from mmcv import Config
    from mmdet3d.datasets.causal_ego import load_artifact
    from mmdet3d.models.sparsedetectors.query_memory import FutureMemoryAdapterV2
    cfg = Config.fromfile(str(ROOT / CONFIG))
    assert_memory_only_config(cfg)
    for path, expected in [(SOURCE, SOURCE_SHA256), (CACHE_SOURCE, CACHE_SOURCE_SHA256),
                            (EGO_INPUT, EGO_SHA256)]:
        if sha256(ROOT / path) != expected: raise ValueError('Source hash mismatch: ' + path)
    artifact, _ = load_artifact(ROOT / EGO_INPUT)
    for entry in artifact['manifest']['inputs'].values():
        if sha256(ROOT / entry['path']) != entry['sha256']:
            raise ValueError('Causal ego metadata source changed: ' + entry['path'])
    current = torch.load(ROOT / SOURCE, map_location='cpu')
    baseline = torch.load(ROOT / CACHE_SOURCE, map_location='cpu')['state_dict']
    state = current['state_dict']
    assert current['meta']['epoch'] == 3
    for k, tensor in baseline.items():
        if k not in state or not torch.equal(tensor, state[k]):
            raise ValueError('Source Baseline weight differs: ' + k)
    adapter = FutureMemoryAdapterV2(**dict(cfg.model.query_memory_cfg.future_memory_adapter_v2))
    prefix = 'future_memory_adapter_v2.'
    adapter.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    assert all(p.grad is None for p in adapter.parameters())
    PREPARATION.mkdir(parents=True, exist_ok=True)
    generated = cache_generation_config().pretty_text
    dest = PREPARATION / 'cache_config.py'
    if dest.exists():
        if dest.read_text() != generated:
            raise ValueError('Prepared cache config changed; retain old preparation and choose a new version')
    else:
        dest.write_text(generated)
    report = dict(protocol='causal_history_only__legacy_current_baseline',
        source_checkpoint=SOURCE, source_checkpoint_sha256=SOURCE_SHA256,
        cache_source_checkpoint=CACHE_SOURCE, cache_source_sha256=CACHE_SOURCE_SHA256,
        causal_ego_sha256=EGO_SHA256, ego_records=len(artifact['records']),
        unchanged_baseline_tensors=len(baseline), adapter_strict_load=True,
        trainable_prefix=prefix, optimizer='fresh AdamW, fixed 2e-5, 100 warmup iters',
        local_epochs=cfg.runner.max_epochs, output=WORK_DIR,
        cache_generation_config=str(dest.relative_to(ROOT)),
        full_model_gpu_verified=False, cache_ready=False)
    if require_cache:
        report['cache'] = {s: verify_cache(s) for s in ('train', 'val')}
        report['cache_ready'] = True
    (PREPARATION / 'last_check.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-cache', action='store_true')
    args = parser.parse_args()
    prepare(args.require_cache)
