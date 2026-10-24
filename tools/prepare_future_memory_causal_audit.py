"""Prepare a bounded paired OOD audit, never starts a GPU job or training."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def causal_dataset_options(cfg, artifact, digest, checkpoint_digest, cache_root):
    """Effective config, pinned paths and hashes, no legacy cache fallback."""
    for split in ('train', 'val', 'test'):
        ds = cfg.data[split]
        ds.ego_input_mode = 'causal_v1'
        ds.ego_gt_path = str(artifact)
        ds.ego_input_sha256 = digest
        ds.query_memory_cache_root = str(cache_root)
        def patch(pipeline):
            for transform in pipeline:
                if transform['type'] == 'LoadQueryMemoryFromFiles':
                    transform.update(cache_root=str(cache_root), strict=True,
                        require_causal_inputs=True, expected_ego_input_sha256=digest,
                        expected_source_checkpoint_sha256=checkpoint_digest)
                if 'transforms' in transform:
                    patch(transform['transforms'])
        patch(ds.pipeline)
    return cfg


def main():
    import numpy as np
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.datasets.causal_ego import (load_artifact, file_sha256,
        build_record, feature_vector, observation_available_timestamp_us)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--ego-input', default='data/ego_causal_v1/ego_inputs.pkl')
    parser.add_argument('--baseline-checkpoint', default='ckpts/epoch_56.pth')
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if not 2 <= args.samples <= 256:
        parser.error('Only 2..256 diagnostic samples, not full validation')
    root = Path(args.output_dir).resolve()
    if root.exists():
        parser.error('Refusing to overwrite existing output directory')
    artifact, digest = load_artifact(args.ego_input)
    legacy = Config.fromfile(args.config)
    legacy.model.query_memory_cfg.future_memory_adapter_v2.query_chunk_size = 48
    legacy.data.test.test_mode = True
    legacy.resume_from = None
    legacy.load_from = args.checkpoint
    legacy.causal_experiment_stage = 'R1_ood_sensitivity_not_clean_training'
    causal = causal_dataset_options(copy.deepcopy(legacy), Path(args.ego_input).resolve(),
        digest, file_sha256(args.baseline_checkpoint), root / 'causal_cache')
    dataset = build_dataset(causal.data.test)
    # Uniform across scenes, take each selected scene midpoint. No label/class selection.
    by_scene = {}
    for i, raw in enumerate(dataset.temp2nusc_map):
        by_scene.setdefault(dataset.data_infos[raw]['scene_token'], []).append(i)
    scenes = sorted(by_scene)
    chosen_scenes = np.unique(np.linspace(0, len(scenes)-1, min(args.samples, len(scenes)), dtype=int))
    selected = [by_scene[scenes[j]][len(by_scene[scenes[j]])//2] for j in chosen_scenes]
    if len(selected) < args.samples:
        for i in np.linspace(0, len(dataset)-1, len(dataset), dtype=int):
            if int(i) not in selected: selected.append(int(i))
            if len(selected) == args.samples: break
    token_to_public = {dataset.data_infos[raw]['token']: i
                       for i, raw in enumerate(dataset.temp2nusc_map)}
    manifest, sources = [], set()
    packet_offsets_us = []
    for i in selected:
        raw = dataset.temp2nusc_map[i]
        info = dataset.data_infos[raw]
        manifest.append(dict(index=i, scene=info['scene_name'], token=info['token']))
        sources.update(token_to_public[h['sample_idx']]
                       for h in dataset._build_query_memory_history_infos(raw))
    # All chosen current/source ego features must be reproducible from past poses
    # even when future pose objects are inaccessible.
    for i in sorted(sources | set(selected)):
        raw = dataset.temp2nusc_map[i]
        info = dataset.data_infos[raw]
        window = dataset.data_infos[max(0, raw-4):raw+1]
        rebuilt = build_record(info, window)
        assert np.array_equal(feature_vector(rebuilt), feature_vector(artifact['records'][info['token']]))
        future = dict(token='future_probe', scene_token=info['scene_token'],
                      timestamp=info['timestamp']+1, ego2global_rotation=object())
        assert build_record(info, window+[future]) == rebuilt
        available = observation_available_timestamp_us(
            info, dataset.get_adj_info(info, raw))
        packet_offsets_us.append(available - int(info['timestamp']))
        for h in dataset._build_query_memory_history_infos(raw):
            if h['available_timestamp_us'] >= available:
                raise ValueError('History packet is unavailable at the current decision')
    for i in selected[:2]:
        data = dataset.get_data_info(dataset.temp2nusc_map[i])
        assert set(data['temporal_ego_states']) == {0}
        assert data['temporal_ego_states'][0].shape == (1, 21)
    root.mkdir(parents=True, exist_ok=False)
    legacy.dump(str(root / 'legacy.py'))
    causal.dump(str(root / 'causal.py'))
    (root / 'manifest.json').write_text(json.dumps(dict(manifest=manifest), indent=2))
    (root / 'cache_indices.json').write_text(json.dumps(sorted(sources)))
    result = dict(stage=causal.causal_experiment_stage, samples=len(selected),
        scenes=len({m['scene'] for m in manifest}), cache_sources=len(sources),
        ego_input_sha256=digest, checkpoint=args.checkpoint,
        checkpoint_sha256=file_sha256(args.checkpoint), baseline_checkpoint=args.baseline_checkpoint,
        baseline_checkpoint_sha256=file_sha256(args.baseline_checkpoint),
        input_reconstruction_and_future_corruption_check=True,
        packet_availability=dict(clock='last consumed camera/LiDAR timestamp',
                                 max_offset_from_lidar_us=max(packet_offsets_us)),
        original_config=args.config,
        caveat='Uses old trained weights: input sensitivity, NOT clean retraining results.')
    (root / 'preparation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
