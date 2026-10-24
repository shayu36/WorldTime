"""CPU-only temporal audit; no dataset iteration, inference or cache writes.

Walks all public train/test metadata, but samples cache contents to bound I/O.
This cannot certify how externally supplied ego-state/cache files were created.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets.pipelines.loading_query_memory import LoadQueryMemoryFromFiles
from mmdet3d.models.sparsedetectors.opus_head import OPUSHead, get_matched_inds
from mmdet3d.models.sparsedetectors.query_memory import FutureMemoryAdapterV2


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(cfg, checkpoint, cache_samples):
    report = {'scope': 'all sample metadata; sampled cache contents; CPU only'}
    split_sets = {}
    for split in ('train', 'test'):
        dataset = build_dataset(cfg.data[split])
        loader_cfg = next(p for p in cfg.data[split].pipeline
                          if p['type'] == 'LoadQueryMemoryFromFiles').copy()
        loader_cfg.pop('type')
        loader = LoadQueryMemoryFromFiles(**loader_cfg)
        scenes, tokens, ages, future_offsets = set(), set(), [], []
        errors, counts = [], Counter()
        history_reads = []
        for index in dataset.temp2nusc_map:
            info = dataset.data_infos[index]
            scenes.add(info['scene_token']); tokens.add(info['token'])
            ts = dataset._query_memory_timestamp(info)
            adjacent = dataset.get_adj_info(info, index)
            for prev in adjacent:
                counts['image_records'] += 1
                if prev['scene_token'] != info['scene_token'] or prev['timestamp'] > info['timestamp']:
                    errors.append(('image_time', info['token']))
                # Same-camera timestamp, rather than unsynchronised lidar time.
                for camera, frame in prev['cams'].items():
                    if frame['timestamp'] > info['cams'][camera]['timestamp']:
                        errors.append(('camera_future', info['token'], camera))
            histories = dataset._build_query_memory_history_infos(index)
            if len({h['sample_idx'] for h in histories}) != len(histories):
                errors.append(('duplicate_history', info['token']))
            for hist in histories:
                age = ts - hist['timestamp']; ages.append(age)
                counts['history_reads'] += 1
                counts['slot_' + str(hist['slot_index'])] += 1
                if not (hist['scene_id'] == str(dataset._query_memory_scene_id(info))
                        and hist['frame_idx'] < info['frame_idx'] and age > 0
                        and abs(age - hist['target_age']) <= dataset.query_memory_history_age_tolerance):
                    errors.append(('history_time', info['token']))
                history_reads.append((hist, ts, info['frame_idx']))
            for step in range(1, 7):
                future = dataset.data_infos[index + step]
                offset = (future['timestamp'] - info['timestamp']) / 1e6
                future_offsets.append(offset / step)
                if future['scene_token'] != info['scene_token'] or offset <= 0:
                    errors.append(('gt_horizon', info['token'], step))
        cache_sources, read_paths = Counter(), set()
        for j in np.linspace(0, len(history_reads)-1, min(cache_samples, len(history_reads)), dtype=int):
            hist, ts, frame = history_reads[j]
            path = loader._find_cache_path(loader.cache_root, hist)
            cache = torch.load(path, map_location='cpu')
            loader._validate_cache(cache, path, hist=hist, current_ts=ts, current_frame=frame)
            assert cache['schema_version'] == 2
            cache_sources[(cache.get('source_checkpoint'), cache.get('source_config'))] += 1
            read_paths.add(str(path))
            del cache
        split_sets[split] = (scenes, tokens)
        report[split] = dict(samples=len(dataset), scenes=len(scenes), counts=dict(counts),
                             history_age_range=[min(ages), max(ages)],
                             gt_half_second_step_range=[min(future_offsets), max(future_offsets)],
                             sampled_cache_files=len(read_paths), strict=loader.strict,
                             cache_sources=[dict(checkpoint=k[0], config=k[1], reads=v)
                                            for k, v in cache_sources.items()],
                             errors=errors[:30], error_count=len(errors),
                             ego_input_fields_excluding_gt=sorted(set(next(iter(dataset.ad_info.values()))) - {'gt'}))
        print(json.dumps({split: report[split]}, indent=2), flush=True)
        del dataset, history_reads
        gc.collect()
    report['split_overlap'] = dict(scenes=len(split_sets['train'][0] & split_sets['test'][0]),
                                   samples=len(split_sets['train'][1] & split_sets['test'][1]))
    state = torch.load(checkpoint, map_location='cpu')['state_dict']
    source = torch.load(cfg.resume_from, map_location='cpu')['state_dict']
    report['checkpoint'] = dict(sha256=sha256(checkpoint),
        non_adapter_changed=[k for k in source if not k.startswith('future_memory_adapter_v2.')
                             and (k not in state or not torch.equal(source[k], state[k]))],
        nonfinite=[k for k,v in state.items() if v.is_floating_point() and not torch.isfinite(v).all()])
    stamps = state['pts_bbox_head.num_stamps_all'].float()
    head_cfg = cfg.model.pts_bbox_head
    ids = get_matched_inds(stamps / stamps.sum(-1, keepdim=True),
                           [head_cfg.num_query] + list(head_cfg.num_fu_query))
    layers = [SimpleNamespace(self_attn=SimpleNamespace()) for _ in range(6)]
    head = SimpleNamespace(init_points=SimpleNamespace(weight=torch.zeros(len(ids), 3)),
                           ind_stamps_all=ids, num_fu_frames=6,
                           transformer=SimpleNamespace(decoder=SimpleNamespace(decoder_layers=layers)))
    OPUSHead.reset_mask(head)
    expected = ids[None, :] > ids[:, None]
    report['rap_mask'] = dict(shape=list(expected.shape), group_counts=torch.bincount(ids.long()).tolist(),
        blocked_future_edges=int(expected.sum()),
        all_layers_correct=all(torch.equal(l.self_attn.ind_mask < 0, expected) for l in layers),
        blocked_softmax_mass=float(torch.softmax(layers[0].self_attn.ind_mask, -1)[expected].sum()))
    # Deliberately malformed direct-module input: distinguish loader defence
    # from the adapter's own validation; do not silently assert it is protected.
    adapter = FutureMemoryAdapterV2(embed_dims=8, num_classes=3, num_points=2,
        num_heads=2, coarse_topk_geometry=1, coarse_topk_semantic=1, point_topk=1).eval()
    memory = dict(memory_query_feat=torch.ones(1,1,1,8), memory_points_metric=torch.zeros(1,1,1,2,3),
                  memory_valid=torch.ones(1,1,1,dtype=torch.bool), memory_age=torch.full((1,1,1), -.5),
                  memory_reliability=torch.ones(1,1,1), memory_semantic_distribution=torch.ones(1,1,1,3)/3)
    with torch.no_grad():
        out = adapter(torch.ones(1,1,8), torch.zeros(1,1,2,3), torch.zeros(1,1,2,3), memory, 0)
    report['direct_invalid_memory_probe'] = dict(base_age=-.5, horizon=1.,
        has_candidate=bool(out['diagnostics']['has_candidate'].any()),
        note='Injected memory_valid=True bypasses loader; not a production input.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cache-samples', type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(audit(Config.fromfile(args.config), args.checkpoint, args.cache_samples), indent=2))
