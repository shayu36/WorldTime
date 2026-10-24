#!/usr/bin/env python
import argparse
import hashlib
import json
from pathlib import Path

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.models.sparsedetectors.query_memory import (
    decode_points_metric, logits_to_query_confidence,
    compute_query_reliability, select_diverse_memory_queries)


# Schema v2 (Problem 4/5): adds per-query semantic reliability + label so the
# loader / online bank can run reliability-aware attention and class-diverse
# selection. query_conf is retained for schema-v1 backward compatibility.
CACHE_SCHEMA_VERSION = 2

REQUIRED_CACHE_KEYS = [
    'schema_version', 'sample_idx', 'scene_id', 'frame_idx', 'timestamp',
    'ego2global', 'query_feat', 'query_points_metric', 'query_conf',
    'query_semantic_distribution', 'query_label', 'query_margin',
    'query_entropy', 'query_reliability', 'valid_mask', 'pc_range',
    'embed_dims', 'num_points', 'num_classes', 'source_config',
    'source_checkpoint'
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Precompute SparseWorld observation query memory cache.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-queries-per-frame', type=int, default=256)
    parser.add_argument('--write-threshold', type=float, default=0.35)
    parser.add_argument('--min-reliability', type=float, default=0.0)
    parser.add_argument('--spatial-cell-size', type=float, default=4.0)
    parser.add_argument('--max-per-spatial-cell', type=int, default=16)
    parser.add_argument('--max-per-class', type=int, default=64)
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--skip-existing', action='store_true',
                        help='skip samples whose cache file already exists '
                             '(resume)')
    parser.add_argument('--indices-json',
                        help='Optional JSON list of PUBLIC dataset indices; bounded causal audit only')
    return parser.parse_args()


def meta_value(meta, key, fallback=None):
    value = meta.get(key, fallback)
    if isinstance(value, (list, tuple)):
        value = value[0]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1)[0].item()
    return value


def scene_id_from_meta(meta):
    scene_id = meta.get('scene_token', None)
    if scene_id is None:
        scene_id = meta.get('scene_name', meta.get('scene_id', None))
    if scene_id is None:
        raise KeyError('scene_token or scene_name is required in img_metas')
    return str(scene_id)


def timestamp_from_meta(meta):
    timestamp = meta.get('timestamp', None)
    if timestamp is None:
        timestamp = meta.get('img_timestamp', None)
    if isinstance(timestamp, (list, tuple)):
        timestamp = timestamp[0]
    if isinstance(timestamp, torch.Tensor):
        timestamp = timestamp.detach().cpu().reshape(-1)[0].item()
    if timestamp is None:
        raise KeyError('timestamp or img_timestamp is required in img_metas')
    return float(timestamp)


def frame_idx_from_meta(meta):
    frame_idx = meta.get('frame_idx', None)
    if frame_idx is None and 'curr' in meta:
        frame_idx = meta['curr'].get('frame_idx', None)
    if frame_idx is None:
        raise KeyError('frame_idx is required in img_metas')
    return int(frame_idx)


def output_path(output_dir, scene_id, sample_idx):
    return Path(output_dir) / scene_id / f'{sample_idx}.pt'


def select_queries(query_points_metric, reliability, query_conf, labels,
                   max_queries_per_frame, write_threshold, min_reliability,
                   spatial_cell_size, max_per_spatial_cell, max_per_class):
    # gate low-confidence queries out first (parity with the write_threshold),
    # then apply the shared deterministic diversity selection.
    valid = query_conf >= write_threshold
    return select_diverse_memory_queries(
        query_points_metric,
        reliability,
        valid=valid,
        labels=labels,
        max_queries=max_queries_per_frame,
        min_reliability=min_reliability,
        spatial_cell_size=spatial_cell_size,
        max_per_spatial_cell=max_per_spatial_cell,
        max_per_class=max_per_class)


def build_cache_record(model, outs, img_meta, args):
    ind_stamps_all = model.pts_bbox_head.ind_stamps_all
    obs_mask = ind_stamps_all == 0
    query_feat = outs['query_feat'][:, obs_mask]
    query_points = outs['all_refine_pts'][-1][:, obs_mask]
    query_logits = outs['all_cls_scores'][-1][:, obs_mask]
    query_points_metric = decode_points_metric(
        query_points, model.pts_bbox_head.pc_range)
    query_conf = logits_to_query_confidence(query_logits)
    rel = compute_query_reliability(query_logits)
    query_semantic_distribution = rel['query_semantic_distribution']
    query_label = rel['query_label']
    query_margin = rel['query_margin']
    query_entropy = rel['query_entropy']
    query_reliability = rel['query_reliability']

    if query_feat.shape[0] != 1:
        raise RuntimeError('precompute_query_memory expects batch_size=1')
    indices = select_queries(
        query_points_metric[0].detach().float().cpu(),
        query_reliability[0].detach().float().cpu(),
        query_conf[0].detach().float().cpu(),
        query_label[0].detach().cpu(),
        args.max_queries_per_frame,
        args.write_threshold,
        args.min_reliability,
        args.spatial_cell_size,
        args.max_per_spatial_cell,
        args.max_per_class)
    sample_idx = str(meta_value(img_meta, 'sample_idx'))
    scene_id = scene_id_from_meta(img_meta)
    ego2global = img_meta.get('ego2global', None)
    if ego2global is None:
        raise KeyError('ego2global is required to precompute query memory')
    if isinstance(ego2global, torch.Tensor):
        ego2global = ego2global.detach().cpu().float()
    else:
        ego2global = torch.as_tensor(ego2global, dtype=torch.float32)
    cache = dict(
        schema_version=CACHE_SCHEMA_VERSION,
        sample_idx=sample_idx,
        scene_id=scene_id,
        frame_idx=frame_idx_from_meta(img_meta),
        timestamp=timestamp_from_meta(img_meta),
        ego2global=ego2global,
        query_feat=query_feat[0, indices].detach().cpu(),
        query_points_metric=query_points_metric[0, indices].detach().cpu(),
        query_conf=query_conf[0, indices].detach().cpu(),
        query_semantic_distribution=query_semantic_distribution[
            0, indices].detach().cpu(),
        query_label=query_label[0, indices].detach().cpu(),
        query_margin=query_margin[0, indices].detach().cpu(),
        query_entropy=query_entropy[0, indices].detach().cpu(),
        query_reliability=query_reliability[0, indices].detach().cpu(),
        valid_mask=torch.ones(indices.numel(), dtype=torch.bool),
        pc_range=model.pts_bbox_head.pc_range.detach().cpu().tolist(),
        embed_dims=int(query_feat.shape[-1]),
        num_points=int(query_points.shape[-2]),
        num_classes=int(query_logits.shape[-1]),
        source_config=str(args.config),
        source_checkpoint=str(args.checkpoint))
    if getattr(args, 'causal_dataset', None) is not None:
        dataset = args.causal_dataset
        record = dataset.ad_info[sample_idx]
        from mmdet3d.datasets.causal_ego import validate_record, observation_available_timestamp_us
        raw_index = args.causal_raw_indices[sample_idx]
        info = dataset.data_infos[raw_index]
        validate_record(record, info)
        # The packet becomes available after its last camera exposure. The
        # source LiDAR timestamp is only the frame identifier; cameras can be
        # a fraction of a millisecond later without using a future frame.
        source_us = record['timestamp_us']
        image_us = [int(round(float(t) * 1e6)) for t in img_meta['img_timestamp']]
        available_us = observation_available_timestamp_us(
            info, dataset.get_adj_info(info, raw_index))
        latest = max([record['provenance']['max_input_timestamp_us']] + image_us)
        if latest > available_us:
            raise ValueError('Causal cache input exceeds observation availability')
        cache['input_provenance'] = dict(
            protocol=record['provenance']['protocol'],
            sample_idx=sample_idx, scene_id=scene_id,
            source_timestamp_us=source_us,
            source_observation_available_timestamp_us=available_us,
            max_input_timestamp_us=latest,
            ego_input_sha256=dataset.causal_ego_sha256,
            source_checkpoint_sha256=args.source_checkpoint_sha256,
            generator_sha256=args.generator_sha256,
            ego_implementation_sha256=args.ego_implementation_sha256)
    validate_cache_record(cache)
    return cache


def validate_cache_record(cache):
    missing = [key for key in REQUIRED_CACHE_KEYS if key not in cache]
    if missing:
        raise KeyError(f'cache record missing keys: {missing}')
    M = cache['query_feat'].shape[0]
    if cache['query_feat'].dim() != 2:
        raise ValueError('query_feat must be [M, C]')
    if cache['query_points_metric'].shape[:1] != (M,):
        raise ValueError('query_points_metric must share M with query_feat')
    if cache['query_points_metric'].shape[-1] != 3:
        raise ValueError('query_points_metric must be [M, R, 3]')
    if cache['query_conf'].shape != (M,):
        raise ValueError('query_conf must be [M]')
    num_classes = int(cache['num_classes'])
    if cache['query_semantic_distribution'].shape != (M, num_classes):
        raise ValueError(
            'query_semantic_distribution must be [M, num_classes]')
    for key in ('query_margin', 'query_entropy', 'query_reliability'):
        if cache[key].shape != (M,):
            raise ValueError(f'{key} must be [M]')
        if not torch.isfinite(cache[key]).all():
            raise ValueError(f'{key} must contain only finite values')
    if cache['query_label'].shape != (M,):
        raise ValueError('query_label must be [M]')
    if not torch.isfinite(cache['query_semantic_distribution']).all():
        raise ValueError(
            'query_semantic_distribution must contain only finite values')
    if M > 0:
        distribution = cache['query_semantic_distribution'].float()
        if (distribution < 0).any() or (distribution > 1).any():
            raise ValueError('query_semantic_distribution must be in [0, 1]')
        if not torch.allclose(
                distribution.sum(dim=-1), torch.ones(M), atol=1e-5):
            raise ValueError('query_semantic_distribution rows must sum to 1')
        reliability = cache['query_reliability'].float()
        margin = cache['query_margin'].float()
        if (reliability < 0).any() or (reliability > 1).any():
            raise ValueError('query_reliability must be in [0, 1]')
        if (margin < 0).any() or (margin > 1).any():
            raise ValueError('query_margin must be in [0, 1]')
    if cache['valid_mask'].shape != (M,):
        raise ValueError('valid_mask must be [M]')
    if tuple(cache['ego2global'].shape) != (4, 4):
        raise ValueError('ego2global must be [4, 4]')


def unwrap_img_metas(data):
    img_metas = data['img_metas']
    if hasattr(img_metas, 'data'):
        img_metas = img_metas.data[0]
    return img_metas


def unwrap_data_value(value):
    if hasattr(value, 'data'):
        value = value.data[0]
    return value


def run_sparseworld_rap_prefix(model, img, img_metas, data):
    """Run the same RAP prefix used by SparseWorld forward_backbone."""
    temporal_ego_states = unwrap_data_value(data['temporal_ego_states'])
    ego_states = temporal_ego_states[0].cuda(non_blocking=True)
    bs, _, dim_ = ego_states.shape
    ego_states = ego_states.view((bs, 1, dim_))
    ego_feat = model.plan_head(ego_states)
    points_scale = model.points_scale_branch(ego_feat)
    points_scale = torch.tanh(points_scale)
    model.pts_bbox_head.points_scale = (
        (points_scale + 1) / 2 * (1.5 - 0.8) + 0.8)
    img_feats = model.extract_feat(img, img_metas)
    return model.pts_bbox_head(img_feats, img_metas)


def setup_causal_generation(args, cfg, dataset, indices):
    """Pin one source protocol and permit resume only within that same run."""
    from mmdet3d.datasets.causal_ego import file_sha256, PROTOCOL
    if args.overwrite:
        raise ValueError('Causal cache forbids overwrite; use a new versioned directory')
    args.causal_dataset = dataset
    args.causal_raw_indices = {r['token']: i for i, r in enumerate(dataset.data_infos)}
    args.source_checkpoint_sha256 = file_sha256(args.checkpoint)
    args.generator_sha256 = file_sha256(__file__)
    repo = Path(__file__).resolve().parents[2]
    args.ego_implementation_sha256 = file_sha256(repo / 'mmdet3d/datasets/causal_ego.py')
    source_paths = ['mmdet3d/datasets/causal_ego.py',
                    'mmdet3d/datasets/nuscenes_dataset_occ_trajectory.py',
                    'mmdet3d/models/sparsedetectors/opus_head.py',
                    'mmdet3d/models/sparsedetectors/query_memory.py',
                    'mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py']
    manifest = dict(protocol=PROTOCOL, split=args.split,
        ego_input_sha256=dataset.causal_ego_sha256,
        source_checkpoint_sha256=args.source_checkpoint_sha256,
        generator_sha256=args.generator_sha256,
        source_sha256={p: file_sha256(repo / p) for p in source_paths},
        config_sha256=hashlib.sha256(cfg.pretty_text.encode()).hexdigest(),
        annotation_sha256=file_sha256(cfg.data[args.split].ann_file),
        dataset_count=len(dataset), selected_indices=indices,
        selection={k: getattr(args, k) for k in (
            'max_queries_per_frame', 'write_threshold', 'min_reliability',
            'spatial_cell_size', 'max_per_spatial_cell', 'max_per_class')})
    target = Path(args.output_dir)
    manifest_path = target / 'generation.json'
    if target.exists():
        if not args.skip_existing or not manifest_path.is_file():
            raise FileExistsError('Existing causal cache needs --skip-existing and a matching generation manifest')
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Causal cache generation/source/config changed; use a new output directory')
    else:
        target.mkdir(parents=True, exist_ok=False)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def validate_resumed_causal_record(cache, path, meta, args):
    """Never silently skip old, future-dependent, wrong-source or damaged cache."""
    from mmdet3d.datasets.causal_ego import observation_available_timestamp_us
    from mmdet3d.datasets.pipelines.loading_query_memory import LoadQueryMemoryFromFiles
    info = args.causal_dataset.data_infos[args.causal_raw_indices[str(meta_value(meta, 'sample_idx'))]]
    raw = args.causal_raw_indices[info['token']]
    available = observation_available_timestamp_us(info, args.causal_dataset.get_adj_info(info, raw))
    validator = LoadQueryMemoryFromFiles(require_causal_inputs=True,
        expected_ego_input_sha256=args.causal_dataset.causal_ego_sha256,
        expected_source_checkpoint_sha256=args.source_checkpoint_sha256)
    # No target frame exists during generation: +1 is only this internal
    # validator's strict-earlier sentinel, not a change to any stored time.
    validator._validate_cache(cache, path, hist=dict(sample_idx=info['token'],
        scene_id=str(info['scene_token']), timestamp=int(info['timestamp'])/1e6,
        available_timestamp_us=available), current_available_us=available+1)
    prov = cache['input_provenance']
    if (prov['generator_sha256'] != args.generator_sha256 or
            prov['ego_implementation_sha256'] != args.ego_implementation_sha256):
        raise ValueError('Resumed cache producer implementation mismatch')
    validate_cache_record(cache)


def main():
    args = parse_args()
    from mmdet.apis import set_random_seed
    set_random_seed(0, deterministic=True)
    cfg = Config.fromfile(args.config)
    if 'query_memory_cfg' in cfg.model:
        cfg.model.query_memory_cfg = dict(enabled=False)
    # Strip LoadQueryMemoryFromFiles from pipeline (we are generating cache, not loading it)
    # Also remove memory-related keys from Collect4D (including nested inside
    # MultiScaleFlipAug3D transforms, which is how the val/test pipeline is built)
    MEMORY_KEYS = {
        'memory_query_feat', 'memory_points_metric', 'memory_conf',
        'memory_reliability', 'memory_label', 'memory_valid',
        'memory_source_ego2global', 'memory_age', 'memory_semantic_distribution'
    }

    def _sanitize_transform(p):
        p = dict(p)  # shallow copy
        if p.get('type') == 'Collect4D':
            p['keys'] = [k for k in p.get('keys', []) if k not in MEMORY_KEYS]
            meta_keys = p.get('meta_keys', [])
            if isinstance(meta_keys, (list, tuple)):
                meta_keys = [k for k in meta_keys if k not in MEMORY_KEYS]
                p['meta_keys'] = type(meta_keys)(meta_keys)  # preserve tuple type
        if p.get('type') == 'MultiScaleFlipAug3D' and 'transforms' in p:
            p['transforms'] = [_sanitize_transform(t) for t in p['transforms']]
        return p

    for pipe_key in ('pipeline', 'test_pipeline'):
        if pipe_key not in cfg.data[args.split]:
            continue
        new_pipe = []
        for p in cfg.data[args.split][pipe_key]:
            if p.get('type') == 'LoadQueryMemoryFromFiles':
                continue
            # Flatten MultiScaleFlipAug3D into its inner transforms: the
            # precompute forward expects a plain dict result, not a
            # dict-of-lists produced by the test-time augmentation wrapper.
            if p.get('type') == 'MultiScaleFlipAug3D':
                for inner in p.get('transforms', []):
                    new_pipe.append(_sanitize_transform(inner))
                continue
            new_pipe.append(_sanitize_transform(p))
        cfg.data[args.split][pipe_key] = new_pipe
    dataset = build_dataset(cfg.data[args.split])
    selected_dataset = dataset
    indices = list(range(len(dataset)))
    if args.indices_json:
        indices = json.loads(Path(args.indices_json).read_text())
        if (not isinstance(indices, list) or not indices
                or any(type(i) is not int or not 0 <= i < len(dataset) for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('indices-json must contain unique valid integer dataset indices')
        selected_dataset = torch.utils.data.Subset(dataset, indices)
    causal_manifest = None
    if getattr(dataset, 'ego_input_mode', 'legacy') == 'causal_v1':
        causal_manifest = setup_causal_generation(args, cfg, dataset, indices)
    data_loader = build_dataloader(
        selected_dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval().cuda()
    model.requires_grad_(False)

    output_dir = Path(args.output_dir)
    skipped = 0
    completed = []
    with torch.no_grad():
        for data in data_loader:
            img_metas = unwrap_img_metas(data)
            if len(img_metas) != 1:
                raise RuntimeError('precompute_query_memory requires batch_size=1')
            if args.skip_existing:
                meta = img_metas[0]
                pre_path = output_path(
                    output_dir,
                    scene_id_from_meta(meta),
                    str(meta_value(meta, 'sample_idx')))
                if pre_path.exists():
                    if causal_manifest is not None:
                        cache = torch.load(pre_path, map_location='cpu')
                        validate_resumed_causal_record(cache, pre_path, meta, args)
                        completed.append(str(pre_path.relative_to(output_dir)))
                    skipped += 1
                    if skipped % 1000 == 0:
                        print(f'[skip-existing] skipped {skipped} so far', flush=True)
                    continue
            img = data['img']
            if hasattr(img, 'data'):
                img = img.data[0]
            img = img.cuda(non_blocking=True)
            outs = run_sparseworld_rap_prefix(model, img, img_metas, data)
            cache = build_cache_record(model, outs, img_metas[0], args)
            path = output_path(output_dir, cache['scene_id'], cache['sample_idx'])
            if path.exists() and not args.overwrite:
                raise FileExistsError(
                    f'{path} already exists; pass --overwrite to replace it')
            path.parent.mkdir(parents=True, exist_ok=True)
            if causal_manifest is not None:
                # A killed write must not leave a .pt file accepted on resume.
                temporary = path.with_suffix('.pt.partial')
                torch.save(cache, temporary)
                temporary.replace(path)
                completed.append(str(path.relative_to(output_dir)))
                if len(completed) % 100 == 0:
                    print(f'[causal-cache] {len(completed)}/{len(indices)}', flush=True)
            else:
                torch.save(cache, path)
    if skipped:
        print(f'[skip-existing] total skipped: {skipped}', flush=True)
    if causal_manifest is not None:
        from mmdet3d.datasets.causal_ego import file_sha256
        assert len(completed) == len(set(completed)) == len(indices)
        completion = dict(generation_sha256=file_sha256(output_dir / 'generation.json'),
                          dataset_count=len(dataset), count=len(completed),
                          full_dataset=indices == list(range(len(dataset))), files=completed)
        temporary = output_dir / 'complete.json.partial'
        temporary.write_text(json.dumps(completion, indent=2))
        temporary.replace(output_dir / 'complete.json')
        print('[causal-cache] COMPLETE', len(completed), flush=True)


if __name__ == '__main__':
    main()
