#!/usr/bin/env python
"""Audit a formal STAC-QM cache against actual dataset history selection."""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from mmcv import Config

from mmdet3d.datasets import build_dataset
from mmdet3d.datasets.pipelines.loading_query_memory import (
    LoadQueryMemoryFromFiles)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument(
        '--split', required=True, choices=['train', 'val', 'test'])
    parser.add_argument('--cache-root', default=None)
    parser.add_argument('--expected-schema-version', type=int, default=2)
    parser.add_argument('--expected-source-checkpoint', default=None)
    parser.add_argument('--json-out', default=None)
    parser.add_argument('--max-error-examples', type=int, default=20)
    return parser.parse_args()


def _find_loader_cfg(pipeline):
    for transform in pipeline:
        transform = dict(transform)
        if transform.get('type') == 'LoadQueryMemoryFromFiles':
            return transform
        nested = transform.get('transforms')
        if nested:
            found = _find_loader_cfg(nested)
            if found is not None:
                return found
    return None


def _dataset_current(dataset, dataset_index):
    info_index = dataset.temp2nusc_map[dataset_index]
    info = dataset.data_infos[info_index]
    return info_index, dict(
        sample_idx=str(info['token']),
        scene_id=str(dataset._query_memory_scene_id(info)),
        scene_token=info.get('scene_token'),
        scene_name=info.get('scene_name'),
        frame_idx=dataset._query_memory_frame_idx(info, info_index),
        timestamp=dataset._query_memory_timestamp(info))


def _categorize_error(exc):
    message = str(exc).lower()
    if isinstance(exc, FileNotFoundError):
        return 'missing'
    if 'not causal' in message:
        return 'noncausal'
    if 'scene_id mismatch' in message:
        return 'scene_mismatch'
    if 'shape' in message or 'must be [' in message or ' m mismatch' in message:
        return 'shape_error'
    if 'timestamp mismatch' in message or 'frame_idx mismatch' in message:
        return 'temporal_mismatch'
    if 'sample_idx mismatch' in message:
        return 'sample_mismatch'
    return 'corrupt_or_schema'


def _load_cache(path, cache_by_path, load_errors):
    key = str(path.resolve())
    if key in cache_by_path:
        return cache_by_path[key]
    if key in load_errors:
        raise load_errors[key]
    try:
        cache = torch.load(path, map_location='cpu')
    except Exception as exc:
        load_errors[key] = exc
        raise
    cache_by_path[key] = cache
    return cache


def _append_error(report, category, path, exc, limit):
    report['failures'][category] += 1
    examples = report['error_examples']
    if len(examples) < limit:
        examples.append(dict(
            category=category, path=str(path), error=str(exc)))


def _median(values):
    if not values:
        return None
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset_cfg = cfg.data[args.split].copy()
    if args.split in ('val', 'test'):
        dataset_cfg.test_mode = True
    dataset = build_dataset(dataset_cfg)

    loader_cfg = _find_loader_cfg(dataset_cfg.pipeline)
    if loader_cfg is None:
        raise RuntimeError(
            f'{args.split} pipeline has no LoadQueryMemoryFromFiles')
    loader_cfg = dict(loader_cfg)
    loader_cfg.pop('type')
    if args.cache_root is not None:
        loader_cfg['cache_root'] = args.cache_root
    loader_cfg['strict'] = True
    loader = LoadQueryMemoryFromFiles(**loader_cfg)
    cache_root = Path(loader.cache_root)
    expected_checkpoint = args.expected_source_checkpoint
    if expected_checkpoint is None:
        expected_checkpoint = cfg.get('load_from', None)

    failure_names = [
        'missing', 'missing_current_cache', 'load_error',
        'corrupt_or_schema', 'schema_mismatch',
        'source_checkpoint_mismatch', 'noncausal', 'scene_mismatch',
        'sample_mismatch', 'temporal_mismatch', 'shape_error',
        'duplicate_history_frame'
    ]
    report = dict(
        config=str(args.config),
        split=args.split,
        cache_root=str(cache_root),
        dataset_samples=len(dataset),
        cache_files=0,
        expected_current_cache_files=len(dataset),
        current_cache_files_found=0,
        orphan_cache_files=0,
        schema_versions={},
        source_checkpoints={},
        source_configs={},
        target_age_slots={},
        reliability=dict(min=None, median=None, max=None, count=0),
        class_histogram={},
        spatial_cell_coverage=0,
        failures={name: 0 for name in failure_names},
        error_examples=[])

    cache_paths = sorted(cache_root.rglob('*.pt')) if cache_root.exists() else []
    report['cache_files'] = len(cache_paths)
    expected_current_samples = set()
    expected_current_records = []
    for dataset_index in range(len(dataset)):
        _, current = _dataset_current(dataset, dataset_index)
        expected_current_samples.add(current['sample_idx'])
        expected_current_records.append(current)

    cache_by_path = {}
    load_errors = {}
    schema_versions = Counter()
    source_checkpoints = Counter()
    source_configs = Counter()
    class_histogram = Counter()
    reliability_values = []
    spatial_cells = set()
    cached_samples = set()

    for path in cache_paths:
        try:
            cache = _load_cache(path, cache_by_path, load_errors)
        except Exception as exc:
            _append_error(
                report, 'load_error', path, exc, args.max_error_examples)
            continue
        try:
            loader._validate_cache(cache, path)
        except Exception as exc:
            _append_error(
                report, _categorize_error(exc), path, exc,
                args.max_error_examples)
            continue

        schema = int(cache['schema_version'])
        schema_versions[str(schema)] += 1
        source_checkpoint = str(cache.get('source_checkpoint', '<missing>'))
        source_config = str(cache.get('source_config', '<missing>'))
        source_checkpoints[source_checkpoint] += 1
        source_configs[source_config] += 1
        cached_samples.add(str(cache['sample_idx']))
        if schema != args.expected_schema_version:
            _append_error(
                report, 'schema_mismatch', path,
                ValueError(
                    f'schema_version={schema}, expected '
                    f'{args.expected_schema_version}'),
                args.max_error_examples)
        if expected_checkpoint is not None and os.path.normpath(
                source_checkpoint) != os.path.normpath(str(expected_checkpoint)):
            _append_error(
                report, 'source_checkpoint_mismatch', path,
                ValueError(
                    f'source_checkpoint={source_checkpoint!r}, expected '
                    f'{expected_checkpoint!r}'),
                args.max_error_examples)

        reliability, labels = loader._cache_reliability_labels(cache)
        valid = cache['valid_mask'].detach().cpu().bool()
        reliability_values.extend(
            reliability[valid].detach().cpu().float().tolist())
        class_histogram.update(
            int(label) for label in labels[valid].tolist())
        points = cache['query_points_metric'].detach().cpu().float()
        if valid.any():
            centers = points[valid].mean(dim=-2)
            cells = torch.floor(
                centers[:, :2] / loader.spatial_cell_size).long()
            spatial_cells.update(tuple(int(v) for v in row) for row in cells)

    report['orphan_cache_files'] = len(cached_samples - expected_current_samples)
    for current in expected_current_records:
        path = loader._find_cache_path(cache_root, current)
        if path.exists():
            report['current_cache_files_found'] += 1
        else:
            _append_error(
                report, 'missing_current_cache', path,
                FileNotFoundError(
                    'cache generation did not produce this split sample'),
                args.max_error_examples)

    target_selected = Counter()
    target_loaded = Counter()
    target_total = Counter()

    for dataset_index in range(len(dataset)):
        info_index, current = _dataset_current(dataset, dataset_index)
        histories = dataset._build_query_memory_history_infos(info_index)
        seen_samples = set()
        for slot in range(loader.num_slots):
            target_total[slot] += 1
        for hist in histories:
            slot = int(hist.get('slot_index', 0))
            target_selected[slot] += 1
            sample = str(hist.get('sample_idx'))
            if sample in seen_samples:
                _append_error(
                    report, 'duplicate_history_frame', sample,
                    ValueError('one history frame was assigned to multiple slots'),
                    args.max_error_examples)
                continue
            seen_samples.add(sample)
            path = loader._find_cache_path(cache_root, hist)
            if not path.exists():
                _append_error(
                    report, 'missing', path,
                    FileNotFoundError(
                        f'missing selected history cache for {sample}'),
                    args.max_error_examples)
                continue
            try:
                cache = _load_cache(path, cache_by_path, load_errors)
                loader._validate_cache(
                    cache, path, hist=hist,
                    current_ts=current['timestamp'],
                    current_frame=current['frame_idx'])
            except Exception as exc:
                _append_error(
                    report, _categorize_error(exc), path, exc,
                    args.max_error_examples)
                continue
            target_loaded[slot] += 1

    report['schema_versions'] = dict(sorted(schema_versions.items()))
    report['source_checkpoints'] = dict(source_checkpoints)
    report['source_configs'] = dict(source_configs)
    for slot in range(loader.num_slots):
        target = loader.history_target_ages[slot] \
            if loader.history_selection_mode == 'target_age' else None
        total = target_total[slot]
        report['target_age_slots'][str(slot)] = dict(
            target_age=target,
            selected=target_selected[slot],
            loaded=target_loaded[slot],
            total_samples=total,
            loaded_coverage=(target_loaded[slot] / total if total else 0.0))
    if reliability_values:
        report['reliability'] = dict(
            min=min(reliability_values),
            median=_median(reliability_values),
            max=max(reliability_values),
            count=len(reliability_values))
    report['class_histogram'] = {
        str(key): value for key, value in sorted(class_histogram.items())}
    report['spatial_cell_coverage'] = len(spatial_cells)
    report['failure_count'] = sum(report['failures'].values())

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + '\n')
    if report['failure_count']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
