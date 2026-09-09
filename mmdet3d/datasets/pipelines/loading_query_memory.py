import os
from pathlib import Path

import torch

try:
    from ..builder import PIPELINES
except Exception:
    class _FallbackRegistry:
        def register_module(self):
            def _decorator(cls):
                return cls
            return _decorator
    PIPELINES = _FallbackRegistry()

# Share the EXACT deterministic diversity-selection and reliability functions
# with the model / online bank. Fall back to a standalone import so the loader
# can be exercised in a torch-only environment without mmdet3d installed.
try:
    from mmdet3d.models.sparsedetectors.query_memory import (
        select_diverse_memory_queries, compute_query_reliability)
except Exception:
    import importlib.util as _ilu
    _qm_path = (Path(__file__).resolve().parents[2] /
                'models' / 'sparsedetectors' / 'query_memory.py')
    _spec = _ilu.spec_from_file_location('_qm_for_loader', _qm_path)
    _qm = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_qm)
    select_diverse_memory_queries = _qm.select_diverse_memory_queries
    compute_query_reliability = _qm.compute_query_reliability


def _first_scalar(value, default=None):
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return _first_scalar(value[0], default)
    return float(value)


def _scene_id(results):
    return results.get(
        'scene_token', results.get('scene_id', results.get('scene_name', None)))


def _sample_idx(results):
    return results.get(
        'sample_idx', results.get('sample_token', results.get('token', None)))


def _frame_idx(results):
    frame_idx = results.get('frame_idx', None)
    if frame_idx is None and 'curr' in results:
        frame_idx = results['curr'].get('frame_idx', None)
    return None if frame_idx is None else int(frame_idx)


def _timestamp(results):
    timestamp = results.get('timestamp', None)
    if timestamp is None:
        timestamp = results.get('img_timestamp', None)
    if timestamp is None and 'curr' in results:
        curr = results['curr']
        if 'timestamp' in curr:
            timestamp = float(curr['timestamp']) / 1e6
    return _first_scalar(timestamp, None)


@PIPELINES.register_module()
class LoadQueryMemoryFromFiles(object):
    """Load causal same-scene observation query cache for STAC-QM."""

    def __init__(self,
                 cache_root=None,
                 history_frames=3,
                 max_queries_per_frame=256,
                 strict=False,
                 embed_dims=256,
                 num_points=48,
                 num_classes=17,
                 file_suffix='.pt',
                 history_selection_mode='recent',
                 history_target_ages=(2.5, 3.5, 4.5),
                 min_reliability=0.0,
                 spatial_cell_size=4.0,
                 max_per_spatial_cell=16,
                 max_per_class=64):
        self.cache_root = cache_root
        self.history_frames = int(history_frames)
        self.max_queries_per_frame = int(max_queries_per_frame)
        self.strict = bool(strict)
        self.embed_dims = int(embed_dims)
        self.num_points = int(num_points)
        self.num_classes = int(num_classes)
        self.file_suffix = file_suffix
        # Problem 6 (history selection) + Problem 5 (diversity selection).
        self.history_selection_mode = str(history_selection_mode)
        self.history_target_ages = [float(a) for a in history_target_ages]
        self.min_reliability = float(min_reliability)
        self.spatial_cell_size = float(spatial_cell_size)
        self.max_per_spatial_cell = int(max_per_spatial_cell)
        self.max_per_class = int(max_per_class)

    @property
    def num_slots(self):
        if self.history_selection_mode == 'target_age':
            return len(self.history_target_ages)
        return self.history_frames

    def _candidate_paths(self, root, hist):
        sample = str(hist.get('sample_idx', hist.get('sample_token')))
        scene_values = []
        for key in ('scene_id', 'scene_token', 'scene_name'):
            value = hist.get(key, None)
            if value is not None and str(value) not in scene_values:
                scene_values.append(str(value))
        paths = []
        for scene in scene_values:
            paths.append(Path(root) / scene / f'{sample}{self.file_suffix}')
        paths.append(Path(root) / f'{sample}{self.file_suffix}')
        return paths

    def _find_cache_path(self, root, hist):
        paths = self._candidate_paths(root, hist)
        for path in paths:
            if path.exists():
                return path
        return paths[0]

    def _hist_scene_id(self, hist):
        return hist.get('scene_id', hist.get('scene_token', hist.get('scene_name')))

    def _hist_sample_idx(self, hist):
        return hist.get('sample_idx', hist.get('sample_token', hist.get('token')))

    def _validate_cache(self, cache, path, hist=None, current_ts=None,
                        current_frame=None):
        required = [
            'schema_version', 'sample_idx', 'scene_id', 'frame_idx',
            'timestamp', 'ego2global', 'query_feat', 'query_points_metric',
            'query_conf', 'valid_mask'
        ]
        missing = [key for key in required if key not in cache]
        if missing:
            raise KeyError(f'{path} missing query memory keys: {missing}')
        if int(cache['schema_version']) not in (1, 2):
            raise ValueError(
                f'{path} has unsupported schema_version '
                f'{cache["schema_version"]}')
        query_feat = cache['query_feat']
        query_points = cache['query_points_metric']
        query_conf = cache['query_conf']
        valid_mask = cache['valid_mask']
        ego2global = cache['ego2global']
        if query_feat.dim() != 2:
            raise ValueError(f'{path} query_feat must be [M, C]')
        if query_points.dim() != 3 or query_points.shape[-1] != 3:
            raise ValueError(f'{path} query_points_metric must be [M, R, 3]')
        if query_conf.dim() != 1 or valid_mask.dim() != 1:
            raise ValueError(f'{path} query_conf and valid_mask must be [M]')
        if query_feat.shape[0] != query_points.shape[0]:
            raise ValueError(f'{path} query_feat/query_points M mismatch')
        if query_feat.shape[0] != query_conf.shape[0]:
            raise ValueError(f'{path} query_feat/query_conf M mismatch')
        if query_feat.shape[0] != valid_mask.shape[0]:
            raise ValueError(f'{path} query_feat/valid_mask M mismatch')
        if tuple(ego2global.shape) != (4, 4):
            raise ValueError(f'{path} ego2global must be [4, 4]')
        M = query_feat.shape[0]
        if int(cache['schema_version']) >= 2:
            v2_required = [
                'query_semantic_distribution', 'query_label', 'query_margin',
                'query_entropy', 'query_reliability', 'num_classes'
            ]
            v2_missing = [key for key in v2_required if key not in cache]
            if v2_missing:
                raise KeyError(
                    f'{path} schema-v2 cache missing keys: {v2_missing}')
            num_classes = int(cache['num_classes'])
            expected_shapes = dict(
                query_semantic_distribution=(M, num_classes),
                query_label=(M,),
                query_margin=(M,),
                query_entropy=(M,),
                query_reliability=(M,))
            for key, expected in expected_shapes.items():
                if tuple(cache[key].shape) != expected:
                    raise ValueError(
                        f'{path} {key} must have shape {expected}, got '
                        f'{tuple(cache[key].shape)}')
            for key in ('query_semantic_distribution', 'query_margin',
                        'query_entropy', 'query_reliability'):
                if not torch.isfinite(cache[key]).all():
                    raise ValueError(f'{path} {key} must be finite')
            reliability = cache['query_reliability'].float()
            margin = cache['query_margin'].float()
            if ((reliability < 0) | (reliability > 1)).any():
                raise ValueError(
                    f'{path} query_reliability must be in [0, 1]')
            if ((margin < 0) | (margin > 1)).any():
                raise ValueError(f'{path} query_margin must be in [0, 1]')
            distribution = cache['query_semantic_distribution'].float()
            if ((distribution < 0) | (distribution > 1)).any():
                raise ValueError(
                    f'{path} query_semantic_distribution must be in [0, 1]')
            if M > 0 and not torch.allclose(
                    distribution.sum(dim=-1), torch.ones(M), atol=1e-5):
                raise ValueError(
                    f'{path} query_semantic_distribution rows must sum to 1')
        if hist is None:
            return
        hist_scene = self._hist_scene_id(hist)
        hist_sample = self._hist_sample_idx(hist)
        hist_frame = hist.get('frame_idx', None)
        hist_ts = hist.get('timestamp', None)
        if hist_scene is not None and str(cache['scene_id']) != str(hist_scene):
            raise ValueError(
                f'{path} scene_id mismatch: cache={cache["scene_id"]}, '
                f'history={hist_scene}')
        if hist_sample is not None and str(cache['sample_idx']) != str(hist_sample):
            raise ValueError(
                f'{path} sample_idx mismatch: cache={cache["sample_idx"]}, '
                f'history={hist_sample}')
        if hist_frame is not None and int(cache['frame_idx']) != int(hist_frame):
            raise ValueError(
                f'{path} frame_idx mismatch: cache={cache["frame_idx"]}, '
                f'history={hist_frame}')
        if hist_ts is not None and float(cache['timestamp']) != float(hist_ts):
            raise ValueError(
                f'{path} timestamp mismatch: cache={cache["timestamp"]}, '
                f'history={hist_ts}')
        if current_frame is not None and int(cache['frame_idx']) >= int(current_frame):
            raise ValueError(f'{path} is not causal for current frame_idx')
        if current_ts is not None and float(cache['timestamp']) >= float(current_ts):
            raise ValueError(f'{path} is not causal for current timestamp')

    def _cache_reliability_labels(self, cache):
        """Return (reliability [M], label [M]) honoring schema version.

        Schema v2 stores these directly. Schema v1 has neither; reliability
        degrades to query_conf and label to -1 (unknown), which disables the
        class-diversity constraint downstream.
        """
        conf = cache['query_conf'].detach().cpu().float()
        M = conf.shape[0]
        if int(cache.get('schema_version', 1)) >= 2 and \
                'query_reliability' in cache:
            reliability = cache['query_reliability'].detach().cpu().float()
        else:
            reliability = conf.clone()
        if int(cache.get('schema_version', 1)) >= 2 and 'query_label' in cache:
            label = cache['query_label'].detach().cpu().long()
        else:
            label = torch.full((M,), -1, dtype=torch.long)
        return reliability, label

    def _select_queries(self, cache):
        feat = cache['query_feat'].detach().cpu()
        points = cache['query_points_metric'].detach().cpu().float()
        conf = cache['query_conf'].detach().cpu().float()
        valid = cache['valid_mask'].detach().cpu().bool()
        reliability, label = self._cache_reliability_labels(cache)
        if int(cache.get('schema_version', 1)) >= 2 and \
                'query_semantic_distribution' in cache:
            semantic = cache['query_semantic_distribution'].detach().cpu().float()
        else:
            # Schema-v1 has no semantic distribution.  Preserve a valid
            # uniform prior; the adapter still receives feature/geometry,
            # reliability and age, while labels remain -1 (unknown).
            semantic = torch.full(
                (feat.shape[0], self.num_classes),
                1.0 / float(self.num_classes), dtype=torch.float32)
        # deterministic reliability + class + spatial diversity, identical to
        # the online bank's selection rule.
        indices = select_diverse_memory_queries(
            points,
            reliability,
            valid=valid,
            labels=label,
            max_queries=self.max_queries_per_frame,
            min_reliability=self.min_reliability,
            spatial_cell_size=self.spatial_cell_size,
            max_per_spatial_cell=self.max_per_spatial_cell,
            max_per_class=self.max_per_class)
        sel_valid = torch.ones(indices.numel(), dtype=torch.bool)
        return (feat[indices], points[indices], conf[indices],
                reliability[indices], label[indices], semantic[indices],
                sel_valid)

    def _history_candidates(self, results):
        current_scene = _scene_id(results)
        current_sample = _sample_idx(results)
        current_frame = _frame_idx(results)
        current_ts = _timestamp(results)
        histories = results.get('query_memory_history_infos', [])
        filtered = []
        for hist in histories:
            hist_scene = hist.get(
                'scene_id', hist.get('scene_token', hist.get('scene_name')))
            if current_scene is not None and hist_scene != current_scene:
                continue
            hist_sample = hist.get('sample_idx', hist.get('sample_token'))
            if current_sample is not None and hist_sample == current_sample:
                continue
            hist_frame = hist.get('frame_idx', None)
            if current_frame is not None and hist_frame is not None:
                if int(hist_frame) >= current_frame:
                    continue
            hist_ts = hist.get('timestamp', None)
            if current_ts is not None and hist_ts is not None:
                if float(hist_ts) >= current_ts:
                    continue
            filtered.append(hist)
        if self.history_selection_mode == 'target_age':
            # dataset already assigned one info per target-age slot; keep all
            # causal candidates that carry a slot_index.
            slotted = [h for h in filtered if h.get('slot_index') is not None]
            return slotted if slotted else filtered[-self.num_slots:]
        return filtered[-self.history_frames:]

    def _empty_outputs(self, embed_dims=None, num_points=None,
                       num_classes=None):
        embed_dims = self.embed_dims if embed_dims is None else int(embed_dims)
        num_points = self.num_points if num_points is None else int(num_points)
        num_classes = self.num_classes if num_classes is None else int(num_classes)
        K = self.num_slots
        M = self.max_queries_per_frame
        return dict(
            memory_query_feat=torch.zeros(K, M, embed_dims),
            memory_points_metric=torch.zeros(K, M, num_points, 3),
            memory_conf=torch.zeros(K, M),
            memory_reliability=torch.zeros(K, M),
            memory_label=torch.full((K, M), -1, dtype=torch.long),
            memory_semantic_distribution=torch.zeros(K, M, num_classes),
            memory_valid=torch.zeros(K, M, dtype=torch.bool),
            memory_source_ego2global=torch.eye(4).repeat(K, 1, 1),
            memory_age=torch.zeros(K, M))

    def __call__(self, results):
        root = self.cache_root or results.get('query_memory_cache_root', None)
        histories = self._history_candidates(results)
        current_ts = _timestamp(results)
        loaded = []
        if root is None:
            if self.strict and histories:
                raise FileNotFoundError(
                    'query memory cache_root is required in strict mode')
        else:
            for hist in histories:
                path = self._find_cache_path(root, hist)
                if not path.exists():
                    if self.strict:
                        raise FileNotFoundError(
                            'missing query memory cache for sample '
                            f'{hist.get("sample_idx")}: {path}')
                    loaded.append((hist, path, None))
                    continue
                cache = torch.load(path, map_location='cpu')
                self._validate_cache(
                    cache, path, hist=hist, current_ts=current_ts,
                    current_frame=_frame_idx(results))
                loaded.append((hist, path, cache))

        first_cache = next((cache for _, _, cache in loaded if cache is not None), None)
        embed_dims = self.embed_dims
        num_points = self.num_points
        if first_cache is not None:
            embed_dims = int(first_cache['query_feat'].shape[-1])
            num_points = int(first_cache['query_points_metric'].shape[-2])
        memory = self._empty_outputs(embed_dims, num_points, self.num_classes)

        num_slots = self.num_slots
        if self.history_selection_mode == 'target_age':
            # place each cache into its dataset-assigned target-age slot.
            placements = []
            for hist, _, cache in loaded:
                slot = hist.get('slot_index')
                if slot is None or not (0 <= int(slot) < num_slots):
                    continue
                placements.append((int(slot), cache))
        else:
            # right-align recent frames (legacy behavior).
            offset = num_slots - len(loaded)
            placements = [
                (offset + k, cache) for k, (_, _, cache) in enumerate(loaded)]

        for out_index, cache in placements:
            if cache is None:
                continue
            feat, points, conf, reliability, label, semantic, valid = \
                self._select_queries(cache)
            n = min(feat.shape[0], self.max_queries_per_frame)
            # base_age = t_current - t_history (NEVER mutate cached timestamps).
            age = float(current_ts) - float(cache['timestamp'])
            memory['memory_query_feat'][out_index, :n] = feat[:n]
            memory['memory_points_metric'][out_index, :n] = points[:n]
            memory['memory_conf'][out_index, :n] = conf[:n]
            memory['memory_reliability'][out_index, :n] = reliability[:n]
            memory['memory_label'][out_index, :n] = label[:n]
            output_classes = memory['memory_semantic_distribution'].shape[-1]
            if semantic.shape[-1] > output_classes:
                raise ValueError(
                    'cache semantic class dimension exceeds the configured '
                    f'memory schema: {semantic.shape[-1]} > {output_classes}')
            # Legacy/synthetic schema-v2 fixtures can contain a smaller class
            # vocabulary.  Pad into the configured model vocabulary while
            # preserving the validated distribution; production NuScenes
            # caches use the full 17-class dimension.
            memory['memory_semantic_distribution'][
                out_index, :n, :semantic.shape[-1]] = semantic[:n]
            memory['memory_valid'][out_index, :n] = valid[:n] & (age > 0)
            memory['memory_source_ego2global'][out_index] = \
                cache['ego2global'].detach().cpu().float()
            memory['memory_age'][out_index, :n] = age
        results.update(memory)
        return results

    def __repr__(self):
        return (self.__class__.__name__ +
                f'(cache_root={self.cache_root}, '
                f'history_frames={self.history_frames}, '
                f'max_queries_per_frame={self.max_queries_per_frame}, '
                f'strict={self.strict})')
