import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 1e-6


def _as_float(value, default=None):
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return _as_float(value[0], default)
    return float(value)


def decode_points_metric(points, pc_range):
    """Decode normalized SparseWorld points to metric coordinates."""
    pc_range = torch.as_tensor(pc_range, device=points.device, dtype=points.dtype)
    out = points.clone()
    out[..., 0] = out[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    out[..., 1] = out[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    out[..., 2] = out[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    return out


def encode_points_normalized(points_metric, pc_range):
    """Encode metric points back to SparseWorld normalized coordinates."""
    pc_range = torch.as_tensor(
        pc_range, device=points_metric.device, dtype=points_metric.dtype)
    out = points_metric.clone()
    out[..., 0] = (out[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    out[..., 1] = (out[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
    out[..., 2] = (out[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])
    return out


def logits_to_query_confidence(logits):
    """Sigmoid-max-mean confidence for independent semantic logits.

    Args:
        logits (Tensor): [B, Q, R, C_sem]

    Returns:
        Tensor: [B, Q] float32 confidence.
    """
    if logits.dim() != 4:
        raise ValueError(
            'query logits must have shape [B, Q, R, C_sem], '
            f'got {tuple(logits.shape)}')
    return logits.float().sigmoid().amax(dim=-1).mean(dim=-1)


def compute_query_reliability(logits, eps=_EPS):
    """Point-level semantic reliability for STAC-QM memory queries (schema v2).

    Args:
        logits (Tensor): per-point class logits ``[..., R, C]`` where ``C`` is
            the number of semantic occupancy classes. NaN/Inf are sanitized
            before the softmax so the result is always finite.

    Returns:
        dict (leading dims equal to ``logits.shape[:-2]``):
            ``query_semantic_distribution`` ``[..., C]``
            ``query_label``    ``[...]`` (long, ``-1`` for empty queries)
            ``query_margin``   ``[...]``
            ``query_entropy``  ``[...]``
            ``query_reliability`` ``[...]`` in ``[0, 1]``

    Reliability is the mean of three terms, each mapped to ``[0, 1]``:

        ``top1``               = ``max_c mean_r softmax(logits)_c``
        ``margin``             = ``top1 - top2``
        ``normalized_entropy`` = ``1 - H(p) / log(C)``

    with ``p`` the point-averaged distribution. This is a *semantic
    reliability* over all ``C`` occupancy classes, NOT a foreground
    probability: all 17 classes are semantic occupancy categories.
    """
    if logits.dim() < 2:
        raise ValueError(
            'compute_query_reliability expects [..., R, C], got '
            f'{tuple(logits.shape)}')
    logits = logits.float()
    logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
    lead = logits.shape[:-2]
    num_points = logits.shape[-2]
    num_classes = logits.shape[-1]
    if num_points == 0 or num_classes == 0:
        c = max(int(num_classes), 1)
        return dict(
            query_semantic_distribution=logits.new_zeros(*lead, c),
            query_label=torch.full(
                lead, -1, dtype=torch.long, device=logits.device),
            query_margin=logits.new_zeros(*lead),
            query_entropy=logits.new_zeros(*lead),
            query_reliability=logits.new_zeros(*lead))
    point_probs = F.softmax(logits, dim=-1)
    dist = point_probs.mean(dim=-2)
    dist = dist / dist.sum(dim=-1, keepdim=True).clamp_min(eps)
    topk = torch.topk(dist, k=min(2, num_classes), dim=-1).values
    top1 = topk[..., 0]
    second = topk[..., 1] if num_classes >= 2 else torch.zeros_like(top1)
    margin = (top1 - second).clamp(0.0, 1.0)
    label = dist.argmax(dim=-1).long()
    entropy = -(dist * torch.log(dist.clamp_min(eps))).sum(dim=-1)
    norm_entropy = (1.0 - entropy / math.log(max(int(num_classes), 2))).clamp(
        0.0, 1.0)
    reliability = ((top1 + margin + norm_entropy) / 3.0).clamp(0.0, 1.0)
    reliability = torch.nan_to_num(reliability, nan=0.0, posinf=1.0, neginf=0.0)
    entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
    margin = torch.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)
    return dict(
        query_semantic_distribution=dist,
        query_label=label,
        query_margin=margin,
        query_entropy=entropy,
        query_reliability=reliability)


def compute_effective_age(base_age, future_offset=0.0):
    """Future-aware age: ``effective_age = base_age + future_offset`` (seconds).

    ``base_age = t_current - t_history`` is stored in the cache/online bank and
    NEVER mutated. ``future_offset`` is 0 for the observation query and
    ``(k+1) * frame_interval`` for scheduled future query group ``k``.
    """
    if not torch.is_tensor(base_age):
        base_age = torch.as_tensor(base_age, dtype=torch.float32)
    return base_age.float() + float(future_offset)


def select_diverse_memory_queries(points_metric,
                                  reliability,
                                  valid=None,
                                  labels=None,
                                  max_queries=256,
                                  min_reliability=0.0,
                                  spatial_cell_size=4.0,
                                  max_per_spatial_cell=16,
                                  max_per_class=64):
    """Deterministic reliability + class + spatial diversity selection.

    Shared by the cache precompute, the cache loader, and the online
    ``QueryMemoryBank`` so all three obey identical selection rules.

    Args:
        points_metric (Tensor): ``[M, R, 3]`` metric points per query.
        reliability   (Tensor): ``[M]`` reliability in ``[0, 1]``.
        valid         (Tensor): optional ``[M]`` bool validity.
        labels        (Tensor): optional ``[M]`` long semantic label. When
            ``None`` or all ``-1`` (schema v1) class constraints are disabled
            and selection degrades to spatial-diversity + reliability.

    Returns:
        LongTensor of selected original indices, deterministic order
        (reliability desc, ties broken by ascending index).
    """
    reliability = reliability.float()
    device = reliability.device
    M = reliability.shape[0]
    if M == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    keep = torch.isfinite(reliability) & (reliability >= float(min_reliability))
    if valid is not None:
        keep = keep & valid.to(device=device).bool()
    cand = torch.nonzero(keep, as_tuple=False).flatten()
    if cand.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    order = torch.argsort(reliability[cand], descending=True, stable=True)
    cand_sorted = cand[order].tolist()

    centers = points_metric.float().mean(dim=-2)  # [M, 3]
    cell_size = float(spatial_cell_size)
    use_labels = labels is not None
    labels_list = labels.long().tolist() if use_labels else None
    # if every label is -1 (unknown), disable class constraints entirely
    if use_labels and all(l < 0 for l in labels_list):
        use_labels = False

    max_queries = int(max_queries)
    max_per_cell = int(max_per_spatial_cell)
    max_per_class = int(max_per_class)
    cx = torch.floor(centers[:, 0] / cell_size).long().tolist()
    cy = torch.floor(centers[:, 1] / cell_size).long().tolist()

    selected = []
    selected_set = set()
    seen_cells = set()
    seen_classes = set()
    per_cell = {}
    per_class = {}

    def _cell(i):
        return (cx[i], cy[i])

    def _cls(i):
        return labels_list[i] if use_labels else -1

    # phase 1: prefer new class / new spatial cell, high reliability first
    for i in cand_sorted:
        if len(selected) >= max_queries:
            break
        cell = _cell(i)
        cls = _cls(i)
        if per_cell.get(cell, 0) >= max_per_cell:
            continue
        if use_labels and cls >= 0 and per_class.get(cls, 0) >= max_per_class:
            continue
        novel = cell not in seen_cells
        if use_labels and cls >= 0:
            novel = novel or (cls not in seen_classes)
        if not novel:
            continue
        selected.append(i)
        selected_set.add(i)
        per_cell[cell] = per_cell.get(cell, 0) + 1
        seen_cells.add(cell)
        if use_labels and cls >= 0:
            per_class[cls] = per_class.get(cls, 0) + 1
            seen_classes.add(cls)

    # phase 2: fill remaining capacity by reliability under caps
    for i in cand_sorted:
        if len(selected) >= max_queries:
            break
        if i in selected_set:
            continue
        cell = _cell(i)
        cls = _cls(i)
        if per_cell.get(cell, 0) >= max_per_cell:
            continue
        if use_labels and cls >= 0 and per_class.get(cls, 0) >= max_per_class:
            continue
        selected.append(i)
        selected_set.add(i)
        per_cell[cell] = per_cell.get(cell, 0) + 1
        if use_labels and cls >= 0:
            per_class[cls] = per_class.get(cls, 0) + 1

    return torch.tensor(selected, dtype=torch.long, device=device)


def safe_masked_softmax(scores, mask, dim=-1, eps=_EPS):
    """Softmax that returns strict zeros for fully masked rows."""
    if mask.dtype != torch.bool:
        mask = mask.bool()
    scores = scores.float()
    mask = mask.to(device=scores.device)
    has_candidate = mask.any(dim=dim, keepdim=True)
    neg_inf = torch.finfo(scores.dtype).min
    masked_scores = scores.masked_fill(~mask, neg_inf)
    row_max = masked_scores.max(dim=dim, keepdim=True).values
    row_max = torch.where(has_candidate, row_max, torch.zeros_like(row_max))
    exp_scores = torch.where(
        mask, torch.exp(masked_scores - row_max), torch.zeros_like(scores))
    denom = exp_scores.sum(dim=dim, keepdim=True)
    weights = exp_scores / denom.clamp_min(eps)
    weights = torch.where(has_candidate, weights, torch.zeros_like(weights))
    return weights


def _inverse_softplus(value):
    value = float(value)
    if value <= 0:
        return torch.tensor(-20.0, dtype=torch.float32)
    return torch.log(torch.expm1(torch.tensor(value, dtype=torch.float32)))



@dataclass
class ObservationMemoryEntry:
    query_feat: torch.Tensor
    query_points_metric: torch.Tensor
    query_conf: torch.Tensor
    valid_mask: torch.Tensor
    ego2global: torch.Tensor
    timestamp: float
    frame_idx: Optional[int]
    scene_id: Optional[str]
    sample_idx: Optional[str]
    query_reliability: Optional[torch.Tensor] = None
    query_label: Optional[torch.Tensor] = None


class QueryMemoryBank:
    """Online observation memory for strict sequential batch-size-one eval."""

    def __init__(self,
                 history_frames=3,
                 max_queries_per_frame=256,
                 write_threshold=0.35,
                 max_time_gap=None,
                 bank_size=None,
                 confidence_threshold=None,
                 history_selection_mode='recent',
                 history_target_ages=(2.5, 3.5, 4.5),
                 history_age_tolerance=0.35,
                 visual_history_window=2.0,
                 retention_seconds=5.0,
                 max_bank_entries=16,
                 min_reliability=0.0,
                 spatial_cell_size=4.0,
                 max_per_spatial_cell=16,
                 max_per_class=64):
        if bank_size is not None:
            history_frames = bank_size
        if confidence_threshold is not None:
            write_threshold = confidence_threshold
        self.history_frames = int(history_frames)
        self.max_queries_per_frame = int(max_queries_per_frame)
        self.write_threshold = float(write_threshold)
        self.max_time_gap = max_time_gap
        # Problem 5 (diversity) + Problem 6 (target-age history selection).
        self.history_selection_mode = str(history_selection_mode)
        self.history_target_ages = [float(a) for a in history_target_ages]
        self.history_age_tolerance = float(history_age_tolerance)
        self.visual_history_window = float(visual_history_window)
        self.retention_seconds = (
            None if retention_seconds is None else float(retention_seconds))
        self.max_bank_entries = (
            None if max_bank_entries is None else int(max_bank_entries))
        self.min_reliability = float(min_reliability)
        self.spatial_cell_size = float(spatial_cell_size)
        self.max_per_spatial_cell = int(max_per_spatial_cell)
        self.max_per_class = int(max_per_class)
        self.entries: List[ObservationMemoryEntry] = []
        self._last_scene_id = None
        self._last_sample_idx = None
        self._last_frame_idx = None
        self._last_timestamp = None

    def clear(self):
        self.entries.clear()
        self._last_scene_id = None
        self._last_sample_idx = None
        self._last_frame_idx = None
        self._last_timestamp = None

    def _maybe_reset_for_sequence(self, scene_id, sample_idx, frame_idx,
                                  timestamp):
        if self._last_scene_id is not None and scene_id != self._last_scene_id:
            self.clear()
            return 'scene_change'
        if sample_idx is not None and sample_idx == self._last_sample_idx:
            return 'duplicate_sample'
        if frame_idx is not None and self._last_frame_idx is not None:
            if frame_idx < self._last_frame_idx:
                self.clear()
                return 'frame_rollback'
            if frame_idx == self._last_frame_idx:
                return 'duplicate_frame'
        if timestamp is not None and self._last_timestamp is not None:
            if timestamp < self._last_timestamp:
                self.clear()
                return 'time_rollback'
            if timestamp == self._last_timestamp:
                return 'duplicate_timestamp'
            if self.max_time_gap is not None:
                if timestamp - self._last_timestamp > float(self.max_time_gap):
                    self.clear()
                    return 'time_gap_reset'
        return None

    def write(self,
              query_feat,
              query_points_metric,
              cls_scores=None,
              ego2global=None,
              timestamp=None,
              scene_id=None,
              sample_idx=None,
              frame_idx=None,
              query_conf=None,
              source_type='observation'):
        del source_type
        if query_feat.dim() != 3:
            raise ValueError('query_feat must have shape [B, Q, C]')
        if query_feat.shape[0] != 1:
            raise RuntimeError(
                'QueryMemoryBank online mode only supports batch_size=1; '
                f'got B={query_feat.shape[0]}')
        if query_points_metric.dim() != 4:
            raise ValueError(
                'query_points_metric must have shape [B, Q, R, 3]')
        if ego2global is None:
            raise ValueError('ego2global is required for query memory writes')
        if ego2global.dim() == 2:
            ego2global = ego2global.unsqueeze(0)
        if ego2global.shape[0] != 1 or ego2global.shape[-2:] != (4, 4):
            raise ValueError('ego2global must have shape [1, 4, 4] or [4, 4]')
        timestamp = _as_float(timestamp, 0.0)
        frame_idx = None if frame_idx is None else int(frame_idx)
        sample_idx = None if sample_idx is None else str(sample_idx)
        scene_id = None if scene_id is None else str(scene_id)

        reset_reason = self._maybe_reset_for_sequence(
            scene_id, sample_idx, frame_idx, timestamp)
        if reset_reason and reset_reason.startswith('duplicate'):
            return False

        if query_conf is None:
            if cls_scores is None:
                raise ValueError('cls_scores or query_conf is required')
            query_conf = logits_to_query_confidence(cls_scores)
        if query_conf.dim() == 3 and query_conf.shape[-1] == 1:
            query_conf = query_conf.squeeze(-1)
        if query_conf.shape[:2] != query_feat.shape[:2]:
            raise ValueError('query_conf must have shape [B, Q]')

        conf = query_conf[0].detach().float().cpu()
        points_cpu = query_points_metric[0].detach().float().cpu()

        # schema-v2 per-query semantic reliability + label (Problem 4). When no
        # class logits are available fall back to conf as reliability, label=-1.
        if cls_scores is not None:
            rel_dict = compute_query_reliability(cls_scores[0].detach().cpu())
            reliability = rel_dict['query_reliability'].float()
            labels = rel_dict['query_label'].long()
        else:
            reliability = conf.clone()
            labels = torch.full((conf.shape[0],), -1, dtype=torch.long)

        conf_ok = conf >= self.write_threshold
        # deterministic reliability + class + spatial diversity (Problem 5),
        # the SAME function the cache/loader use so online and offline agree.
        sel = select_diverse_memory_queries(
            points_cpu,
            reliability,
            valid=conf_ok,
            labels=labels,
            max_queries=self.max_queries_per_frame,
            min_reliability=self.min_reliability,
            spatial_cell_size=self.spatial_cell_size,
            max_per_spatial_cell=self.max_per_spatial_cell,
            max_per_class=self.max_per_class)
        valid_inds = sel

        entry = ObservationMemoryEntry(
            query_feat=query_feat[0, valid_inds].detach().cpu(),
            query_points_metric=points_cpu[valid_inds],
            query_conf=conf[valid_inds],
            valid_mask=torch.ones(valid_inds.numel(), dtype=torch.bool),
            ego2global=ego2global[0].detach().float().cpu(),
            timestamp=float(timestamp),
            frame_idx=frame_idx,
            scene_id=scene_id,
            sample_idx=sample_idx,
            query_reliability=reliability[valid_inds],
            query_label=labels[valid_inds])
        self.entries.append(entry)
        self._prune_entries(float(timestamp))

        self._last_scene_id = scene_id
        self._last_sample_idx = sample_idx
        self._last_frame_idx = frame_idx
        self._last_timestamp = float(timestamp)
        return True

    def _prune_entries(self, current_timestamp):
        """Bound the bank by retention window and entry count.

        ``recent`` mode keeps the historical ``history_frames`` cap (so the
        legacy right-aligned read is byte-for-byte unchanged). ``target_age``
        mode keeps a wider bank governed by ``retention_seconds`` and
        ``max_bank_entries`` so distant target ages remain reachable.
        """
        if self.history_selection_mode == 'recent':
            cap = self.history_frames
        else:
            if self.retention_seconds is not None:
                self.entries = [
                    e for e in self.entries
                    if current_timestamp - e.timestamp <= self.retention_seconds]
            cap = (self.max_bank_entries if self.max_bank_entries is not None
                   else self.history_frames)
        if cap is not None:
            while len(self.entries) > cap:
                self.entries.pop(0)

    def _causal_candidates(self, scene_id, sample_idx, frame_idx, timestamp):
        """Strictly-past, same-scene entries with a positive base age."""
        cands = []
        for entry in self.entries:
            if scene_id is not None and entry.scene_id != scene_id:
                continue
            if sample_idx is not None and entry.sample_idx == sample_idx:
                continue
            if frame_idx is not None and entry.frame_idx is not None:
                if entry.frame_idx >= frame_idx:
                    continue
            age = timestamp - entry.timestamp
            if age <= 0:
                continue
            cands.append((entry, age))
        return cands

    def _assign_target_age_slots(self, candidates):
        """Greedily map candidates to target-age slots (Problem 6).

        Slot ``j`` takes the strictly-past frame whose base age is closest to
        ``history_target_ages[j]`` within ``history_age_tolerance``. No frame is
        used by two slots; an unmatched slot stays invalid. Slot order follows
        ``history_target_ages`` exactly.
        """
        assigned = {}
        used = set()
        for j, target in enumerate(self.history_target_ages):
            best = None
            best_gap = None
            for idx, (entry, age) in enumerate(candidates):
                if idx in used:
                    continue
                gap = abs(age - target)
                if gap > self.history_age_tolerance:
                    continue
                if best is None or gap < best_gap:
                    best = idx
                    best_gap = gap
            if best is not None:
                used.add(best)
                assigned[j] = candidates[best]
        return assigned

    def read(self,
             scene_id=None,
             sample_idx=None,
             frame_idx=None,
             timestamp=None,
             device=None,
             dtype=torch.float32):
        if not self.entries:
            return None
        timestamp = _as_float(timestamp, 0.0)
        frame_idx = None if frame_idx is None else int(frame_idx)
        sample_idx = None if sample_idx is None else str(sample_idx)
        scene_id = None if scene_id is None else str(scene_id)

        candidates = self._causal_candidates(
            scene_id, sample_idx, frame_idx, timestamp)
        if not candidates:
            return None

        if self.history_selection_mode == 'target_age':
            num_slots = len(self.history_target_ages)
            assigned = self._assign_target_age_slots(candidates)
            if not assigned:
                return None
            slot_entries = assigned  # {slot_index: (entry, age)}
            template = next(iter(assigned.values()))[0]
        else:
            num_slots = self.history_frames
            recent = candidates[-self.history_frames:]
            offset = self.history_frames - len(recent)
            slot_entries = {
                offset + k: pair for k, pair in enumerate(recent)}
            template = recent[0][0]

        max_m = self.max_queries_per_frame
        embed_dims = template.query_feat.shape[-1]
        num_points = template.query_points_metric.shape[-2]
        device = torch.device('cpu') if device is None else device
        feat = torch.zeros(
            1, num_slots, max_m, embed_dims, device=device, dtype=dtype)
        points = torch.zeros(
            1, num_slots, max_m, num_points, 3, device=device, dtype=dtype)
        conf = torch.zeros(1, num_slots, max_m, device=device)
        reliability = torch.zeros(1, num_slots, max_m, device=device)
        label = torch.full(
            (1, num_slots, max_m), -1, device=device, dtype=torch.long)
        valid = torch.zeros(
            1, num_slots, max_m, device=device, dtype=torch.bool)
        source_ego = torch.eye(4, device=device).repeat(1, num_slots, 1, 1)
        age = torch.zeros(1, num_slots, max_m, device=device)

        for k, (entry, curr_age) in slot_entries.items():
            n = min(entry.query_feat.shape[0], max_m)
            if n == 0:
                continue
            feat[0, k, :n] = entry.query_feat[:n].to(device=device, dtype=dtype)
            points[0, k, :n] = entry.query_points_metric[:n].to(
                device=device, dtype=dtype)
            conf[0, k, :n] = entry.query_conf[:n].to(device=device)
            if entry.query_reliability is not None:
                reliability[0, k, :n] = entry.query_reliability[:n].to(
                    device=device)
            else:
                reliability[0, k, :n] = entry.query_conf[:n].to(device=device)
            if entry.query_label is not None:
                label[0, k, :n] = entry.query_label[:n].to(device=device)
            source_ego[0, k] = entry.ego2global.to(device=device)
            age[0, k, :n] = curr_age
            valid[0, k, :n] = entry.valid_mask[:n].to(device=device)
            valid[0, k, :n] &= curr_age > 0
        return dict(
            memory_query_feat=feat,
            memory_points_metric=points,
            memory_conf=conf,
            memory_reliability=reliability,
            memory_label=label,
            memory_valid=valid,
            memory_source_ego2global=source_ego,
            memory_age=age)

    def read_all(self, current_timestamp=0.0):
        return self.read(timestamp=current_timestamp)

    def __len__(self):
        return len(self.entries)


class EgoPoseAligner(nn.Module):
    """Align metric points from source ego frames to current ego frame."""

    def __init__(self, pc_range=None):
        super().__init__()
        if pc_range is None:
            pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.register_buffer(
            'pc_range', torch.as_tensor(pc_range, dtype=torch.float32))

    def forward(self, points_metric, source_ego2global, target_ego2global):
        orig_dtype = points_metric.dtype
        points = points_metric.float()
        source = source_ego2global.to(points.device).float()
        target = target_ego2global.to(points.device).float()

        if points.dim() == 4:
            if source.dim() == 2:
                source = source.unsqueeze(0)
            if target.dim() == 2:
                target = target.unsqueeze(0)
            transform = torch.linalg.inv(target) @ source
            rotation = transform[..., :3, :3]
            translation = transform[..., :3, 3]
            aligned = (
                torch.matmul(points, rotation.transpose(-1, -2)[:, None]) +
                translation[:, None, None, :])
        elif points.dim() == 5:
            if source.dim() == 3:
                source = source.unsqueeze(0)
            if target.dim() == 2:
                target = target.unsqueeze(0)
            transform = torch.linalg.inv(target)[:, None] @ source
            rotation = transform[..., :3, :3]
            translation = transform[..., :3, 3]
            aligned = (
                torch.matmul(
                    points, rotation.transpose(-1, -2)[:, :, None]) +
                translation[:, :, None, None, :])
        else:
            raise ValueError(
                'points_metric must have shape [B, M, R, 3] or '
                f'[B, K, M, R, 3], got {tuple(points_metric.shape)}')
        return aligned.to(orig_dtype)

    def align_normalized(self, points_normalized, source_ego2global,
                         target_ego2global):
        metric = decode_points_metric(points_normalized, self.pc_range)
        aligned = self.forward(metric, source_ego2global, target_ego2global)
        return encode_points_normalized(aligned, self.pc_range)


class QueryMotionCompensator(nn.Module):
    """Zero-initialized, ego-independent object-motion residual.

    Predicts a per-query velocity from the memory feature and its
    ``effective_age`` and applies a single translation shared by all ``R``
    points of a query::

        v_i     = v_max * tanh(MLP[LN(m_i), phi(dt_i)])
        P_hat_i = P_i(ego-aligned) + dt_i * v_i

    The last linear layer is zero-initialized, so at initialization the module
    is an exact no-op (``velocity == 0``) and the pipeline degrades to pure
    ego-only alignment. It never overwrites the raw cached positions; the
    compensated positions are only used for candidate distance/spatial gating.
    """

    def __init__(self, embed_dims=256, hidden_dims=None, max_velocity=20.0,
                 max_age=8.0):
        super().__init__()
        hidden_dims = embed_dims if hidden_dims is None else int(hidden_dims)
        self.embed_dims = int(embed_dims)
        self.max_velocity = float(max_velocity)
        self.max_age = float(max_age)
        self.time_dims = 2
        self.norm = nn.LayerNorm(self.embed_dims)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dims + self.time_dims, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, 3))
        # zero-init last layer -> exact no-op until trained
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _time_features(self, effective_age):
        age = effective_age.float()
        norm = age / (self.max_age + _EPS)
        return torch.stack([age, norm], dim=-1)

    def forward(self, memory_feat, effective_age):
        """Args: memory_feat ``[B, K, M, C]``, effective_age ``[B, K, M]``.

        Returns velocity ``[B, K, M, 3]`` (metres/second).
        """
        proj_dtype = self.mlp[0].weight.dtype
        feat = self.norm(memory_feat.to(proj_dtype))
        time_feat = self._time_features(effective_age).to(proj_dtype)
        mlp_in = torch.cat([feat, time_feat], dim=-1)
        velocity = self.max_velocity * torch.tanh(self.mlp(mlp_in))
        return velocity.float()


class CausalQueryMemoryAttention(nn.Module):
    """Spatio-temporal, reliability-aware causal multi-head memory read."""

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.0,
                 lambda_position=1.0,
                 lambda_time=1.0,
                 lambda_reliability=None,
                 lambda_confidence=1.0,
                 lambda_pos=None,
                 lambda_conf=None,
                 spatial_radius=12.0,
                 topk=32,
                 max_age=3.0,
                 pc_range=None):
        super().__init__()
        if lambda_pos is not None:
            lambda_position = lambda_pos
        if lambda_conf is not None:
            lambda_confidence = lambda_conf
        # legacy `lambda_confidence` maps to the reliability weight
        if lambda_reliability is None:
            lambda_reliability = lambda_confidence
        if embed_dims % num_heads != 0:
            raise ValueError(
                'STAC-QM requires embed_dims % num_heads == 0, got '
                f'embed_dims={embed_dims}, num_heads={num_heads}')
        if spatial_radius is None or float(spatial_radius) <= 0:
            raise ValueError('spatial_radius must be a positive float')
        if int(topk) <= 0:
            raise ValueError('topk must be a positive integer')
        if float(max_age) <= 0:
            raise ValueError('max_age must be a positive float')
        self.embed_dims = int(embed_dims)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dims // self.num_heads
        self.spatial_radius = float(spatial_radius)
        self.topk = int(topk)
        self.max_age = float(max_age)
        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(float(dropout))
        self.raw_lambda_position = nn.Parameter(
            _inverse_softplus(lambda_position).repeat(self.num_heads))
        self.raw_lambda_time = nn.Parameter(
            _inverse_softplus(lambda_time).repeat(self.num_heads))
        self.raw_lambda_reliability = nn.Parameter(
            _inverse_softplus(lambda_reliability).repeat(self.num_heads))
        if pc_range is None:
            pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.register_buffer(
            'pc_range', torch.as_tensor(pc_range, dtype=torch.float32))
        nn.init.zeros_(self.out_proj.bias)

    def _project_heads(self, layer, tensor, seq_len):
        proj_dtype = layer.weight.dtype
        projected = layer(tensor.to(proj_dtype)).float()
        return projected.view(
            tensor.shape[0], seq_len, self.num_heads,
            self.head_dim).permute(0, 2, 1, 3)

    def forward(self,
                query_feat,
                query_points_metric,
                memory_query_feat,
                memory_points_metric,
                memory_conf,
                memory_age,
                memory_valid,
                memory_reliability=None):
        """``memory_age`` is the future-aware EFFECTIVE age (base + offset).

        ``memory_reliability`` is the schema-v2 reliability; when ``None`` the
        legacy ``memory_conf`` is used as the reliability fallback.
        """
        B, Q, C = query_feat.shape
        if C != self.embed_dims:
            raise ValueError(
                f'query feature dim {C} does not match embed_dims '
                f'{self.embed_dims}')
        if memory_query_feat is None or memory_query_feat.numel() == 0:
            zeros = query_feat.new_zeros(B, Q, C)
            return zeros, self._empty_diagnostics(B, Q, query_feat.device)

        if memory_query_feat.dim() == 3:
            memory_query_feat = memory_query_feat.unsqueeze(0)
            memory_points_metric = memory_points_metric.unsqueeze(0)
            memory_conf = memory_conf.unsqueeze(0)
            memory_age = memory_age.unsqueeze(0)
            memory_valid = memory_valid.unsqueeze(0)
            if memory_reliability is not None and memory_reliability.dim() == 2:
                memory_reliability = memory_reliability.unsqueeze(0)
        if memory_query_feat.shape[0] != B:
            raise ValueError(
                'memory batch size must match query batch size, got '
                f'{memory_query_feat.shape[0]} and {B}')

        K, M = memory_query_feat.shape[1:3]
        N = K * M
        if N == 0:
            zeros = query_feat.new_zeros(B, Q, C)
            return zeros, self._empty_diagnostics(B, Q, query_feat.device)

        mem_feat = memory_query_feat.reshape(B, N, C).to(query_feat.device)
        mem_points = memory_points_metric.reshape(
            B, N, memory_points_metric.shape[-2], 3).to(query_feat.device)
        mem_conf = memory_conf.reshape(B, N).to(query_feat.device).float()
        mem_age = memory_age.reshape(B, N).to(query_feat.device).float()
        mem_valid = memory_valid.reshape(B, N).to(query_feat.device).bool()
        if memory_reliability is None:
            mem_reliability = mem_conf
        else:
            mem_reliability = memory_reliability.reshape(
                B, N).to(query_feat.device).float()

        q = self._project_heads(self.q_proj, query_feat, Q)
        k = self._project_heads(self.k_proj, mem_feat, N)
        v = self._project_heads(self.v_proj, mem_feat, N)

        semantic = torch.matmul(q, k.transpose(-1, -2))
        semantic = semantic / math.sqrt(float(self.head_dim))

        q_center = query_points_metric.to(query_feat.device).float().mean(dim=-2)
        m_center = mem_points.float().mean(dim=-2)
        dist_sq = ((q_center[:, :, None] - m_center[:, None])**2).sum(dim=-1)
        dist = torch.sqrt(dist_sq.clamp_min(0.0))

        # effective-age causal + max-age filter (mem_age is effective age)
        age_valid = (mem_age > 0.0) & (mem_age <= self.max_age)
        reliability_valid = torch.isfinite(mem_reliability) & (
            mem_reliability > 0.0)
        base_valid = mem_valid & age_valid & reliability_valid
        spatial_valid = dist <= self.spatial_radius
        candidate = base_valid[:, None, :] & spatial_valid

        lambda_position = F.softplus(self.raw_lambda_position).view(
            1, self.num_heads, 1, 1)
        lambda_time = F.softplus(self.raw_lambda_time).view(
            1, self.num_heads, 1, 1)
        lambda_reliability = F.softplus(self.raw_lambda_reliability).view(
            1, self.num_heads, 1, 1)
        radius_norm = self.spatial_radius**2 + _EPS
        age_norm = self.max_age + _EPS
        scores = semantic
        scores = scores - lambda_position * dist_sq[:, None] / radius_norm
        scores = scores - lambda_time * mem_age[:, None, None] / age_norm
        scores = scores + lambda_reliability * torch.log(
            mem_reliability.clamp_min(_EPS))[:, None, None]

        topk = min(self.topk, N)
        expanded_candidate = candidate[:, None].expand(
            B, self.num_heads, Q, N)
        topk_scores, topk_indices = torch.topk(
            scores.masked_fill(~expanded_candidate, torch.finfo(scores.dtype).min),
            k=topk,
            dim=-1)
        topk_valid = torch.gather(expanded_candidate, -1, topk_indices)
        weights = safe_masked_softmax(topk_scores, topk_valid, dim=-1)
        weights = self.dropout(weights)

        gather_index = topk_indices[..., None].expand(
            B, self.num_heads, Q, topk, self.head_dim)
        selected_v = torch.gather(
            v[:, :, None].expand(B, self.num_heads, Q, N, self.head_dim),
            3,
            gather_index)
        read_heads = (weights[..., None] * selected_v).sum(dim=-2)
        readout = read_heads.permute(0, 2, 1, 3).reshape(B, Q, C)
        output = self.out_proj(readout.to(self.out_proj.weight.dtype))
        output = output.to(query_feat.dtype)

        has_candidate_head = topk_valid.any(dim=-1)
        has_candidate = has_candidate_head.any(dim=1)
        output = output * has_candidate.unsqueeze(-1).to(output.dtype)

        selected_reliability = torch.gather(
            mem_reliability[:, None, None].expand(B, self.num_heads, Q, N),
            -1,
            topk_indices)
        selected_dist = torch.gather(
            dist[:, None].expand(B, self.num_heads, Q, N), -1,
            topk_indices)
        selected_age = torch.gather(
            mem_age[:, None, None].expand(B, self.num_heads, Q, N),
            -1,
            topk_indices)
        support_rel_head = (weights * selected_reliability).sum(dim=-1)
        avg_dist_head = (weights * selected_dist).sum(dim=-1)
        avg_age_head = (weights * selected_age).sum(dim=-1)
        head_den = has_candidate_head.float().sum(dim=1).clamp_min(1.0)
        support_reliability = (
            support_rel_head * has_candidate_head.float()).sum(dim=1) / head_den
        avg_dist = (
            avg_dist_head * has_candidate_head.float()).sum(dim=1) / head_den
        avg_age = (
            avg_age_head * has_candidate_head.float()).sum(dim=1) / head_den
        diagnostics = dict(
            has_candidate=has_candidate,
            # support_conf kept as an alias for the fusion gate / legacy logs
            support_conf=support_reliability,
            support_reliability=support_reliability,
            candidate_count=candidate.sum(dim=-1).detach(),
            topk_candidate_count=topk_valid.sum(dim=-1).max(dim=1).values.detach(),
            avg_distance=avg_dist.detach(),
            avg_age=avg_age.detach(),
            effective_age=avg_age.detach(),
            attention_shape=(B, self.num_heads, Q, topk))
        return output, diagnostics

    def _empty_diagnostics(self, B, Q, device):
        return dict(
            has_candidate=torch.zeros(B, Q, device=device, dtype=torch.bool),
            support_conf=torch.zeros(B, Q, device=device),
            support_reliability=torch.zeros(B, Q, device=device),
            candidate_count=torch.zeros(B, Q, device=device, dtype=torch.long),
            topk_candidate_count=torch.zeros(B, Q, device=device, dtype=torch.long),
            avg_distance=torch.zeros(B, Q, device=device),
            avg_age=torch.zeros(B, Q, device=device),
            effective_age=torch.zeros(B, Q, device=device),
            attention_shape=(B, self.num_heads, Q, 0))


class ConfidenceGatedFusion(nn.Module):
    """Confidence-gated residual query fusion with LayerScale.

    ``alpha_init=0`` is retained as an explicit strict-identity option for
    legacy STAC-QM modes.  Phase 3 uses a small nonzero alpha so the first
    backward pass reaches attention, gate, and projection parameters.
    """

    def __init__(self,
                 embed_dims=256,
                 ffn_dims=512,
                 gate_bias=-4.0,
                 alpha_init=0.0,
                 out_proj_gain=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dims)
        self.norm_h = nn.LayerNorm(embed_dims)
        self.norm_delta = nn.LayerNorm(embed_dims)
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dims * 3 + 2, ffn_dims),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dims, embed_dims),
        )
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        if float(out_proj_gain) == 0.0:
            nn.init.zeros_(self.out_proj.weight)
        else:
            nn.init.xavier_uniform_(self.out_proj.weight,
                                    gain=float(out_proj_gain))
        nn.init.zeros_(self.out_proj.bias)
        nn.init.constant_(self.gate_mlp[-1].bias, float(gate_bias))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    @property
    def memory_norm(self):
        """Named alias used by the Phase 3 fusion specification."""
        return self.norm_h

    def forward(self,
                query_feat,
                memory_output,
                current_confidence,
                support_confidence,
                has_candidate):
        orig_dtype = query_feat.dtype
        if current_confidence.dim() == 2:
            current_confidence = current_confidence.unsqueeze(-1)
        if support_confidence.dim() == 2:
            support_confidence = support_confidence.unsqueeze(-1)
        has = has_candidate.to(device=query_feat.device).bool()
        h = memory_output.to(query_feat.device)
        gate_input = torch.cat([
            self.norm_q(query_feat.float()),
            self.memory_norm(h.float()),
            self.norm_delta((query_feat - h).float()),
            current_confidence.to(query_feat.device).float(),
            support_confidence.to(query_feat.device).float(),
        ], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))
        residual = self.out_proj(h.to(self.out_proj.weight.dtype)).to(orig_dtype)
        applied_residual = (
            has.unsqueeze(-1).to(orig_dtype) * self.alpha.to(orig_dtype) *
            gate.to(orig_dtype) * residual)
        fused = query_feat + applied_residual
        has_float = has.float()
        candidate_count = has_float.sum().clamp_min(1.0)
        conditional_gate = (
            (gate.detach().float().mean(dim=-1) * has_float).sum() /
            candidate_count)
        residual_ratio = applied_residual.detach().float().norm(dim=-1) / (
            query_feat.detach().float().norm(dim=-1) + 1e-6)
        diagnostics = dict(
            avg_gate=(
                gate.detach() * has.unsqueeze(-1).float()).mean(dim=-1),
            conditional_gate=conditional_gate.detach(),
            candidate_ratio=has_float.mean().detach(),
            residual_norm=applied_residual.detach().float().norm(dim=-1),
            residual_ratio=residual_ratio.detach(),
            fusion_alpha=self.alpha.detach().clone())
        return fused, diagnostics


class STACQueryMemory(nn.Module):
    """STAC-QM wrapper: align memory, read it, then gate-fuse residuals."""

    def __init__(self,
                 enabled=True,
                 embed_dims=256,
                 num_heads=8,
                 spatial_radius=12.0,
                 topk=32,
                 max_age=3.0,
                 lambda_position=1.0,
                 lambda_time=1.0,
                 lambda_reliability=None,
                 lambda_confidence=1.0,
                 dropout=0.0,
                 motion_compensation=True,
                 max_velocity=20.0,
                 pc_range=None,
                 fusion_gate_bias=-4.0,
                 fusion_alpha_init=0.0,
                 fusion_out_proj_gain=0.0,
                 **kwargs):
        super().__init__()
        del kwargs
        self.enabled = bool(enabled)
        self.motion_compensation = bool(motion_compensation)
        self.max_velocity = float(max_velocity)
        if pc_range is None:
            pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.aligner = EgoPoseAligner(pc_range)
        self.attention = CausalQueryMemoryAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            lambda_position=lambda_position,
            lambda_time=lambda_time,
            lambda_reliability=lambda_reliability,
            lambda_confidence=lambda_confidence,
            spatial_radius=spatial_radius,
            topk=topk,
            max_age=max_age,
            pc_range=pc_range)
        self.fusion = ConfidenceGatedFusion(
            embed_dims=embed_dims,
            ffn_dims=embed_dims * 2,
            gate_bias=fusion_gate_bias,
            alpha_init=fusion_alpha_init,
            out_proj_gain=fusion_out_proj_gain)
        # zero-initialized -> exact ego-only alignment until trained
        self.motion_compensator = QueryMotionCompensator(
            embed_dims=embed_dims, max_velocity=max_velocity, max_age=max_age)

    def forward(self,
                query_feat,
                query_points_metric,
                current_confidence,
                memory=None,
                target_ego2global=None,
                future_offset=0.0,
                **memory_kwargs):
        if not self.enabled:
            return query_feat, dict(enabled=False)
        if memory is None:
            memory = memory_kwargs
        if not memory:
            identity = self._training_identity(query_feat)
            return identity, self._identity_diagnostics(query_feat)
        memory_query_feat = memory.get('memory_query_feat')
        memory_points_metric = memory.get('memory_points_metric')
        memory_conf = memory.get('memory_conf')
        memory_valid = memory.get('memory_valid')
        memory_age = memory.get('memory_age')
        memory_reliability = memory.get('memory_reliability')
        source_ego = memory.get('memory_source_ego2global')
        required = [
            memory_query_feat, memory_points_metric, memory_conf, memory_valid,
            memory_age
        ]
        if any(x is None for x in required):
            raise KeyError(
                'STAC-QM memory requires memory_query_feat, '
                'memory_points_metric, memory_conf, memory_valid, and '
                'memory_age')
        if memory_valid.numel() == 0 or not memory_valid.to(
                query_feat.device).bool().any():
            identity = self._training_identity(query_feat)
            return identity, self._identity_diagnostics(query_feat)

        if memory_query_feat.dim() == 3:
            memory_query_feat = memory_query_feat.unsqueeze(0)
            memory_points_metric = memory_points_metric.unsqueeze(0)
            memory_conf = memory_conf.unsqueeze(0)
            memory_valid = memory_valid.unsqueeze(0)
            memory_age = memory_age.unsqueeze(0)
            if memory_reliability is not None and memory_reliability.dim() == 2:
                memory_reliability = memory_reliability.unsqueeze(0)
            if source_ego is not None and source_ego.dim() == 3:
                source_ego = source_ego.unsqueeze(0)
        if memory_query_feat.shape[0] != query_feat.shape[0]:
            raise ValueError(
                'STAC-QM does not share memory across batch samples: '
                f'query B={query_feat.shape[0]}, memory B='
                f'{memory_query_feat.shape[0]}')
        if source_ego is None or target_ego2global is None:
            raise ValueError(
                'memory_source_ego2global and target_ego2global are required '
                'to align valid STAC-QM memory')

        memory_query_feat = memory_query_feat.to(query_feat.device)
        base_age = memory_age.to(query_feat.device).float()  # [B, K, M]
        # future-aware age: observation uses future_offset=0, scheduled query
        # group k uses (k+1) * frame_interval. Cached timestamps are unchanged.
        effective_age = compute_effective_age(base_age, future_offset)

        # (1) ego-pose alignment: P_{h->t} = G_t^{-1} G_h P_h
        aligned_points = self.aligner(
            memory_points_metric.to(query_feat.device),
            source_ego.to(query_feat.device),
            target_ego2global.to(query_feat.device))

        # (2) zero-initialized object-motion residual on the ego-aligned points
        motion_diag = {}
        if self.motion_compensation:
            velocity = self.motion_compensator(memory_query_feat, effective_age)
            # single translation shared by all R points of a query
            shift = effective_age[..., None, None] * velocity[..., None, :]
            aligned_points = aligned_points + shift.to(aligned_points.dtype)
            residual_norm = shift.norm(dim=-1)  # [B, K, M, R]
            motion_diag = dict(
                motion_residual_mean=residual_norm.mean().detach(),
                motion_residual_max=residual_norm.max().detach(),
                motion_velocity_norm=velocity.norm(dim=-1).mean().detach())

        if memory_reliability is not None:
            memory_reliability = memory_reliability.to(query_feat.device)
        memory_output, attn_diag = self.attention(
            query_feat=query_feat,
            query_points_metric=query_points_metric,
            memory_query_feat=memory_query_feat,
            memory_points_metric=aligned_points,
            memory_conf=memory_conf.to(query_feat.device),
            memory_age=effective_age,
            memory_valid=memory_valid.to(query_feat.device),
            memory_reliability=memory_reliability)
        fused, gate_diag = self.fusion(
            query_feat,
            memory_output,
            current_confidence,
            attn_diag['support_conf'],
            attn_diag['has_candidate'])
        diagnostics = dict(attn_diag)
        diagnostics.update(gate_diag)
        diagnostics.update(motion_diag)
        diagnostics['base_age'] = base_age.mean().detach()
        diagnostics['future_offset'] = float(future_offset)
        diagnostics['effective_age_mean'] = effective_age.mean().detach()
        return fused, diagnostics

    def _training_identity(self, query_feat):
        """Keep empty-memory batches numerically exact but differentiable.

        Scene-boundary samples can have no target-age history. In Memory-only
        training the base model is frozen, so returning ``query_feat`` directly
        would produce a loss with no grad_fn and make both smoke and formal
        optimizer hooks fail. Zero-valued anchors give every trainable STAC-QM
        parameter an explicit zero gradient without inventing connectivity.
        """
        if not self.training:
            return query_feat
        zero = query_feat.new_zeros(())
        has_trainable = False
        for param in self.parameters():
            if param.requires_grad and param.numel():
                zero = zero + param.reshape(-1)[0].to(query_feat.dtype) * 0.0
                has_trainable = True
        if not has_trainable:
            return query_feat
        return query_feat + zero

    def _identity_diagnostics(self, query_feat):
        B, Q = query_feat.shape[:2]
        device = query_feat.device
        return dict(
            has_candidate=torch.zeros(B, Q, device=device, dtype=torch.bool),
            support_conf=torch.zeros(B, Q, device=device),
            support_reliability=torch.zeros(B, Q, device=device),
            candidate_count=torch.zeros(B, Q, device=device, dtype=torch.long),
            topk_candidate_count=torch.zeros(B, Q, device=device, dtype=torch.long),
            avg_distance=torch.zeros(B, Q, device=device),
            avg_age=torch.zeros(B, Q, device=device),
            effective_age=torch.zeros(B, Q, device=device),
            avg_gate=torch.zeros(B, Q, device=device),
            residual_norm=torch.zeros(B, Q, device=device),
            attention_shape=(B, self.attention.num_heads, Q, 0))
