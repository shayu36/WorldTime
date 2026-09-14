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


class FutureMemoryAdapter(nn.Module):
    """Independent residual adapter for the 1/2/3 second outputs.

    This module deliberately has no state-stream side effects.  It consumes a
    baseline query and an aligned fixed-memory context and returns residuals
    for the *current output only*.  The caller must keep the baseline future
    recurrence separate from these returned tensors.

    The attention score combines feature similarity, semantic compatibility,
    relative position, temporal age and memory reliability.  Semantic memory
    is represented as a probability distribution over the 17 non-empty
    occupancy classes; a cached label is used as a one-hot fallback when a
    legacy record does not contain the distribution.
    """

    def __init__(self,
                 embed_dims=256,
                 num_classes=17,
                 num_points=48,
                 num_heads=8,
                 horizon_count=3,
                 topk=16,
                 spatial_radius=12.0,
                 max_age=8.0,
                 dropout=0.0,
                 gate_bias=-1.0,
                 pc_range=None):
        super().__init__()
        if int(embed_dims) % int(num_heads) != 0:
            raise ValueError('FutureMemoryAdapter requires embed_dims % num_heads == 0')
        if int(num_classes) <= 0 or int(num_points) <= 0:
            raise ValueError('FutureMemoryAdapter requires positive class/point counts')
        if int(horizon_count) != 3:
            raise ValueError('FutureMemoryAdapter currently supports 3 target horizons')
        self.embed_dims = int(embed_dims)
        self.num_classes = int(num_classes)
        self.num_points = int(num_points)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dims // self.num_heads
        self.horizon_count = int(horizon_count)
        self.topk = int(topk)
        self.spatial_radius = float(spatial_radius)
        self.max_age = float(max_age)
        self.query_norm = nn.LayerNorm(self.embed_dims)
        self.memory_norm = nn.LayerNorm(self.embed_dims)
        self.query_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.key_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.value_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.out_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.horizon_embedding = nn.Embedding(self.horizon_count,
                                              self.embed_dims)
        self.semantic_query_proj = nn.Linear(self.num_classes, self.embed_dims)
        self.semantic_memory_proj = nn.Linear(self.num_classes, self.embed_dims)
        self.adapter = nn.Sequential(
            nn.Linear(self.embed_dims * 4 + 4, self.embed_dims * 2),
            nn.LayerNorm(self.embed_dims * 2),
            nn.GELU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.GELU())
        self.delta_s_head = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims), nn.GELU(),
            nn.Linear(self.embed_dims, self.num_points * self.num_classes))
        self.delta_o_head = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims), nn.GELU(),
            nn.Linear(self.embed_dims, self.num_points))
        self.delta_p_head = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims), nn.GELU(),
            nn.Linear(self.embed_dims, self.num_points * 3))
        self.gate_head = nn.Linear(self.embed_dims, 1)
        self.dropout = nn.Dropout(float(dropout))
        if pc_range is None:
            pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.register_buffer('pc_range', torch.as_tensor(pc_range,
                                                          dtype=torch.float32))
        self.aligner = EgoPoseAligner(pc_range)

        # Standard non-zero projections preserve a live second-step gradient
        # path.  Only residual-head last layers are zero so the initial output
        # is exactly the baseline.  There is intentionally no tiny global
        # alpha (the old Phase3 ``1e-3`` scale is not used here).
        for layer in (self.query_proj, self.key_proj, self.value_proj,
                      self.out_proj, self.semantic_query_proj,
                      self.semantic_memory_proj):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.horizon_embedding.weight, mean=0.0, std=0.02)
        for head in (self.delta_s_head, self.delta_o_head, self.delta_p_head):
            nn.init.xavier_uniform_(head[0].weight)
            nn.init.zeros_(head[0].bias)
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        nn.init.xavier_uniform_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(gate_bias))

    def _check_query(self, query_feat, query_points, baseline_logits):
        if query_feat.dim() != 3:
            raise ValueError('baseline query feature must be [B, Q, C]')
        if query_feat.shape[-1] != self.embed_dims:
            raise ValueError('baseline query feature has an unexpected channel dimension')
        if query_points.dim() != 4 or query_points.shape[:2] != query_feat.shape[:2] \
                or query_points.shape[-1] != 3:
            raise ValueError('baseline query points must be [B, Q, R, 3]')
        if query_points.shape[-2] != self.num_points:
            raise ValueError(
                f'baseline query points use R={query_points.shape[-2]}, '
                f'but adapter was configured for num_points={self.num_points}')
        if baseline_logits.dim() != 4 or \
                baseline_logits.shape[:3] != query_points.shape[:3] or \
                baseline_logits.shape[-1] != self.num_classes:
            raise ValueError(
                'baseline absolute logits must be [B, Q, R, num_classes] '
                f'with num_classes={self.num_classes}')

    def _memory_semantics(self, memory, B, N, device, dtype):
        dist = memory.get('memory_semantic_distribution')
        labels = memory.get('memory_label')
        if dist is None:
            if labels is None:
                raise KeyError(
                    'FutureMemoryAdapter requires memory_semantic_distribution '
                    'or memory_label')
            labels = labels.to(device=device).long().reshape(B, N)
            if ((labels >= self.num_classes) | (labels < -1)).any():
                raise ValueError('memory_label contains an invalid class index')
            dist = torch.zeros(B, N, self.num_classes, device=device,
                               dtype=dtype)
            valid_label = labels >= 0
            if valid_label.any():
                dist[valid_label] = F.one_hot(
                    labels[valid_label], self.num_classes).to(dtype)
            # Unknown schema-v1 labels use an uninformative, valid prior.
            unknown = ~valid_label
            if unknown.any():
                dist[unknown] = 1.0 / float(self.num_classes)
        else:
            dist = dist.to(device=device, dtype=dtype)
            if dist.numel() != B * N * self.num_classes:
                raise ValueError(
                    'memory_semantic_distribution must have shape '
                    f'[B,K,M,{self.num_classes}] (flattened size '
                    f'{B * N * self.num_classes}), got {tuple(dist.shape)}')
            dist = dist.reshape(B, N, self.num_classes)
            if not torch.isfinite(dist).all():
                raise ValueError('memory_semantic_distribution must be finite')
            if (dist < 0).any():
                raise ValueError('memory_semantic_distribution must be non-negative')
            dist = dist / dist.sum(dim=-1, keepdim=True).clamp_min(_EPS)
            if labels is not None:
                labels = labels.to(device=device).long().reshape(B, N)
                if ((labels >= self.num_classes) | (labels < -1)).any():
                    raise ValueError('memory_label contains an invalid class index')
                # A valid cached label is also a light semantic prior.  This
                # keeps the label functionally causal even when a soft
                # distribution is present, while retaining the cache's richer
                # distribution as the dominant signal.
                valid_label = labels >= 0
                if valid_label.any():
                    one_hot = F.one_hot(
                        labels.clamp_min(0), self.num_classes).to(dtype)
                    prior = valid_label.unsqueeze(-1).to(dtype) * one_hot
                    dist = (0.9 * dist + 0.1 * prior)
                    dist = dist / dist.sum(dim=-1, keepdim=True).clamp_min(_EPS)
        return dist

    def _reshape_memory(self, memory, B, device, dtype):
        required = ('memory_query_feat', 'memory_points_metric', 'memory_valid',
                    'memory_reliability', 'memory_age')
        missing = [key for key in required if memory.get(key) is None]
        if missing:
            raise KeyError('FutureMemoryAdapter memory is missing: ' +
                           ', '.join(missing))
        feat = memory['memory_query_feat'].to(device=device, dtype=dtype)
        points = memory['memory_points_metric'].to(device=device, dtype=dtype)
        valid = memory['memory_valid'].to(device=device).bool()
        reliability = memory['memory_reliability'].to(device=device,
                                                        dtype=torch.float32)
        age = memory['memory_age'].to(device=device, dtype=torch.float32)
        # Normalize the common unbatched cache form without mutating the
        # caller-owned dictionary (the same context is reused at 1/2/3 s).
        semantic_value = memory.get('memory_semantic_distribution')
        label_value = memory.get('memory_label')
        if feat.dim() == 3:
            feat = feat.unsqueeze(0)
            points = points.unsqueeze(0)
            valid = valid.unsqueeze(0)
            reliability = reliability.unsqueeze(0)
            age = age.unsqueeze(0)
            if label_value is not None and label_value.dim() in (1, 2):
                label_value = label_value.unsqueeze(0)
            if semantic_value is not None and semantic_value.dim() in (2, 3):
                semantic_value = semantic_value.unsqueeze(0)
        if feat.dim() != 4 or feat.shape[0] != B:
            raise ValueError('memory_query_feat must be [B, K, M, C]')
        if feat.shape[-1] != self.embed_dims:
            raise ValueError('memory feature channel dimension mismatch')
        if points.dim() != 5 or points.shape[:3] != feat.shape[:3] \
                or points.shape[-1] != 3:
            raise ValueError('memory_points_metric must be [B, K, M, R, 3]')
        expected_memory_shape = feat.shape[:3]
        if label_value is not None and label_value.shape != expected_memory_shape:
            raise ValueError(
                'memory_label must be [B, K, M] matching memory features, '
                f'got {tuple(label_value.shape)}')
        if semantic_value is not None and (
                semantic_value.dim() != 4 or
                semantic_value.shape[:3] != expected_memory_shape or
                semantic_value.shape[-1] != self.num_classes):
            raise ValueError(
                'memory_semantic_distribution must be [B, K, M, '
                f'{self.num_classes}] matching memory features, got '
                f'{tuple(semantic_value.shape)}')
        if valid.shape != feat.shape[:3] or reliability.shape != feat.shape[:3] \
                or age.shape != feat.shape[:3]:
            raise ValueError('memory feature/validity/reliability/age shapes mismatch')
        K, M = feat.shape[1:3]
        N = K * M
        flat = dict(
            feat=feat.reshape(B, N, self.embed_dims),
            points=points.reshape(B, N, points.shape[-2], 3),
            valid=valid.reshape(B, N),
            reliability=reliability.reshape(B, N),
            age=age.reshape(B, N))
        semantic_memory = dict(memory)
        if label_value is not None:
            semantic_memory['memory_label'] = label_value
        if semantic_value is not None:
            semantic_memory['memory_semantic_distribution'] = semantic_value
        flat['semantic'] = self._memory_semantics(semantic_memory, B, N, device,
                                                  dtype)
        return flat

    def _empty_result(self, query_feat, horizon_id):
        B, Q, _ = query_feat.shape
        zeros_s = query_feat.new_zeros(B, Q, self.num_points, self.num_classes)
        zeros_o = query_feat.new_zeros(B, Q, self.num_points, 1)
        zeros_p = query_feat.new_zeros(B, Q, self.num_points, 3)
        zeros_g = query_feat.new_zeros(B, Q, 1)
        return dict(delta_s=zeros_s, delta_o=zeros_o, delta_p=zeros_p,
                    gate=zeros_g, context=query_feat.new_zeros(B, Q, self.embed_dims),
                    attention_query=query_feat.new_zeros(B, Q, self.embed_dims),
                    diagnostics=dict(has_candidate=torch.zeros(
                        B, Q, dtype=torch.bool, device=query_feat.device),
                        candidate_count=torch.zeros(B, Q, dtype=torch.long,
                                                     device=query_feat.device),
                        horizon_id=int(horizon_id)))

    def forward(self,
                query_feat,
                query_points_metric,
                baseline_logits,
                memory,
                horizon_id,
                target_ego2global=None):
        """Return independent semantic/occupancy/position residuals.

        ``horizon_id`` is 0, 1, or 2 for 1s, 2s, or 3s.  The embedding is
        added to the projected query before attention scores are computed.
        """
        self._check_query(query_feat, query_points_metric, baseline_logits)
        if int(horizon_id) not in range(self.horizon_count):
            raise ValueError(f'horizon_id must be in [0, {self.horizon_count})')
        if memory is None:
            return self._empty_result(query_feat, horizon_id)
        B, Q, _ = query_feat.shape
        # Cache points are stored in their source ego frame.  Align them to
        # the current query frame before distance/support computation.  The
        # optional argument keeps the adapter easy to exercise with synthetic
        # already-aligned memory in unit tests.
        memory_for_read = dict(memory)
        source_ego = memory_for_read.get('memory_source_ego2global')
        if source_ego is not None and target_ego2global is not None:
            raw_points = memory_for_read['memory_points_metric'].to(
                device=query_feat.device, dtype=query_feat.dtype)
            source_ego = source_ego.to(device=query_feat.device,
                                       dtype=torch.float32)
            target_ego2global = target_ego2global.to(
                device=query_feat.device, dtype=torch.float32)
            if raw_points.dim() == 4:
                raw_points = raw_points.unsqueeze(0)
            memory_for_read['memory_points_metric'] = self.aligner(
                raw_points, source_ego, target_ego2global)
            # The points are now in the target frame; do not attempt a second
            # alignment after the K/M flattening step.
            memory_for_read.pop('memory_source_ego2global', None)
        flat = self._reshape_memory(memory_for_read, B, query_feat.device,
                                    query_feat.dtype)
        N = flat['feat'].shape[1]
        if N == 0:
            return self._empty_result(query_feat, horizon_id)
        current_sem = F.softmax(baseline_logits.float(), dim=-1).mean(dim=-2)
        current_sem = current_sem / current_sem.sum(dim=-1, keepdim=True).clamp_min(_EPS)
        horizon = self.horizon_embedding.weight[int(horizon_id)].to(
            device=query_feat.device, dtype=query_feat.dtype).view(1, 1, -1)
        horizon = horizon.expand(B, Q, -1)
        q_input = self.query_norm(query_feat) + horizon + self.semantic_query_proj(
            current_sem.to(self.semantic_query_proj.weight.dtype)).to(query_feat.dtype)
        q = self.query_proj(q_input).float().view(
            B, Q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key_proj(self.memory_norm(flat['feat'])).float().view(
            B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        memory_value = self.value_proj(self.memory_norm(flat['feat'])) + \
            self.semantic_memory_proj(flat['semantic'].to(
                self.semantic_memory_proj.weight.dtype)).to(query_feat.dtype)
        v = memory_value.float().view(
            B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        feature_score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
            float(self.head_dim))
        semantic_compat = torch.einsum(
            'bqc,bnc->bqn', current_sem, flat['semantic']).clamp_min(_EPS)
        semantic_score = torch.log(semantic_compat).unsqueeze(1)
        q_center = query_points_metric.float().mean(dim=-2)
        m_center = flat['points'].float().mean(dim=-2)
        distance = torch.sqrt(((q_center[:, :, None] - m_center[:, None]) ** 2).sum(-1).clamp_min(0.0))
        memory_valid = flat['valid'] & torch.isfinite(flat['reliability']) \
            & (flat['reliability'] > 0) & torch.isfinite(flat['age']) \
            & (flat['age'] > 0) & (flat['age'] <= self.max_age)
        candidate = memory_valid[:, None, :] & (distance <= self.spatial_radius)
        score = feature_score + semantic_score
        score = score - distance[:, None] ** 2 / (self.spatial_radius ** 2 + _EPS)
        score = score - flat['age'][:, None, None] / (self.max_age + _EPS)
        score = score + torch.log(flat['reliability'].clamp_min(_EPS))[:, None, None]
        topk = min(self.topk, N)
        expanded = candidate[:, None].expand(B, self.num_heads, Q, N)
        top_scores, top_idx = torch.topk(
            score.masked_fill(~expanded, torch.finfo(score.dtype).min),
            k=topk, dim=-1)
        top_valid = torch.gather(expanded, -1, top_idx)
        weights = self.dropout(safe_masked_softmax(top_scores, top_valid, dim=-1))
        gather = top_idx[..., None].expand(B, self.num_heads, Q, topk,
                                            self.head_dim)
        selected = torch.gather(v[:, :, None].expand(
            B, self.num_heads, Q, N, self.head_dim), 3, gather)
        context = (weights[..., None] * selected).sum(dim=-2).permute(
            0, 2, 1, 3).reshape(B, Q, self.embed_dims)
        context = self.out_proj(context.to(self.out_proj.weight.dtype)).to(query_feat.dtype)
        has_candidate = top_valid.any(dim=-1).any(dim=1)
        context = context * has_candidate.unsqueeze(-1).to(context.dtype)
        selected_rel = torch.gather(flat['reliability'][:, None, None].expand(
            B, self.num_heads, Q, N), -1, top_idx)
        selected_age = torch.gather(flat['age'][:, None, None].expand(
            B, self.num_heads, Q, N), -1, top_idx)
        selected_dist = torch.gather(distance[:, None].expand(
            B, self.num_heads, Q, N), -1, top_idx)
        den = top_valid.float().sum(dim=-1).clamp_min(1.0)
        support_rel = (weights * selected_rel).sum(-1) / den
        support_age = (weights * selected_age).sum(-1) / den
        support_dist = (weights * selected_dist).sum(-1) / den
        any_head = top_valid.any(dim=-1)
        head_den = any_head.float().sum(dim=1).clamp_min(1.0)
        support_rel = (support_rel * any_head.float()).sum(dim=1) / head_den
        support_age = (support_age * any_head.float()).sum(dim=1) / head_den
        support_dist = (support_dist * any_head.float()).sum(dim=1) / head_den
        diagnostics = dict(
            has_candidate=has_candidate,
            candidate_count=candidate.sum(dim=-1),
            support_reliability=support_rel.detach(),
            average_age=support_age.detach(),
            average_distance=support_dist.detach(),
            horizon_id=int(horizon_id),
            attention_query=q_input,
            attention_scores=score.detach())
        diag_features = torch.stack([
            candidate.sum(dim=-1).float() / float(max(N, 1)),
            support_rel,
            (support_age / (self.max_age + _EPS)).clamp(0, 1),
            (support_dist / (self.spatial_radius + _EPS)).clamp(0, 1)], dim=-1)
        adapter_input = torch.cat([
            query_feat, context, query_feat - context, horizon, diag_features],
            dim=-1)
        adapted = self.adapter(adapter_input)
        delta_s = self.delta_s_head(adapted).reshape(
            B, Q, self.num_points, self.num_classes)
        delta_o = self.delta_o_head(adapted).reshape(
            B, Q, self.num_points, 1)
        delta_p = self.delta_p_head(adapted).reshape(B, Q, self.num_points, 3)
        gate = torch.sigmoid(self.gate_head(adapted)) * has_candidate.unsqueeze(-1).to(adapted.dtype)
        diagnostics['attention_query'] = q_input
        diagnostics['memory_context'] = context
        return dict(delta_s=delta_s, delta_o=delta_o, delta_p=delta_p,
                    gate=gate, context=context,
                    attention_query=q_input, diagnostics=diagnostics)


def decode_semantic_occupancy_logits(base_logits, delta_s, delta_o, gate):
    """Compose V2 semantic and occupancy residuals for legacy decoding."""
    if gate.dim() == delta_s.dim() - 1:
        gate = gate.unsqueeze(-1)
    s_sem = base_logits + gate * delta_s
    occupancy = base_logits.max(dim=-1, keepdim=True).values + gate * delta_o
    s_decode = s_sem + (occupancy - s_sem.max(dim=-1, keepdim=True).values)
    return s_sem, occupancy, s_decode


class FutureMemoryAdapterV2(nn.Module):
    """Two-stage geometry/semantic Query-to-Point future memory adapter.

    The original :class:`FutureMemoryAdapter` is intentionally retained for
    V1 checkpoint compatibility.  This class is output-only and consumes a
    baseline snapshot; its point-level residuals are never fed to recurrence.
    """

    HORIZON_SECONDS = (1.0, 2.0, 3.0)

    def __init__(self, embed_dims=256, num_classes=17, num_points=48,
                 num_heads=8, coarse_topk_geometry=8,
                 coarse_topk_semantic=8, point_topk=8, coarse_radius=6.0,
                 point_radius=6.0, max_effective_age=8.0,
                 semantic_uncertainty_gamma=2.0, semantic_weight_floor=0.1,
                 semantic_dropout_probability=0.0, query_chunk_size=64,
                 gate_bias=-1.0, dropout=0.0, pc_range=None,
                 voxel_size=(0.4, 0.4, 0.4), **kwargs):
        super().__init__()
        del kwargs
        if int(embed_dims) % int(num_heads) != 0:
            raise ValueError('FutureMemoryAdapterV2 requires embed_dims % num_heads == 0')
        self.embed_dims = int(embed_dims)
        self.num_classes = int(num_classes)
        self.num_points = int(num_points)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dims // self.num_heads
        self.coarse_topk_geometry = max(int(coarse_topk_geometry), 1)
        self.coarse_topk_semantic = max(int(coarse_topk_semantic), 1)
        self.point_topk = max(int(point_topk), 1)
        self.coarse_radius = float(coarse_radius)
        self.point_radius = float(point_radius)
        self.max_effective_age = float(max_effective_age)
        self.semantic_uncertainty_gamma = float(semantic_uncertainty_gamma)
        self.semantic_weight_floor = float(semantic_weight_floor)
        self.semantic_dropout_probability = float(semantic_dropout_probability)
        self.query_chunk_size = max(int(query_chunk_size), 1)
        self.voxel_size = tuple(float(x) for x in voxel_size)
        if pc_range is None:
            pc_range = [-40., -40., -1., 40., 40., 5.4]
        self.register_buffer('pc_range', torch.as_tensor(pc_range, dtype=torch.float32))
        self.aligner = EgoPoseAligner(pc_range)
        self.query_norm = nn.LayerNorm(self.embed_dims)
        self.memory_norm = nn.LayerNorm(self.embed_dims)
        self.query_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.key_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.value_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.out_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.point_query_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.point_key_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.point_value_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.current_point_pos = nn.Sequential(
            nn.Linear(3, self.embed_dims), nn.LayerNorm(self.embed_dims), nn.GELU())
        self.memory_point_pos = nn.Sequential(
            nn.Linear(3, self.embed_dims), nn.LayerNorm(self.embed_dims), nn.GELU())
        self.horizon_embedding = nn.Embedding(3, self.embed_dims)
        self.semantic_query_proj = nn.Linear(self.num_classes, self.embed_dims)
        self.semantic_memory_proj = nn.Linear(self.num_classes, self.embed_dims)
        # query, point position, context, difference, horizon and six features
        self.adapter = nn.Sequential(
            nn.Linear(self.embed_dims * 5 + 6, self.embed_dims * 2),
            nn.LayerNorm(self.embed_dims * 2), nn.GELU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
            nn.LayerNorm(self.embed_dims), nn.GELU())
        self.delta_s_head = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims), nn.GELU(),
            nn.Linear(self.embed_dims, self.num_classes))
        self.delta_o_head = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims), nn.GELU(),
            nn.Linear(self.embed_dims, 1))
        self.delta_p_head = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims), nn.GELU(),
            nn.Linear(self.embed_dims, 3))
        self.gate_head = nn.Linear(self.embed_dims, 1)
        self.dropout = nn.Dropout(float(dropout))
        self._init_parameters(gate_bias)

    def _init_parameters(self, gate_bias):
        for layer in (self.query_proj, self.key_proj, self.value_proj,
                      self.out_proj, self.point_query_proj,
                      self.point_key_proj, self.point_value_proj,
                      self.semantic_query_proj, self.semantic_memory_proj):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.horizon_embedding.weight, mean=0., std=.02)
        for head in (self.delta_s_head, self.delta_o_head, self.delta_p_head):
            nn.init.xavier_uniform_(head[0].weight)
            nn.init.zeros_(head[0].bias)
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        nn.init.xavier_uniform_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(gate_bias))

    def _empty(self, query_feat, horizon_id):
        B, Q, _ = query_feat.shape
        z = query_feat.new_zeros
        return dict(
            delta_s=z(B, Q, self.num_points, self.num_classes),
            delta_o=z(B, Q, self.num_points, 1),
            delta_p=z(B, Q, self.num_points, 3),
            gate=z(B, Q, self.num_points, 1),
            point_gate=z(B, Q, self.num_points, 1),
            context=z(B, Q, self.num_points, self.embed_dims),
            point_context=z(B, Q, self.num_points, self.embed_dims),
            attention_query=z(B, Q, self.num_points, self.embed_dims),
            diagnostics=dict(
                has_candidate=torch.zeros(B, Q, self.num_points,
                                           dtype=torch.bool, device=query_feat.device),
                candidate_count=torch.zeros(B, Q, self.num_points,
                                             dtype=torch.long, device=query_feat.device),
                coarse_candidate_count=torch.zeros(B, Q, dtype=torch.long,
                                                    device=query_feat.device),
                base_age=z(B, 0), effective_age=z(B, 0),
                horizon_seconds=float(self.HORIZON_SECONDS[horizon_id]),
                point_attention_shape=(B, 0, self.num_points, self.num_heads, 0)))

    def _semantics(self, memory, B, N, device, dtype):
        sem = memory.get('memory_semantic_distribution')
        labels = memory.get('memory_label')
        if sem is None:
            sem = torch.full((B, N, self.num_classes), 1. / self.num_classes,
                             device=device, dtype=dtype)
            if labels is not None:
                labels = labels.to(device=device).long().reshape(B, N)
                if ((labels >= self.num_classes) | (labels < -1)).any():
                    raise ValueError('memory_label contains an invalid class index')
                valid = labels >= 0
                if valid.any():
                    sem[valid] = F.one_hot(labels[valid], self.num_classes).to(dtype)
        else:
            sem = sem.to(device=device, dtype=dtype).reshape(B, N, -1)
            if sem.shape[-1] != self.num_classes or not torch.isfinite(sem).all() or (sem < 0).any():
                raise ValueError('memory semantic distribution is invalid')
            sem = sem / sem.sum(-1, keepdim=True).clamp_min(_EPS)
        if labels is not None:
            labels = labels.to(device=device).long().reshape(B, N)
            if ((labels >= self.num_classes) | (labels < -1)).any():
                raise ValueError('memory_label contains an invalid class index')
            valid = labels >= 0
            if valid.any():
                one_hot = F.one_hot(labels.clamp_min(0), self.num_classes).to(dtype)
                sem = 0.9 * sem + 0.1 * one_hot * valid.unsqueeze(-1).to(dtype)
                sem = sem / sem.sum(-1, keepdim=True).clamp_min(_EPS)
        return sem

    def _reshape_memory(self, memory, B, device, dtype):
        required = ('memory_query_feat', 'memory_points_metric', 'memory_valid',
                    'memory_reliability', 'memory_age')
        missing = [k for k in required if memory.get(k) is None]
        if missing:
            raise KeyError('FutureMemoryAdapterV2 memory is missing: ' + ', '.join(missing))
        feat = memory['memory_query_feat'].to(device=device, dtype=dtype)
        points = memory['memory_points_metric'].to(device=device, dtype=dtype)
        valid = memory['memory_valid'].to(device=device).bool()
        rel = memory['memory_reliability'].to(device=device, dtype=torch.float32)
        age = memory['memory_age'].to(device=device, dtype=torch.float32)
        finite_query = torch.isfinite(feat).all(-1)
        finite_points = torch.isfinite(points).all(-1).all(-1)
        finite_rel = torch.isfinite(rel)
        finite_age = torch.isfinite(age)
        valid = valid & finite_query & finite_points & finite_rel & finite_age
        feat = torch.nan_to_num(feat, nan=0., posinf=0., neginf=0.)
        points = torch.nan_to_num(points, nan=0., posinf=0., neginf=0.)
        rel = torch.nan_to_num(rel, nan=0., posinf=0., neginf=0.)
        age = torch.nan_to_num(age, nan=0., posinf=0., neginf=0.)
        sem = memory.get('memory_semantic_distribution')
        labels = memory.get('memory_label')
        if feat.dim() == 3:
            feat, points, valid, rel, age = [x.unsqueeze(0) for x in
                                              (feat, points, valid, rel, age)]
            if sem is not None and sem.dim() in (2, 3): sem = sem.unsqueeze(0)
            if labels is not None and labels.dim() in (1, 2): labels = labels.unsqueeze(0)
        if feat.dim() != 4 or feat.shape[0] != B or points.dim() != 5 or points.shape[:3] != feat.shape[:3]:
            raise ValueError('memory tensors must be [B,K,M,C] and [B,K,M,R,3]')
        if valid.shape != feat.shape[:3] or rel.shape != feat.shape[:3] or age.shape != feat.shape[:3]:
            raise ValueError('memory feature/validity/reliability/age shapes mismatch')
        K, M = feat.shape[1:3]
        N = K * M
        if sem is not None and (sem.shape[:3] != feat.shape[:3] or sem.shape[-1] != self.num_classes):
            raise ValueError('memory_semantic_distribution shape mismatch')
        if labels is not None and labels.shape != feat.shape[:3]:
            raise ValueError('memory_label shape mismatch')
        view = dict(memory, memory_semantic_distribution=sem, memory_label=labels)
        return dict(feat=feat.reshape(B, N, self.embed_dims),
                    points=points.reshape(B, N, points.shape[-2], 3),
                    valid=valid.reshape(B, N), rel=rel.reshape(B, N),
                    base_age=age.reshape(B, N),
                    semantic=self._semantics(view, B, N, device, dtype))

    def _semantic_features(self, logits):
        logits = torch.nan_to_num(logits.float(), nan=0., posinf=30., neginf=-30.)
        probs = F.softmax(logits, -1)
        entropy = -(probs * torch.log(probs.clamp_min(_EPS))).sum(-1)
        uncertainty = (entropy / math.log(max(self.num_classes, 2))).clamp(0., 1.)
        top = torch.topk(probs, min(2, self.num_classes), -1).values
        margin = top[..., 0] - (top[..., 1] if self.num_classes > 1 else 0.)
        weight = (1. - uncertainty).pow(self.semantic_uncertainty_gamma)
        weight = weight.clamp_min(self.semantic_weight_floor)
        dropped = torch.zeros_like(weight, dtype=torch.bool)
        if self.training and self.semantic_dropout_probability > 0:
            dropped = torch.rand_like(weight) < self.semantic_dropout_probability
            probs = torch.where(dropped.unsqueeze(-1),
                                torch.full_like(probs, 1. / self.num_classes), probs)
            weight = torch.where(dropped, weight.new_full((), self.semantic_weight_floor), weight)
        return probs, uncertainty, margin, weight, dropped

    @staticmethod
    def _union(a, av, b, bv):
        idx = torch.cat([a, b], -1)
        valid = torch.cat([av, bv], -1)
        chosen = torch.zeros_like(valid)
        for j in range(idx.shape[-1]):
            prior = chosen[..., :j] & (idx[..., :j] == idx[..., j:j + 1])
            chosen[..., j] = valid[..., j] & ~prior.any(-1)
        return idx, chosen

    @staticmethod
    def aggregate_support_statistics(weights, top_valid, selected_rel,
                                     selected_age, selected_dist):
        """Return head-averaged support statistics using weight mass.

        ``weights`` may have dropped entries and an explicit head dimension:
        ``[B,Q,R,H,K]``.  The denominator is the valid softmax mass, never the
        number of candidates, and only heads with at least one valid entry are
        included in the final average.
        """
        valid_weights = weights * top_valid.to(weights.dtype)
        mass = valid_weights.sum(-1).clamp_min(_EPS)
        rel_h = (valid_weights * selected_rel).sum(-1) / mass
        age_h = (valid_weights * selected_age).sum(-1) / mass
        dist_h = (valid_weights * selected_dist).sum(-1) / mass
        head_valid = top_valid.any(-1)
        head_mass = head_valid.to(weights.dtype).sum(-1).clamp_min(1.)
        rel = (rel_h * head_valid.to(rel_h.dtype)).sum(-1) / head_mass
        age = (age_h * head_valid.to(age_h.dtype)).sum(-1) / head_mass
        dist = (dist_h * head_valid.to(dist_h.dtype)).sum(-1) / head_mass
        return rel, age, dist

    def _coarse(self, query, points, sem, sem_weight, flat, effective_age, horizon):
        B, Q, _ = query.shape
        N = flat['feat'].shape[1]
        # Geometry/feature retrieval is deliberately independent of the
        # current explicit semantic distribution.  Semantic compatibility is
        # added only in the second candidate path below.
        q = self.query_proj(self.query_norm(query) + horizon).float()
        k = self.key_proj(self.memory_norm(flat['feat'])).float()
        feature = torch.einsum('bqc,bnc->bqn', q, k) / math.sqrt(self.embed_dims)
        qc = points.float().mean(-2); mc = flat['points'].float().mean(-2)
        dist = torch.sqrt(((qc[:, :, None] - mc[:, None]) ** 2).sum(-1).clamp_min(0.))
        valid = flat['valid'] & torch.isfinite(flat['rel']) & (flat['rel'] > 0)
        valid = valid & torch.isfinite(effective_age) & (effective_age > 0) & (effective_age <= self.max_effective_age)
        geom = feature - dist.square() / (self.coarse_radius ** 2 + _EPS)
        geom = geom - effective_age[:, None] / (self.max_effective_age + _EPS)
        geom = geom + torch.log(flat['rel'].clamp_min(_EPS))[:, None]
        gmask = valid[:, None] & (dist <= self.coarse_radius)
        compat = torch.einsum('bqc,bnc->bqn', sem.mean(2), flat['semantic']).clamp_min(_EPS)
        sem_score = geom + sem_weight.mean(2).unsqueeze(-1) * torch.log(compat)
        kg, ks = min(self.coarse_topk_geometry, N), min(self.coarse_topk_semantic, N)
        neg = torch.finfo(geom.dtype).min
        gi = torch.topk(geom.masked_fill(~gmask, neg), kg, -1).indices
        si = torch.topk(sem_score.masked_fill(~gmask, neg), ks, -1).indices
        gv = torch.gather(gmask, -1, gi); sv = torch.gather(gmask, -1, si)
        ui, uv = self._union(gi, gv, si, sv)
        return ui, uv, gmask, dist, geom, sem_score, gi, gv, si, sv

    def forward(self, query_feat, query_points_metric, baseline_logits, memory,
                horizon_id, target_ego2global=None, **kwargs):
        del kwargs
        if query_feat.dim() != 3 or query_points_metric.dim() != 4 or baseline_logits.dim() != 4:
            raise ValueError('V2 expects [B,Q,C], [B,Q,R,3], [B,Q,R,C]')
        B, Q, C = query_feat.shape
        if C != self.embed_dims or query_points_metric.shape[2] != self.num_points or baseline_logits.shape[:3] != query_points_metric.shape[:3] or baseline_logits.shape[-1] != self.num_classes:
            raise ValueError('V2 query shape does not match configured dimensions')
        horizon_id = int(horizon_id)
        if horizon_id not in (0, 1, 2):
            raise ValueError('horizon_id must be 0, 1, or 2')
        if memory is None or not memory:
            return self._empty(query_feat, horizon_id)
        memory_for_read = dict(memory)
        source_ego = memory_for_read.get('memory_source_ego2global')
        if source_ego is not None and target_ego2global is not None:
            raw_points = memory_for_read['memory_points_metric'].to(
                device=query_feat.device, dtype=query_feat.dtype)
            if raw_points.dim() == 4:
                raw_points = raw_points.unsqueeze(0)
            source_ego = source_ego.to(device=query_feat.device, dtype=torch.float32)
            target_ego2global = target_ego2global.to(device=query_feat.device, dtype=torch.float32)
            if source_ego.dim() == 3:
                source_ego = source_ego.unsqueeze(0)
            if target_ego2global.dim() == 2:
                target_ego2global = target_ego2global.unsqueeze(0)
            memory_for_read['memory_points_metric'] = self.aligner(
                raw_points, source_ego, target_ego2global)
            memory_for_read.pop('memory_source_ego2global', None)
        flat = self._reshape_memory(memory_for_read, B, query_feat.device, query_feat.dtype)
        N = flat['feat'].shape[1]
        if N == 0:
            return self._empty(query_feat, horizon_id)
        hs = float(self.HORIZON_SECONDS[horizon_id])
        h = self.horizon_embedding.weight[horizon_id].to(query_feat).view(1, 1, 1, C).expand(B, Q, self.num_points, C)
        sem, uncertainty, margin, sem_weight, dropped = self._semantic_features(baseline_logits)
        effective_age = flat['base_age'] + hs
        coarse_idx, coarse_valid, global_valid, coarse_dist, geom, sem_score, geometry_idx, geometry_valid, semantic_idx, semantic_valid = self._coarse(
            query_feat, query_points_metric, sem, sem_weight, flat, effective_age,
            h[:, :, 0, :])
        Ku = coarse_idx.shape[-1]; Rm = flat['points'].shape[-2]
        context = query_feat.new_zeros(B, Q, self.num_points, C)
        has = torch.zeros(B, Q, self.num_points, dtype=torch.bool, device=query_feat.device)
        counts = torch.zeros(B, Q, self.num_points, dtype=torch.long, device=query_feat.device)
        srel = query_feat.new_zeros(B, Q, self.num_points); sage = srel.clone(); sdist = srel.clone(); disagree = srel.clone()
        max_L = 0
        qbase = self.point_query_proj(self.query_norm(query_feat)).unsqueeze(2)
        qpos = self.current_point_pos(query_points_metric.float()).to(query_feat.dtype)
        coarse_q = self.query_proj(self.query_norm(query_feat) + h[:, :, 0, :] +
                                   self.semantic_query_proj(sem.mean(2))).to(query_feat.dtype)
        coarse_k = self.key_proj(self.memory_norm(flat['feat']))
        coarse_v = self.value_proj(self.memory_norm(flat['feat']))
        qpoint = qbase + qpos + self.semantic_query_proj(sem.to(self.semantic_query_proj.weight.dtype)).to(query_feat.dtype) + h + coarse_q.unsqueeze(2)
        for start in range(0, Q, self.query_chunk_size):
            stop = min(Q, start + self.query_chunk_size)
            idx = coarse_idx[:, start:stop].clamp_min(0); cv = coarse_valid[:, start:stop]
            bi = torch.arange(B, device=query_feat.device)[:, None, None]
            mf = flat['feat'][bi, idx]; ms = flat['semantic'][bi, idx]; mp = flat['points'][bi, idx]
            mk = coarse_k[bi, idx]; mv = coarse_v[bi, idx]
            mr = flat['rel'][bi, idx]; ma = effective_age[bi, idx]
            rm = mp.shape[-2]; L = Ku * rm; max_L = max(max_L, L)
            hf = mf.unsqueeze(-2).expand(-1, -1, -1, rm, -1).reshape(B, stop-start, L, C)
            hk_query = mk.unsqueeze(-2).expand(-1, -1, -1, rm, -1).reshape(B, stop-start, L, C)
            hv_query = mv.unsqueeze(-2).expand(-1, -1, -1, rm, -1).reshape(B, stop-start, L, C)
            hsx = ms.unsqueeze(-2).expand(-1, -1, -1, rm, -1).reshape(B, stop-start, L, self.num_classes)
            hp = mp.reshape(B, stop-start, L, 3)
            hr = mr.unsqueeze(-1).expand(-1, -1, -1, rm).reshape(B, stop-start, L)
            ha = ma.unsqueeze(-1).expand(-1, -1, -1, rm).reshape(B, stop-start, L)
            hvld = flat['valid'][bi, idx] & (ma > 0) & (ma <= self.max_effective_age)
            hvld = hvld.unsqueeze(-1).expand(-1, -1, -1, rm).reshape(B, stop-start, L)
            coarse_point_valid = cv.unsqueeze(-1).expand(-1, -1, -1, rm).reshape(B, stop-start, L)
            hvld = hvld & coarse_point_valid
            qv = qpoint[:, start:stop]
            qk = self.point_query_proj(qv).float().view(B, stop-start, self.num_points, self.num_heads, -1)
            hk = (hk_query + self.point_key_proj(self.memory_norm(hf)) + self.memory_point_pos(hp.float()).to(hf.dtype) + self.semantic_memory_proj(hsx.to(self.semantic_memory_proj.weight.dtype)).to(hf.dtype)).float().view(B, stop-start, L, self.num_heads, -1)
            vv = (hv_query + self.point_value_proj(self.memory_norm(hf)) + self.memory_point_pos(hp.float()).to(hf.dtype)).float().view(B, stop-start, L, self.num_heads, -1)
            score = torch.einsum('bqrhd,bqlhd->bqrhl', qk, hk) / math.sqrt(self.head_dim)
            pd = torch.sqrt(((query_points_metric[:, start:stop, :, None] - hp[:, :, None]) ** 2).sum(-1).clamp_min(0.))
            compat = torch.einsum('bqrc,bqlc->bqrl', sem[:, start:stop], hsx).clamp_min(_EPS)
            score = score - pd.square().unsqueeze(-2) / (self.point_radius ** 2 + _EPS) - ha[:, :, None, None, :] / (self.max_effective_age + _EPS) + torch.log(hr.clamp_min(_EPS))[:, :, None, None, :] + sem_weight[:, start:stop].unsqueeze(-1).unsqueeze(-1) * torch.log(compat).unsqueeze(-2)
            pmask = hvld[:, :, None, :].unsqueeze(-2).expand(-1, -1, self.num_points, self.num_heads, -1) & (pd.unsqueeze(-2) <= self.point_radius)
            pk = min(self.point_topk, L)
            ts, ti = torch.topk(score.masked_fill(~pmask, torch.finfo(score.dtype).min), pk, -1)
            tv = torch.gather(pmask, -1, ti)
            w = self.dropout(safe_masked_softmax(ts, tv, -1))
            gather = ti.transpose(-1, -2)[..., None].expand(B, stop-start, self.num_points, pk, self.num_heads, self.head_dim)
            sv = torch.gather(vv[:, :, None].expand(B, stop-start, self.num_points, L, self.num_heads, self.head_dim), 3, gather).transpose(3, 4)
            # Collapse valid heads only after the per-head weighted read.
            context_heads = (w[..., None] * sv).sum(-2)
            head_valid = tv.any(-1)
            head_den = head_valid.float().sum(-1).clamp_min(1.)
            context_heads = context_heads * head_valid[..., None].to(context_heads.dtype) / head_den[..., None, None]
            context[:, start:stop] = context_heads.reshape(B, stop-start, self.num_points, C).to(context.dtype)
            has[:, start:stop] = head_valid.any(-1); counts[:, start:stop] = pmask.any(-2).sum(-1)
            sr = torch.gather(hr[:, :, None, None, :].expand(B, stop-start, self.num_points, self.num_heads, L), -1, ti)
            sa = torch.gather(ha[:, :, None, None, :].expand(B, stop-start, self.num_points, self.num_heads, L), -1, ti)
            sd = torch.gather(pd.unsqueeze(-2).expand(B, stop-start, self.num_points, self.num_heads, L), -1, ti)
            vw = w * tv.to(w.dtype); mass = vw.sum(-1).clamp_min(_EPS)
            srel[:, start:stop], sage[:, start:stop], sdist[:, start:stop] = self.aggregate_support_statistics(
                w, tv, sr, sa, sd)
            ss = torch.gather(hsx[:, :, None, None].expand(B, stop-start, self.num_points, self.num_heads, L, self.num_classes), 4, ti[..., None].expand(B, stop-start, self.num_points, self.num_heads, pk, self.num_classes))
            mean_s = (vw[..., None] * ss).sum(-2) / mass[..., None]; mean_s = (mean_s * head_valid[..., None].to(mean_s.dtype)).sum(-2) / head_den[..., None]; p = sem[:, start:stop]; mm = (p + mean_s).clamp_min(_EPS) / 2
            js = .5 * ((p * (torch.log(p.clamp_min(_EPS)) - torch.log(mm))).sum(-1) + (mean_s * (torch.log(mean_s.clamp_min(_EPS)) - torch.log(mm))).sum(-1))
            disagree[:, start:stop] = torch.where(has[:, start:stop], js, torch.zeros_like(js))
        # Keep the coarse projections on a differentiable path after the
        # discrete top-k selection.  This gives q/k/v a live second-step
        # gradient while preserving the strict no-candidate zero mask.
        coarse_summary = (coarse_k.mean(dim=1) + coarse_v.mean(dim=1)).view(B, 1, 1, C)
        context = context + qpoint + coarse_summary.to(context.dtype)
        context = self.out_proj(context.to(self.out_proj.weight.dtype)).to(query_feat.dtype) * has.unsqueeze(-1).to(query_feat.dtype)
        features = torch.stack([counts.float() / float(max(Ku * Rm, 1)), srel, (sage / (self.max_effective_age + _EPS)).clamp(0, 1), (sdist / (self.point_radius + _EPS)).clamp(0, 1), uncertainty, disagree], -1)
        qexpanded = query_feat.unsqueeze(2).expand(-1, -1, self.num_points, -1)
        adapted = self.adapter(torch.cat([qexpanded, qpos, context, qexpanded - context, h, features], -1))
        delta_s, delta_o, delta_p = self.delta_s_head(adapted), self.delta_o_head(adapted), self.delta_p_head(adapted)
        point_mask = has.unsqueeze(-1).to(adapted.dtype)
        delta_s = delta_s * point_mask
        delta_o = delta_o * point_mask
        delta_p = delta_p * point_mask
        gate = torch.sigmoid(self.gate_head(adapted)) * point_mask
        diagnostics = dict(has_candidate=has, candidate_count=counts,
                           coarse_candidate_count=coarse_valid.sum(-1),
                           coarse_indices=coarse_idx.detach(), coarse_valid=coarse_valid.detach(),
                           geometry_indices=geometry_idx.detach(), geometry_valid=geometry_valid.detach(),
                           semantic_indices=semantic_idx.detach(), semantic_valid=semantic_valid.detach(),
                           geometry_candidate_count=global_valid.sum(-1),
                           support_reliability=srel.detach(), support_age=sage.detach(), average_age=sage.detach(),
                           support_distance=sdist.detach(), average_distance=sdist.detach(),
                           base_age=flat['base_age'].detach(), effective_age=effective_age.detach(),
                           time_penalty=(effective_age / (self.max_effective_age + _EPS)).detach(),
                           horizon_seconds=hs, semantic_uncertainty=uncertainty.detach(),
                           semantic_margin=margin.detach(), semantic_weight=sem_weight.detach(),
                           semantic_dropout_mask=dropped.detach(), semantic_disagreement=disagree.detach(),
                           coarse_score_shape=tuple(geom.shape), point_attention_shape=(B, min(self.query_chunk_size, Q), self.num_points, self.num_heads, max_L),
                           attention_shape=(B, min(self.query_chunk_size, Q), self.num_points, self.num_heads, max_L))
        diagnostics['point_context'] = context
        diagnostics['point_gate'] = gate
        return dict(delta_s=delta_s, delta_o=delta_o, delta_p=delta_p,
                    gate=gate, point_gate=gate, context=context,
                    point_context=context, attention_query=qpoint,
                    diagnostics=diagnostics)


def sparse_soft_voxel_iou_loss(semantic_logits, occupancy_logits, points,
                               gt_voxel_semantics, points_mask=None,
                               pc_range=(-40., -40., -1., 40., 40., 5.4),
                               voxel_size=(0.4, 0.4, 0.4), num_classes=17,
                               class_weights=None, eps=1e-6):
    """Sparse, differentiable noisy-OR Soft-IoU over trilinear voxels.

    Only eight neighbours per valid point and the union with non-empty GT
    voxels are materialised.  This avoids a dense ``B*200*200*16*17`` tensor
    while retaining gradients through continuous point positions.
    """
    B, Q, R, C = semantic_logits.shape
    device = semantic_logits.device
    dtype = semantic_logits.dtype
    if points_mask is None:
        points_mask = torch.ones(B, Q, R, dtype=torch.bool, device=device)
    probs = torch.sigmoid(occupancy_logits.squeeze(-1)).unsqueeze(-1) * F.softmax(semantic_logits, dim=-1)
    metric = decode_points_metric(points, pc_range).float()
    origin = torch.as_tensor(pc_range[:3], device=device, dtype=torch.float32)
    vs = torch.as_tensor(voxel_size, device=device, dtype=torch.float32)
    grid = torch.as_tensor(gt_voxel_semantics.shape[1:4], device=device, dtype=torch.long)
    offsets = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], device=device)
    weights_cls = (torch.ones(num_classes, device=device, dtype=dtype)
                   if class_weights is None else torch.as_tensor(class_weights, device=device, dtype=dtype))
    total = semantic_logits.new_zeros(())
    for b in range(B):
        gt = gt_voxel_semantics[b].long()
        gt_idx = torch.nonzero(gt != num_classes, as_tuple=False)
        if gt_idx.numel():
            gt_lin = gt_idx[:, 0] * grid[1] * grid[2] + gt_idx[:, 1] * grid[2] + gt_idx[:, 2]
        else:
            gt_lin = torch.zeros(0, dtype=torch.long, device=device)
        p = metric[b].reshape(-1, 3)
        pp = probs[b].reshape(-1, num_classes)
        valid = points_mask[b].reshape(-1).bool() & torch.isfinite(p).all(-1)
        p, pp = p[valid], pp[valid]
        coord = (p - origin) / vs
        base = torch.floor(coord).long()
        frac = (coord - base.float()).clamp(0., 1.)
        idx = base[:, None, :] + offsets[None]
        inside = ((idx >= 0) & (idx < grid)).all(-1)
        corner_w = torch.stack([
            torch.where(offsets[:, d].bool(), frac[:, None, d], 1. - frac[:, None, d])
            for d in range(3)], dim=-1).prod(-1)
        lin = idx[..., 0] * grid[1] * grid[2] + idx[..., 1] * grid[2] + idx[..., 2]
        pred_lin = lin[inside]
        union = torch.unique(torch.cat([pred_lin, gt_lin])) if (pred_lin.numel() or gt_lin.numel()) else pred_lin
        if union.numel() == 0:
            total = total + semantic_logits[b].sum() * 0.
            continue
        pred_vox = semantic_logits.new_zeros(union.numel(), num_classes)
        if pred_lin.numel():
            inv = torch.searchsorted(union, pred_lin)
            # Boolean indexing preserves the point/corner alignment.
            contrib_all = corner_w[..., None] * pp[:, None, :]
            contrib = contrib_all[inside]
            pred_vox.index_add_(0, inv, contrib.to(dtype))
            pred_vox = 1. - torch.exp(-pred_vox)
        target = semantic_logits.new_zeros(union.numel(), num_classes)
        if gt_lin.numel():
            inv_gt = torch.searchsorted(union, gt_lin)
            labels = gt[gt_idx[:, 0], gt_idx[:, 1], gt_idx[:, 2]].clamp(0, num_classes - 1)
            target[inv_gt, labels] = 1.
        inter = (pred_vox * target).sum(0)
        den = pred_vox.sum(0) + target.sum(0) - inter
        total = total + (weights_cls * (1. - inter / den.clamp_min(eps))).sum() / weights_cls.sum().clamp_min(eps)
    return total / max(B, 1)
