"""Preflight real checkpoint migration before starting A+B training."""
import argparse
import copy
import json
from pathlib import Path

import torch
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.models.sparsedetectors.query_memory import (
    FutureMemoryAdapterV2, decode_semantic_occupancy_logits,
    apply_metric_position_residual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('source')
    parser.add_argument('--report', required=True)
    parser.add_argument('--source-config', help='For C migration, the source A+B config')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    cfg = Config.fromfile(args.config)
    original = torch.load(args.source, map_location='cpu')['state_dict']
    original = {k.removeprefix('module.'): v for k, v in original.items()}
    migrated = torch.load(cfg.load_from, map_location='cpu')
    assert 'optimizer' not in migrated
    assert not cfg.resume_from and not cfg.auto_resume
    for key, value in original.items():
        assert torch.equal(value, migrated['state_dict'][key]), key
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.load_state_dict(migrated['state_dict'], strict=True)
    model.train()
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith('future_memory_adapter_v2.') for n in trainable)
    assert all(not child.training for name, child in model.named_children()
               if name != 'future_memory_adapter_v2')
    assert all(p.grad is None for name, p in model.named_parameters()
               if not name.startswith('future_memory_adapter_v2.'))
    adapter = model.future_memory_adapter_v2.eval()
    source_cfg = Config.fromfile(args.source_config) if args.source_config else cfg
    kw = copy.deepcopy(dict(source_cfg.model.query_memory_cfg.future_memory_adapter_v2))
    kw.update(pc_range=model.pc_range)
    if not args.source_config:
        kw.update(gate_mode='shared', horizon_gate_bias=False)
    old = FutureMemoryAdapterV2(**kw).eval()
    prefix = 'future_memory_adapter_v2.'
    old.load_state_dict({k[len(prefix):]: v for k, v in original.items()
                         if k.startswith(prefix)}, strict=True)
    B, Q, R, C, S = 2, 2, adapter.num_points, adapter.embed_dims, adapter.num_classes
    memory = dict(memory_query_feat=torch.randn(B, 1, 4, C),
                  memory_points_metric=torch.randn(B, 1, 4, R, 3) * .1,
                  memory_valid=torch.ones(B, 1, 4, dtype=torch.bool),
                  memory_age=torch.full((B, 1, 4), 2.5),
                  memory_reliability=torch.full((B, 1, 4), .9),
                  memory_semantic_distribution=torch.softmax(torch.randn(B, 1, 4, S), -1))
    q, points, logits = torch.randn(B, Q, C), torch.zeros(B, Q, R, 3), torch.randn(B, Q, R, S)
    normalized = torch.rand(B, Q, R, 3)
    checks = []
    with torch.no_grad():
        for horizon in range(3):
            a = old(q, points, logits, memory, horizon)
            b = adapter(q, points, logits, memory, horizon)
            for key in ('delta_s', 'delta_o', 'delta_p'):
                assert torch.equal(a[key], b[key]), key
            assert torch.equal(a['gate'].expand_as(b['gate']), b['gate'])
            assert torch.equal(a['gate_occ'], b['gate_occ'])
            assert torch.equal(a['gate_pos'], b['gate_pos'])
            ao = decode_semantic_occupancy_logits(logits, a['delta_s'], a['delta_o'], a['gate'], a['gate_occ'])
            bo = decode_semantic_occupancy_logits(logits, b['delta_s'], b['delta_o'], b['gate'], b['gate_occ'])
            assert all(torch.equal(x, y) for x, y in zip(ao, bo))
            ap = apply_metric_position_residual(normalized, a['gate_pos'] * a['delta_p'], adapter.pc_range)
            bp = apply_metric_position_residual(normalized, b['gate_pos'] * b['delta_p'], adapter.pc_range)
            assert torch.equal(ap, bp)
            checks.append(f'{horizon + 1}s: exact gates/residuals/semantic/occupancy/decode/position')
    report = dict(source=args.source, migrated=cfg.load_from,
                  source_sha256=migrated['meta']['source_sha256'],
                  full_model_strict_load=True, baseline_frozen=True,
                  synthetic_real_weight_checks=checks,
                  trainable_tensors=len(trainable),
                  trainable_scalars=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  extra_parameters=sum(p.numel() for p in adapter.parameters()) - sum(p.numel() for p in old.parameters()),
                  local_epochs=cfg.runner.max_epochs, learning_rate=cfg.optimizer.lr)
    with Path(args.report).open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
