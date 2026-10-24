"""Read-only CPU checks for T1 -> Adapter-temporal migration; never trains."""
import argparse
import importlib.util
from pathlib import Path

import torch
from mmcv import Config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-future-adapter-v2-t1-temporal.py'


def load_t1_adapter(adapter, state):
    """Load all old weights; permit ONLY the new temporal read to be absent.

    No source tensors are mutated and no checkpoint is written. This check
    catches accidental shared/V1 sources before MMCV's permissive loader.
    """
    normalized = {k.removeprefix('module.'): v for k, v in state.items()}
    prefix = 'future_memory_adapter_v2.'
    weights = {k[len(prefix):]: v for k, v in normalized.items() if k.startswith(prefix)}
    if not weights:
        raise ValueError('source checkpoint has no future_memory_adapter_v2 weights')
    expected = adapter.state_dict()
    new_keys = {k for k in expected if k.startswith('temporal_read.')}
    missing, unexpected = set(expected) - set(weights), set(weights) - set(expected)
    if missing != new_keys or unexpected:
        raise ValueError(f'not a compatible T1 source: missing={sorted(missing)}, '
                         f'expected_new={sorted(new_keys)}, unexpected={sorted(unexpected)}')
    return adapter.load_state_dict(weights, strict=not new_keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--checkpoint', help='Override source for CPU inspection only')
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = Config.fromfile(args.config)
    path = Path(args.checkpoint or cfg.load_from)
    if not path.is_absolute():
        path = ROOT / path
    if cfg.resume_from is not None or cfg.auto_resume:
        raise ValueError('Round 1 requires a fresh matched optimizer, not resume')
    if not cfg.model.query_memory_cfg.freeze_base_model:
        raise ValueError('Baseline must stay frozen')
    if cfg.optimizer.constructor != 'TrainableOnlyOptimizerConstructor':
        raise ValueError('Trainable-only optimizer constructor is required')
    spec = importlib.util.spec_from_file_location(
        'temporal_check_qm', ROOT / 'mmdet3d/models/sparsedetectors/query_memory.py')
    qm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qm)
    options = dict(cfg.model.query_memory_cfg.future_memory_adapter_v2)
    torch.manual_seed(cfg.seed)
    adapter = qm.FutureMemoryAdapterV2(**options).eval()
    source = torch.load(str(path), map_location='cpu')
    load_t1_adapter(adapter, source['state_dict'])
    reference_options = dict(options, temporal_adapter_enabled=False)
    reference = qm.FutureMemoryAdapterV2(**reference_options).eval()
    load_t1_adapter(reference, source['state_dict'])

    # Nonzero trained residuals, not an untrained zero-head comparison.
    C, R, S = adapter.embed_dims, adapter.num_points, adapter.num_classes
    B, M = 2, 2
    memory = dict(memory_query_feat=torch.randn(B, 1, M, C),
                  memory_points_metric=torch.randn(B, 1, M, R, 3) * .1,
                  memory_valid=torch.ones(B, 1, M, dtype=torch.bool),
                  memory_reliability=torch.full((B, 1, M), .9),
                  memory_age=torch.full((B, 1, M), 2.5))
    state = None
    with torch.no_grad():
        for h in range(3):
            Q = 2 + h
            q, p, s = torch.randn(B, Q, C), torch.randn(B, Q, R, 3) * .1, torch.randn(B, Q, R, S)
            a = reference(q, p, s, memory, h)
            extra = (dict(temporal_state=state, query_ids=torch.arange(Q))
                     if adapter.temporal_adapter_enabled else {})
            b = adapter(q, p, s, memory, h, **extra)
            for key in ('delta_s', 'delta_o', 'delta_p', 'gate_sem', 'gate_occ', 'gate_pos'):
                if not torch.equal(a[key], b[key]):
                    raise AssertionError(f'Initial T1 identity failed: horizon={h}, {key}')
            state = b.get('temporal_state')
    added = sum(p.numel() for k, p in adapter.named_parameters() if k.startswith('temporal_read.'))
    print('Source:', path)
    print('Source epoch:', source.get('meta', {}).get('epoch'))
    print('Strict old-weight coverage / zero temporal contribution / B=2 horizons 1,2,3: PASS')
    print('Adapter parameters:', sum(p.numel() for p in adapter.parameters()), 'new:', added)
    print('Whitelist: future_memory_adapter_v2.* (includes temporal_read.*)')
    print('CPU Adapter only; full model, derivatives, GPU memory and training NOT checked.')


if __name__ == '__main__':
    main()
