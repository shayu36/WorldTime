"""Initialization for the new module controls; legacy studies are untouched."""
from recallocc_common import *
MODES = {
    'E2_full_retrieval_concat': ('point', 'none'),
    'E3_current_only_task_time': ('disabled', 'task_horizon'),
    'E3_current_only_70pct': ('disabled', 'task_horizon_fixed_70pct_support'),
    'E3_current_only_50pct': ('disabled', 'task_horizon_fixed_50pct_support'),
    'E3_current_only_30pct': ('disabled', 'task_horizon_fixed_30pct_support'),
    'E3_current_only_20pct': ('disabled', 'task_horizon_fixed_20pct_support'),
}

def create_initialization(root, seed):
    """One canonical *untrained* E4 per seed; strict subsets for each arm.

    This copies actual tensors, not just integer seeds. Frozen base weights
    and all common randomly initialized parameters are bitwise identical.
    """
    import torch
    from mmcv import Config
    from mmdet.apis import set_random_seed
    from mmdet3d.models import build_model
    path = Path(root) / 'initializations' / f'seed_{seed}'
    path.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    set_random_seed(seed, deterministic=True)
    cfg = Config.fromfile(str(CONFIGS / 'E4.py'))
    model = build_model(cfg.model)
    model.init_weights()
    base_path = ROOT / cfg.recallocc.base_checkpoint
    base = state_dict(torch.load(base_path, map_location='cpu'))
    expected = model.state_dict()
    extra = sorted(set(base) - set(expected))
    missing = sorted(set(expected) - set(base))
    forbidden = [k for k in missing if not k.startswith(NEW_BASE_PREFIXES)]
    wrong = [k for k in base.keys() & expected.keys() if base[k].shape != expected[k].shape]
    if extra or forbidden or wrong:
        raise RuntimeError(dict(unexpected=extra, forbidden_missing=forbidden, wrong_shape=wrong))
    # Populate the exact expected key set; strict load never hides errors.
    full = dict(expected)
    full.update(base)
    model.load_state_dict(full, strict=True)
    model.validate_query_memory_training_setup()
    full = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert all(torch.equal(full[k], v) for k, v in base.items())
    hashes = tensor_hashes(full)
    report = dict(seed=seed, base_checkpoint=str(base_path), base_sha256=digest(base_path),
                  allowed_new_keys=missing, arms={}, common_tensor_hashes=hashes,
                  source_hashes=source_hashes())
    report["full_trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report["base_missing_keys"] = missing
    report["base_unexpected_keys"] = extra
    del model, expected
    current_only_subset = None
    for experiment in MODES:
        cfg = Config.fromfile(str(CONFIGS / f'{experiment}.py'))
        model = build_model(cfg.model)
        expected = model.state_dict()
        changed_key = PREFIX + 'adapter.0.weight'
        subset = {k: (full[k] if k in full and full[k].shape == v.shape else v)
                  for k,v in expected.items()}
        if experiment in ('E3_current_only_70pct', 'E3_current_only_50pct',
                          'E3_current_only_30pct', 'E3_current_only_20pct'):
            # Exact paired initialization, including the current-only input
            # projection that differs in shape from the E4 projection.
            if set(current_only_subset) != set(expected):
                raise RuntimeError('Partial E3 parameter topology differs from original E3')
            subset = {k: current_only_subset[k].clone() for k in expected}
        changed = [k for k in expected if k not in full or expected[k].shape != full[k].shape]
        if changed != [changed_key]:
            raise RuntimeError(f'Unexpected mode-specific initialization: {changed}')
        removed = sorted(set(full) - set(expected))
        validate_state(model, subset)
        model.load_state_dict(subset, strict=True)
        model.validate_query_memory_training_setup()
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert all(torch.equal(subset[k], v) for k, v in base.items())
        out = path / f'{experiment}.pth'
        torch.save(dict(state_dict=subset, meta=dict(seed=seed,
            experiment=experiment, initialization='fresh_common_parameters_and_mode_input_projection',
            base_checkpoint_sha256=report['base_sha256'])), out)
        report['arms'][experiment] = dict(checkpoint=str(out), sha256=digest(out),
            missing_keys=[], unexpected_keys=[], mode_specific_keys=changed,
            removed_canonical_keys=removed, trainable_parameters=count,
            trainable_tensors=sum(p.requires_grad for p in model.parameters()),
            total_parameters=sum(p.numel() for p in model.parameters()),
            common_parameters_bitwise_equal=True)
        if experiment == 'E3_current_only_task_time':
            current_only_subset = {k: v.clone() for k, v in subset.items()}
        del model, expected, subset
    write_json(path / 'initialization.json', report)
    return report
