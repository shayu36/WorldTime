"""Bounded acceptance: one TRAIN sample, two backwards, ONE volatile update.

No training runner, epoch loop, validation or checkpoint writes. The second
backward reuses frozen snapshots of the same training sample. All updated
fusion weights are discarded on exit. This is not a training/smoke result.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import types

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import build_optimizer, wrap_fp16_model
from mmdet.apis import set_random_seed
from mmdet.utils import compat_cfg, setup_multi_processes
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import patch_config
from mmdet3d.core.optimizer import TrainableOnlyOptimizerConstructor  # noqa: F401


def digest_tensor(value):
    arr = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def gradient_report(model):
    frozen_bad = []
    norms = {}
    for name, param in model.named_parameters():
        if not name.startswith('memory_planning_fusion.'):
            if param.requires_grad or param.grad is not None:
                frozen_bad.append(name)
        else:
            if param.grad is None or not torch.isfinite(param.grad).all():
                raise RuntimeError(f'Missing or nonfinite gradient: {name}')
            norms[name] = float(param.grad.float().norm())
    if frozen_bad:
        raise RuntimeError(f'Frozen parameters received gradients: {frozen_bad}')
    for name, module in model.named_children():
        if name != 'memory_planning_fusion' and module.training:
            raise RuntimeError(f'Frozen child is in train mode: {name}')
    groups = {group: sum(v for k, v in norms.items() if k.startswith('memory_planning_fusion.'+group))
              for group in ('token_encoder', 'ego_encoder', 'horizon_embedding', 'attention', 'residual_head')}
    return dict(parameter_gradient_norms=norms, group_gradient_norms=groups,
                frozen_parameters_grad_none=True, frozen_children_eval=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-planning-fusion.py')
    parser.add_argument('--index', type=int, default=64)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    set_random_seed(17, deterministic=True)
    cfg = patch_config(compat_cfg(Config.fromfile(args.config)))
    setup_multi_processes(cfg)
    cfg.dump(str(output/'effective_config.py'))
    model = build_model(cfg.model)
    model.init_weights()  # Strict T1 loading, missing only NEW fusion keys.
    wrap_fp16_model(model)
    model.cuda().train()
    optimizer = build_optimizer(model, cfg.optimizer)
    setup = model.validate_query_memory_training_setup(optimizer, cfg.optimizer)
    print('Model build / strict T1 load / optimizer whitelist: PASS', flush=True)
    frozen_hashes = {k: digest_tensor(v) for k, v in model.state_dict().items()
                     if not k.startswith('memory_planning_fusion.')}
    frozen_parameters = {k: digest_tensor(v) for k, v in model.named_parameters()
                         if not k.startswith('memory_planning_fusion.')}
    dataset = build_dataset(cfg.data.train)
    sample = dataset[args.index]  # Exactly one sample from TRAIN, not validation.
    batch = collate([sample], samples_per_gpu=1)
    snapshots = []
    saved = {}
    def capture(module, values):
        snapshots.append(tuple(v.detach().clone() if torch.is_tensor(v) else v for v in values))
    handle = model.memory_planning_fusion.register_forward_pre_hook(capture)
    original_loss = model.planning_losses
    def capture_loss(self, outputs, targets):
        saved['targets'] = targets.detach().clone()
        saved['native'] = [x.detach().clone() for x in outputs['baseline_pred_trajs_list']]
        saved['initial_identity'] = all(torch.equal(a, b) for a, b in
            zip(outputs['pred_trajs_list'], outputs['baseline_pred_trajs_list']))
        return original_loss(outputs, targets)
    model.planning_losses = types.MethodType(capture_loss, model)
    parallel = MMDataParallel(model, device_ids=[0])
    torch.cuda.reset_peak_memory_stats()
    scaler = torch.cuda.amp.GradScaler()
    try:
        losses = parallel(return_loss=True, **batch)
    finally:
        handle.remove()
        model.planning_losses = original_loss
    del batch, sample
    if not saved['initial_identity'] or len(snapshots) != 3:
        raise RuntimeError('Initial identity / three planning horizons check failed')
    if not all(bool(s[2].any()) for s in snapshots):
        raise RuntimeError('Selected TRAIN sample lacks Memory at a horizon; choose another explicit --index')
    loss = sum(losses.values())
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite initial trajectory loss')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    first = gradient_report(model)
    if first['group_gradient_norms']['residual_head'] <= 0:
        raise RuntimeError('Existing trajectory loss did not reach the new residual head')
    first_loss = float(loss.detach())
    print('Backward 1: initial identity / new head gradient / frozen modules: PASS', flush=True)
    scaler.step(optimizer)  # The ONLY optimizer update in this acceptance run.
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    # Reuse one sample's detached frozen evidence; no second dataset iteration.
    planned = list(saved['native'])
    for step, snapshot in zip((1, 3, 5), snapshots):
        planned[step] = planned[step] + model.memory_planning_fusion(*snapshot)
    losses2 = model.planning_losses({'pred_trajs_list': planned}, saved['targets'])
    second_loss = sum(losses2.values())
    scaler.scale(second_loss).backward()
    scaler.unscale_(optimizer)
    second = gradient_report(model)
    if not all(value > 0 for value in second['group_gradient_norms'].values()):
        raise RuntimeError(f'Upstream fusion gradient absent: {second["group_gradient_norms"]}')
    updated_frozen = [k for k, v in model.named_parameters() if k in frozen_parameters and digest_tensor(v) != frozen_parameters[k]]
    updated_buffers = [k for k, v in model.state_dict().items() if k in frozen_hashes and digest_tensor(v) != frozen_hashes[k]]
    if updated_frozen or updated_buffers:
        raise RuntimeError(dict(changed_frozen=updated_frozen, changed_frozen_state=updated_buffers))
    report = dict(
        status='PASS', config=args.config, source_checkpoint=cfg.planning_source_checkpoint,
        train_sample_index=args.index, initial_baseline_identity=saved['initial_identity'],
        loss_names=sorted(losses), first_loss=first_loss, second_loss=float(second_loss.detach()),
        first_backward=first, second_backward=second, optimizer_steps=1, backward_calls=2,
        trainable_prefix='memory_planning_fusion.', trainable_count=setup['trainable_count'],
        frozen_parameter_and_buffer_hashes_unchanged=True,
        point_query_counts=[s[1].shape[1] for s in snapshots],
        valid_queries=[int(s[2].sum()) for s in snapshots],
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        elapsed_seconds=time.monotonic()-started,
        checkpoint_saved=False, formal_training_started=False)
    (output/'gradient_acceptance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('first_backward', 'second_backward')}, indent=2), flush=True)
    print('Backward 2: all fusion groups / frozen hashes unchanged: PASS', flush=True)


if __name__ == '__main__':
    main()
