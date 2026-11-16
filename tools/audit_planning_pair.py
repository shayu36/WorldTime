"""Export native Baseline/T1 trajectories and score both with one ST-P3 metric."""

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def read_pickle(path):
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def predict(args):
    source = Path(args.source_root).resolve()
    sys.path.insert(0, str(source))
    os.chdir(source)
    import numpy as np
    import torch
    from mmcv import Config
    from mmcv.parallel import DataContainer, MMDataParallel
    from mmcv.runner import wrap_fp16_model
    from mmdet.apis import set_random_seed
    from mmdet.utils import compat_cfg, setup_multi_processes
    from mmdet3d.datasets import build_dataset, build_dataloader
    from mmdet3d.models import build_model
    from mmdet3d.utils import patch_config

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    set_random_seed(0, deterministic=True)
    cfg = patch_config(compat_cfg(Config.fromfile(args.config)))
    setup_multi_processes(cfg)
    cfg.data.test.test_mode = True
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.align_after_view_transfromation = True
    if args.baseline:
        cfg.model.pop('future_memory_adapter_version', None)
        cfg.model.query_memory_cfg = dict(
            enabled=False, source='cache', freeze_base_model=False,
            future_memory_adapter_enabled=False,
            future_memory_adapter_finetune_mode=False,
            memory_conditioned_refiner=False,
            memory_finetune_mode=False, memory_joint_finetune_mode=False,
            memory_phase2_finetune_mode=False,
            memory_phase3_finetune_mode=False)
    dataset = build_dataset(cfg.data.test)
    manifest = [dict(index=i, token=dataset.data_infos[j]['token'],
                     scene=dataset.data_infos[j]['scene_name'])
                for i, j in enumerate(dataset.temp2nusc_map)]
    if args.limit:
        manifest = manifest[:args.limit]
    if len({r['token'] for r in manifest}) != len(manifest):
        raise ValueError('Duplicate validation token')
    write_json(output / 'manifest.json', manifest)
    cfg.dump(str(output / 'effective_config.py'))
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state = {k.removeprefix('module.'): v
             for k, v in checkpoint['state_dict'].items()}
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    dormant = ('memory_refiner.', 'memory_horizon_embedding.',
               'memory_cls_branch.', 'memory_reg_branch.', 'memory_vel_branch.')
    if unexpected or (missing and not args.baseline):
        raise RuntimeError(dict(missing=missing, unexpected=unexpected))
    if args.baseline and any(not k.startswith(dormant) for k in missing):
        raise RuntimeError(f'Missing active Baseline parameters: {missing}')
    if args.baseline:
        if checkpoint.get('meta', {}).get('epoch') != 56:
            raise ValueError('Expected native epoch56 Baseline checkpoint')
        state = {**expected, **state}
    model.load_state_dict(state, strict=True)
    del state, expected
    epoch = checkpoint.get('meta', {}).get('epoch')
    del checkpoint
    wrap_fp16_model(model)
    model.CLASSES = dataset.CLASSES
    model.cuda().eval()
    if args.baseline:
        model.set_epoch(epoch)
        assert not model.future_memory_adapter_enabled
        assert not model.query_memory_enabled
    else:
        model.validate_query_memory_training_setup()
        assert model.future_memory_adapter_enabled
        assert model.future_memory_adapter_version == 'v2'
    model.eval()
    stamps = model.pts_bbox_head.ind_stamps_all.detach().cpu().numpy()
    np.save(output / 'query_timestamps.npy', stamps)
    parallel = MMDataParallel(model, device_ids=[0]).eval()
    loader = build_dataloader(
        torch.utils.data.Subset(dataset, [r['index'] for r in manifest]),
        samples_per_gpu=1, workers_per_gpu=1, num_gpus=1,
        dist=False, shuffle=False, seed=0)

    def tensor_digest(value, hasher):
        if isinstance(value, DataContainer):
            tensor_digest(value.data, hasher)
        elif isinstance(value, torch.Tensor):
            arr = value.detach().cpu().contiguous().numpy()
            hasher.update(str((arr.shape, arr.dtype)).encode())
            hasher.update(memoryview(arr).cast('B'))
        elif isinstance(value, (list, tuple)):
            for child in value:
                tensor_digest(child, hasher)
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                hasher.update(str(key).encode())
                tensor_digest(value[key], hasher)
        elif isinstance(value, np.ndarray):
            arr = np.ascontiguousarray(value)
            hasher.update(str((arr.shape, arr.dtype)).encode())
            hasher.update(memoryview(arr).cast('B'))
        elif value is None or isinstance(value, (str, int, float, bool)):
            hasher.update(repr(value).encode())
        else:
            raise TypeError(type(value))

    provenance = dict(
        checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=digest(args.checkpoint), checkpoint_epoch=epoch,
        source_root=str(source), source_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        command=sys.argv,
        native_model_call='MMDataParallel(return_loss=False,rescale=True)',
        baseline=args.baseline, samples=len(manifest), seed=0,
        trajectory_format='current-frame cumulative XY, np.cumsum(raw six increments)',
        checkpoint_missing_dormant_keys=missing,
        source_sha256={p: digest(source / p) for p in (
            'mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py',
            'mmdet3d/models/sparsedetectors/query_memory.py',
            'mmdet3d/datasets/nuscenes_dataset_occ_trajectory.py',
            'mmdet3d/datasets/pipelines/loading_query_memory.py')},
        exporter_sha256=digest(__file__),
        manifest_sha256=digest(output / 'manifest.json'))
    write_json(output / 'provenance.json', provenance)
    predictions = {}
    started = time.monotonic()
    print(f'Loaded checkpoint {args.checkpoint}; baseline={args.baseline}; '
          f'samples={len(manifest)}', flush=True)
    with torch.no_grad(), (output / 'progress.jsonl').open('x') as progress:
        for i, batch in enumerate(loader):
            h = hashlib.sha256()
            tensor_digest(batch['img'], h)
            tensor_digest(batch['temporal_ego_states'], h)
            result = parallel(return_loss=False, rescale=True, **batch)
            raw = result['pred_traj'].detach().float().cpu().numpy()
            if raw.shape != (1, 6, 2) or not np.isfinite(raw).all():
                raise ValueError(f'Unexpected trajectory {raw.shape}')
            predictions[manifest[i]['token']] = np.cumsum(raw, axis=1)
            progress.write(json.dumps(dict(
                **manifest[i], input_sha256=h.hexdigest())) + '\n')
            progress.flush()
            if i < 3 or (i + 1) % 100 == 0:
                elapsed = time.monotonic() - started
                print(f'{i+1}/{len(manifest)} elapsed={elapsed:.1f}s '
                      f'mean={elapsed/(i+1):.3f}s/sample', flush=True)
            del result, raw, batch
    with (output / 'trajectories.pkl').open('xb') as stream:
        pickle.dump(predictions, stream)
    write_json(output / 'complete.json', dict(
        samples=len(predictions), elapsed_seconds=time.monotonic()-started,
        trajectory_sha256=digest(output / 'trajectories.pkl')))
    print(f'Export complete: {output}', flush=True)


def metrics(args):
    import numpy as np
    import torch
    root = Path(args.metric_root).resolve()
    sys.path.insert(0, str(root / 'AD-MLP/deps/stp3'))
    from stp3.planning_metrics import PlanningMetric
    torch.set_num_threads(1)
    label_root = root / 'stp3_val'
    predictions = read_pickle(args.prediction)
    ground_truth = read_pickle(label_root / 'stp3_traj_gt.pkl')
    occupancy = read_pickle(label_root / 'stp3_occupancy.pkl')
    tokens = read_pickle(label_root / 'filter_token.pkl')
    selected = [t for t in tokens if t in predictions]
    if set(selected) != set(predictions) or len(selected) != len(predictions):
        raise ValueError('Prediction token set differs from GT filter intersection')
    metric = PlanningMetric(6)
    with torch.no_grad():
        for token in selected:
            pred = torch.from_numpy(np.asarray(predictions[token]).copy())
            if pred.shape[-1] == 2:
                pred = torch.cat([pred, torch.zeros_like(pred[..., :1])], -1)
            gt = torch.as_tensor(ground_truth[token]['gt_trajectory']).unsqueeze(0)
            metric(pred, gt[:, 1:7].clone(), occupancy[token][:, :6])
    scores = metric.compute()
    report = dict(
        samples=len(selected), filter_total=len(tokens),
        prediction=str(Path(args.prediction).resolve()),
        prediction_sha256=digest(args.prediction),
        metric_source_sha256=digest(root / 'AD-MLP/deps/stp3/stp3/planning_metrics.py'),
        metric_entry_sha256=digest(__file__),
        gt_sha256={name: digest(label_root / name) for name in (
            'stp3_traj_gt.pkl', 'stp3_occupancy.pkl', 'filter_token.pkl')},
        per_step={key: arr.tolist() for key, arr in scores.items()},
        prefix={}, endpoint={})
    for h in (1, 2, 3):
        k = 2 * h
        for mode in ('prefix', 'endpoint'):
            values = {key: float(v[:k].mean() if mode == 'prefix' else v[k-1])
                      for key, v in scores.items()}
            values['box_collision_percent'] = 100 * values['obj_box_col']
            report[mode][f'{h}s'] = values
    for mode in ('prefix', 'endpoint'):
        report[mode]['mean_L2'] = sum(report[mode][f'{h}s']['L2'] for h in (1, 2, 3))/3
    write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


def compare(args):
    import numpy as np
    left = read_pickle(args.left)
    right = read_pickle(args.right)
    if set(left) != set(right):
        raise ValueError('Different trajectory token sets')
    differences = [np.asarray(left[k]) - np.asarray(right[k]) for k in sorted(left)]
    report = dict(samples=len(left),
                  exactly_equal_tokens=sum(np.array_equal(left[k], right[k]) for k in left),
                  max_abs_difference=float(max(np.abs(d).max() for d in differences)),
                  mean_abs_difference=float(np.abs(np.stack(differences)).mean()),
                  left_sha256=digest(args.left), right_sha256=digest(args.right))
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('predict')
    p.add_argument('--source-root', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--baseline', action='store_true')
    p.add_argument('--limit', type=int, default=0)
    p.set_defaults(run=predict)
    p = sub.add_parser('metrics')
    p.add_argument('--metric-root', required=True)
    p.add_argument('--prediction', required=True)
    p.add_argument('--output', required=True)
    p.set_defaults(run=metrics)
    p = sub.add_parser('compare')
    p.add_argument('--left', required=True)
    p.add_argument('--right', required=True)
    p.add_argument('--output')
    p.set_defaults(run=compare)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
