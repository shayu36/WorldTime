"""Read-only, bounded source/consumer audit of pre-repair T1.

Writes only a new JSON report. No training steps, gradients, cache writes or
checkpoint changes. Cache replay stays in RAM; inability to reproduce old
cache is reported, NOT silently interpreted as proof of its origin.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SNAPSHOT = 'c562513beb0ba2f919c62d524f468d58ddb3bf85'
CONFIG = 'configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1.py'
CHECKPOINT = 'work_dirs/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1-from-center2/epoch_3.pth'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def historical(path):
    return subprocess.check_output(['git', 'show', SNAPSHOT+':'+path], cwd=ROOT).decode()


def original_method(path, clsname, name, namespace):
    tree = ast.parse(historical(path))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == clsname)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    env = dict(namespace)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<historical method>', 'exec'), env)
    return env[name]


def tensor_summary(x):
    import torch
    x = x.detach().cpu().contiguous()
    return dict(shape=list(x.shape), sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                abs_max=float(x.abs().max()) if x.numel() else 0., finite=bool(torch.isfinite(x).all()))


def compare(a, b):
    import torch
    if a.shape != b.shape:
        return dict(shape_equal=False, old_shape=list(a.shape), new_shape=list(b.shape))
    a, b = a.detach().cpu(), b.detach().cpu()
    return dict(shape_equal=True, exact=bool(torch.equal(a, b)),
                max_abs_diff=float((a.float()-b.float()).abs().max()) if a.numel() else 0.)


def scalar_fit(value, reference):
    import numpy as np
    value, reference = np.asarray(value), np.asarray(reference)
    scale = float(value @ reference / (reference @ reference))
    return dict(scale=scale, residual=float(np.max(np.abs(value-scale*reference))))


def strip_memory(pipeline):
    result = []
    for tr in copy.deepcopy(pipeline):
        if tr['type'] == 'LoadQueryMemoryFromFiles': continue
        if tr['type'] == 'MultiScaleFlipAug3D':
            result.extend(strip_memory(tr['transforms']))
            continue
        if tr['type'] == 'Collect4D':
            tr['keys'] = [k for k in tr['keys'] if not k.startswith('memory_')]
        result.append(tr)
    return result


def main():
    import numpy as np
    import torch
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmdet.apis import set_random_seed
    from mmdet3d.datasets import build_dataset
    import mmdet3d.datasets.nuscenes_dataset_occ_trajectory as dm
    from tools.audit_future_memory_ego_provenance import lidar_global_pose

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--index', type=int, default=20)
    parser.add_argument('--splits', nargs='+', default=['train', 'test'],
                        choices=['train', 'test'])
    parser.add_argument('--source-eval-images', action='store_true',
                        help='Probe deterministic source image augmentation; '
                             'does not change the consumer training pipeline')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): parser.error('Refusing to overwrite report')
    torch.set_num_threads(1)
    cfg = Config.fromfile(str(ROOT / CONFIG))
    # Use the EXACT dataset getter used before the ongoing repair, in RAM.
    dm.NuScenesDatasetOccpancy4DTraj.get_data_info = original_method(
        'mmdet3d/datasets/nuscenes_dataset_occ_trajectory.py',
        'NuScenesDatasetOccpancy4DTraj', 'get_data_info', dm.__dict__)
    report = dict(snapshot=SNAPSHOT, checkpoint=CHECKPOINT, checkpoint_sha256=sha(ROOT/CHECKPOINT),
        scope='bounded real-sample forward probes; no backward, optimizer, training run, cache writes or metrics evaluation',
        external_ego_sha256=sha(ROOT/'admlp/stp3_val/data_nuscene.pkl'), samples=[])
    model = None
    if args.gpu:
        from mmdet3d.models import build_model
        import mmdet3d.models.sparsedetectors.query_memory as qm
        qm.FutureMemoryAdapterV2._reshape_memory = original_method(
            'mmdet3d/models/sparsedetectors/query_memory.py',
            'FutureMemoryAdapterV2', '_reshape_memory', qm.__dict__)
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg')).cuda()
        state = torch.load(ROOT/CHECKPOINT, map_location='cpu')['state_dict']
        model.load_state_dict(state, strict=True)
        baseline = torch.load(ROOT/'ckpts/epoch_56.pth', map_location='cpu')['state_dict']
        assert all(torch.equal(v, state[k]) for k, v in baseline.items())
        report['baseline_common_tensors_exact'] = len(baseline)
        del state, baseline
        model._freeze_memory_finetune_temporal_state()
        model.eval()
        prefix = types.ModuleType('historical_precompute')
        exec(compile(historical('tools/query_memory/precompute_query_memory.py'),
                     '<historical precompute>', 'exec'), prefix.__dict__)

    for split in args.splits:
        ds = build_dataset(cfg.data[split])
        public = args.index
        raw = ds.temp2nusc_map[public]
        info = ds.data_infos[raw]
        lookup = {r['token']: i for i, r in enumerate(ds.data_infos)}
        hist = ds._build_query_memory_history_infos(raw)[0]
        source_raw = lookup[hist['sample_idx']]
        source_public = ds.temp2nusc_map.index(source_raw)
        cache_path = Path(ds.query_memory_cache_root) / hist['scene_id'] / (hist['sample_idx']+'.pt')
        stored = torch.load(cache_path, map_location='cpu')
        cache_sha = sha(cache_path)
        sample = dict(split=split, public_index=public, raw_index=raw, token=info['token'],
            scene=info['scene_name'], timestamp_us=info['timestamp'],
            cache_path=str(cache_path), cache_sha256=cache_sha,
            cache_declared_source_config=stored['source_config'],
            cache_declared_checkpoint=stored['source_checkpoint'], records={})
        # Independent metadata poses establish relative times, without relying
        # on the name 'gt' or the position of the public dataset index.
        for label, idx in [('current', raw), ('memory_source', source_raw)]:
            row = ds.data_infos[idx]
            poses = np.array([lidar_global_pose(r) for r in ds.data_infos[idx-4:idx+7]])
            relative = np.linalg.inv(poses[4])[None] @ poses
            traj = np.stack([relative[:, 0, 3], relative[:, 1, 3],
                             np.arctan2(relative[:, 1, 0], relative[:, 0, 0])], -1)
            record = ds.ad_info[row['token']]
            gt = np.array(record['gt'])
            original_input = ds.get_data_info(idx)['temporal_ego_states'][0]
            expected = np.concatenate([np.asarray(record[k]).reshape(-1)
                                       for k in sorted(record) if k != 'gt']).astype('float32')
            assert torch.equal(original_input.flatten(), torch.from_numpy(expected))
            sample['records'][label] = dict(token=row['token'],
                relative_to_prediction_s=(int(row['timestamp'])-int(info['timestamp']))/1e6,
                trajectory_times_relative_to_prediction_s=[(int(ds.data_infos[idx+j]['timestamp'])-int(info['timestamp']))/1e6 for j in range(7)],
                actual_input_equals_same_token_record=True, input_21=expected.tolist(),
                independently_rebuilt_gt_max_error=float(np.abs(traj[4:]-gt).max()),
                v_fit_future=scalar_fit(record['v'], gt[3]-gt[1]),
                a_fit_future=scalar_fit(record['a'], (gt[6]-gt[1])/2.5),
                v_fit_past_1s=scalar_fit(record['v'], traj[4]-traj[2]),
                gt=gt.tolist(), x=np.asarray(record['x']).tolist())

        if model is not None:
            model.eval()
            # Native source split transforms (including train's random image
            # augmentation): unknown historical RNG is reported, not inferred.
            source_cfg = copy.deepcopy(cfg.data[split])
            source_cfg.pipeline = strip_memory(source_cfg.pipeline)
            if args.source_eval_images:
                for transform in source_cfg.pipeline:
                    if transform['type'] == 'RandomTransformImage':
                        transform['training'] = False
            source_ds = build_dataset(source_cfg)
            set_random_seed(17)
            source_batch = scatter(collate([source_ds[source_public]], samples_per_gpu=1), [0])[0]
            record = ds.ad_info[hist['sample_idx']]
            gt = np.asarray(record['gt'])
            scale = scalar_fit(record['a'], (gt[6]-gt[1])/2.5)['scale']
            # Only a coordinate of source g[6] that is AFTER prediction t is
            # changed. This reconstructs the observed algebra, NOT an assertion
            # that the unpublished original generator code was recovered.
            assert ds.data_infos[source_raw+6]['timestamp'] > info['timestamp']
            altered_batch = copy.deepcopy(source_batch)
            altered_batch['temporal_ego_states'][0][..., 0] += scale/2.5
            source_args = types.SimpleNamespace(config=stored['source_config'],
                checkpoint=stored['source_checkpoint'], max_queries_per_frame=256,
                write_threshold=.35, min_reliability=0., spatial_cell_size=4.,
                max_per_spatial_cell=16, max_per_class=64)
            def run_prefix(batch):
                # Prevent mutable image meta from affecting the repeated run.
                batch = copy.deepcopy(batch)
                outs = prefix.run_sparseworld_rap_prefix(model, batch['img'], batch['img_metas'], batch)
                cache = prefix.build_cache_record(model, outs, batch['img_metas'][0], source_args)
                return cache
            with torch.no_grad():
                regenerated = run_prefix(source_batch)
                repeat = run_prefix(source_batch)
                changed = run_prefix(altered_batch)
            keys = ['query_feat', 'query_points_metric', 'query_semantic_distribution', 'query_reliability']
            sample['prefix'] = dict(
                historical_replay={k: compare(stored[k], regenerated[k]) for k in keys},
                deterministic_repeat={k: compare(regenerated[k], repeat[k]) for k in keys},
                source_future_endpoint_perturbation={k: compare(regenerated[k], changed[k]) for k in keys},
                intervention='g_source[6].x += 1m; only its fitted a[0] input is updated; all source images/past fixed',
                source_eval_images=args.source_eval_images,
                original_cache_generator_rng_known=False)
            print(json.dumps({split + '_prefix': sample['prefix']}, indent=2), flush=True)
            del source_batch, altered_batch, source_ds, repeat
            torch.cuda.empty_cache()

            if split == 'train':
                # Actual training forward, including original losses, but no
                # gradient graph or updates. Same images/GT/current ego/RNG.
                set_random_seed(17)
                batch = scatter(collate([ds[public]], samples_per_gpu=1), [0])[0]
                assert torch.equal(batch['temporal_ego_states'][0].cpu().flatten(),
                    torch.tensor(sample['records']['current']['input_21']))
                model.train()
                captured, gates, plan_inputs = {}, {}, []
                original_forward = model.forward_backbone
                def capture_forward(self, *a, **kw):
                    out = original_forward(*a, **kw)
                    for key in ('cls_score', 'refine_pts', 'forecast_semantics_list',
                                'forecast_points_list', 'baseline_forecast_semantics_list',
                                'baseline_forecast_points_list'):
                        value = out[key]
                        captured[key] = ([tensor_summary(v) for v in value]
                                         if isinstance(value, list) else tensor_summary(value))
                    return out
                model.forward_backbone = types.MethodType(capture_forward, model)
                def on_adapter(module, inputs, result):
                    h = int(inputs[4])
                    gates[h] = {key: tensor_summary(result[key]) for key in ('context', 'gate', 'delta_s', 'delta_o', 'delta_p') if key in result}
                def on_plan(module, inputs):
                    plan_inputs.append(inputs[0].detach().cpu().flatten().tolist())
                handles = [model.future_memory_adapter_v2.register_forward_hook(on_adapter),
                           model.plan_head.register_forward_pre_hook(on_plan)]
                # New cache is only an in-memory intervention. Do not touch
                # any source file, original batch, or the other history slots.
                loader = next(t for t in ds.pipeline.transforms if type(t).__name__ == 'LoadQueryMemoryFromFiles')
                slot = int(hist['slot_index'])
                def replace_slot(data, cache):
                    alternative = copy.deepcopy(data)
                    feat, points, conf, rel, labels, sem, valid = loader._select_queries(cache)
                    mapping = dict(memory_query_feat=feat, memory_points_metric=points,
                                   memory_conf=conf, memory_reliability=rel, memory_label=labels,
                                   memory_semantic_distribution=sem, memory_valid=valid)
                    for key, values in mapping.items():
                        dest = alternative[key][0, slot]
                        dest.zero_()
                        dest[:len(values)] = values.to(dest)
                    return alternative
                alternative = replace_slot(batch, changed)
                def train_forward(data):
                    set_random_seed(17)
                    captured.clear(); gates.clear(); plan_inputs.clear()
                    with torch.no_grad(): losses = model(return_loss=True, **copy.deepcopy(data))
                    return dict(predictions=copy.deepcopy(captured), adapter=copy.deepcopy(gates),
                        plan_inputs=copy.deepcopy(plan_inputs),
                        losses={k: float(v.detach().float().mean()) if torch.is_tensor(v)
                                else [float(x.detach().float().mean()) for x in v]
                                for k,v in losses.items()})
                native = train_forward(batch)
                repeat = train_forward(batch)
                altered = train_forward(alternative)
                # This pair isolates the source-field intervention even if
                # the historical train augmentation seed is unavailable.
                reconstructed = train_forward(replace_slot(batch, regenerated))
                ego_alternative = copy.deepcopy(batch)
                current_scale = sample['records']['current']['a_fit_future']['scale']
                ego_alternative['temporal_ego_states'][0][..., 0] += current_scale / 2.5
                ego_altered = train_forward(ego_alternative)
                sample['training_forward'] = dict(native=native, modified_memory=altered,
                    reconstructed_memory=reconstructed,
                    modified_current_ego=ego_altered,
                    native_repeat_exact=native==repeat,
                    native_prediction_repeat_exact=native['predictions']==repeat['predictions'],
                    current_ego_kept_exact=native['plan_inputs']==altered['plan_inputs'],
                    baseline_kept_exact=all(native['predictions'][k]==altered['predictions'][k]
                        for k in ('cls_score','refine_pts','baseline_forecast_semantics_list','baseline_forecast_points_list')),
                    interpretation='If historical replay differs, this is a source-path and real-Memory-consumer probe, not proof of old-cache provenance')
                for h in handles: h.remove()
                model.forward_backbone = original_forward
                assert all(p.grad is None for p in model.parameters())
                del batch, alternative, ego_alternative
            else:
                # Native evaluation tensor path on a real validation sample,
                # not a fabricated Adapter input. We stop at prediction and
                # never accumulate dataset metrics.
                import queue
                set_random_seed(17)
                augmented = scatter(collate([ds[public]], samples_per_gpu=1), [0])[0]
                batch = {key: value[0] for key, value in augmented.items()}
                del augmented
                loader = next(t for t in ds.pipeline.transforms if type(t).__name__ == 'LoadQueryMemoryFromFiles')
                alternative = copy.deepcopy(batch)
                feat, points, conf, rel, labels, sem, valid = loader._select_queries(changed)
                mapping = dict(memory_query_feat=feat, memory_points_metric=points,
                               memory_conf=conf, memory_reliability=rel, memory_label=labels,
                               memory_semantic_distribution=sem, memory_valid=valid)
                for key, values in mapping.items():
                    dest = alternative[key][0, int(hist['slot_index'])]
                    dest.zero_()
                    dest[:len(values)] = values.to(dest)
                def eval_forward(data):
                    set_random_seed(17)
                    model.eval()
                    model.memory = {}
                    model.queue = queue.Queue()
                    data = copy.deepcopy(data)
                    img, metas = data.pop('img'), data.pop('img_metas')
                    with torch.no_grad():
                        out = model.forward_backbone(img, metas, **data)
                        summary = {}
                        for key in ('cls_score', 'refine_pts', 'forecast_semantics_list',
                                    'forecast_points_list', 'baseline_forecast_semantics_list',
                                    'baseline_forecast_points_list'):
                            value = out[key]
                            summary[key] = ([tensor_summary(v) for v in value]
                                            if isinstance(value, list) else tensor_summary(value))
                        labels = [model.pts_bbox_head.get_occ(dict(
                            cls_scores=out['cls_score'], refine_pts=out['refine_pts']))[0]]
                        for step in (1, 3, 5):
                            labels.append(model.pts_bbox_head.get_occ(dict(
                                cls_scores=out['forecast_semantics_list'][step],
                                refine_pts=out['forecast_points_list'][step]))[0])
                        summary['decoded_0_1_2_3s'] = [tensor_summary(x) for x in labels]
                    return summary
                native = eval_forward(batch)
                repeat = eval_forward(batch)
                altered = eval_forward(alternative)
                sample['evaluation_forward'] = dict(native=native, modified_memory=altered,
                    native_repeat_exact=native==repeat,
                    baseline_kept_exact=all(native[k]==altered[k] for k in (
                        'cls_score', 'refine_pts', 'baseline_forecast_semantics_list',
                        'baseline_forecast_points_list')),
                    current_ego_kept_exact=bool(torch.equal(batch['temporal_ego_states'][0],
                                                          alternative['temporal_ego_states'][0])),
                    interpretation='Causal source-field probe of an actual old validation cache ONLY if historical_replay is exact')
                assert all(p.grad is None for p in model.parameters())
                del batch, alternative
            del regenerated, changed
        assert sha(cache_path) == cache_sha
        report['samples'].append(sample)
        print(json.dumps({split: {k:v for k,v in sample.items()
                         if k not in ('training_forward', 'evaluation_forward')}}, indent=2), flush=True)
        del ds
    assert sha(ROOT/CHECKPOINT) == report['checkpoint_sha256']
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f: json.dump(report, f, indent=2)
    print('REPORT:', str(output), flush=True)


if __name__ == '__main__': main()
