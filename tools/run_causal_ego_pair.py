"""One protocol pass of bounded paired T1/Baseline-snapshot OOD diagnostics.

Original native decoder and frozen checkpoint. No training, parameter updates,
threshold changes, full-validation outputs, or legacy-cache substitution.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from diagnose_future_memory_semantics import confusion, summarize
from diagnose_future_memory_branches import sha256


def main():
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import wrap_fp16_model
    from mmdet.apis import set_random_seed
    from mmdet.utils import compat_cfg, setup_multi_processes
    from mmdet3d.datasets import build_dataset, build_dataloader
    from mmdet3d.models import build_model
    from mmdet3d.utils import patch_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--protocol', choices=['legacy', 'causal'], required=True)
    args = parser.parse_args()
    root = Path(args.root)
    prepared = json.loads((root / 'preparation.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())['manifest']
    if not 2 <= len(manifest) <= 256:
        raise ValueError('Only bounded audit manifests allowed')
    if (root / (args.protocol + '_confusions.npz')).exists():
        raise FileExistsError('Protocol already completed')
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    set_random_seed(0, deterministic=True)
    cfg = patch_config(compat_cfg(Config.fromfile(root / (args.protocol + '.py'))))
    setup_multi_processes(cfg)
    assert cfg.causal_experiment_stage == 'R1_ood_sensitivity_not_clean_training'
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.align_after_view_transfromation = True
    dataset = build_dataset(cfg.data.test)
    for m in manifest:
        assert dataset.data_infos[dataset.temp2nusc_map[m['index']]]['token'] == m['token']
    loader = build_dataloader(torch.utils.data.Subset(dataset, [m['index'] for m in manifest]),
        samples_per_gpu=1, workers_per_gpu=1, num_gpus=1, dist=False, shuffle=False, seed=0)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    assert sha256(prepared['checkpoint']) == prepared['checkpoint_sha256']
    checkpoint = torch.load(prepared['checkpoint'], map_location='cpu')['state_dict']
    model.load_state_dict(checkpoint, strict=True)
    baseline = torch.load(prepared['baseline_checkpoint'], map_location='cpu')['state_dict']
    assert all(k in checkpoint and torch.equal(v, checkpoint[k]) for k, v in baseline.items())
    del checkpoint, baseline
    model.requires_grad_(False)
    model.CLASSES = dataset.CLASSES
    model = model.cuda().eval()
    parallel = MMDataParallel(model, device_ids=[0]).eval()
    captured = []
    original = model.forward_backbone
    def capture(self, *a, **kw):
        out = original(*a, **kw)
        captured.append(out)
        return out
    model.forward_backbone = types.MethodType(capture, model)
    hist = np.zeros((len(manifest), 2, 4, 18, 18), dtype=np.int64)
    start = time.monotonic()
    with torch.no_grad():
        for n, data in enumerate(loader):
            captured.clear()
            result = parallel(return_loss=False, rescale=True, **data)
            assert len(captured) == 1
            out = captured.pop()
            raw = dataset.temp2nusc_map[manifest[n]['index']]
            for h, step in enumerate([0, 2, 4, 6]):
                info = dataset.data_infos[raw+step]
                with np.load(Path(info['occ_path']) / 'labels.npz') as f:
                    gt = f['semantics']
                predicted = result[f'semantic_occ_{step}s'][0]
                if step == 0:
                    base = predicted
                else:
                    base = model.pts_bbox_head.get_occ(dict(
                        cls_scores=out['baseline_forecast_semantics_list'][step-1],
                        refine_pts=out['baseline_forecast_points_list'][step-1]))[0].cpu().numpy()
                hist[n, 0, h] = confusion(gt, predicted)
                hist[n, 1, h] = confusion(gt, base)
            del result, out
            if (n+1) % 8 == 0 or n == 0:
                print(f'{args.protocol}: {n+1}/{len(manifest)}, {time.monotonic()-start:.1f}s', flush=True)
        assert all(p.grad is None for p in model.parameters())
    model.forward_backbone = original
    np.savez_compressed(root / (args.protocol + '_confusions.npz'), hist=hist)
    metrics = {name: [summarize(x) for x in hist[:, i].sum(0)]
               for i, name in enumerate(('t1', 'baseline_snapshot'))}
    report = dict(protocol=args.protocol, samples=len(manifest), metrics=metrics,
        elapsed_seconds=time.monotonic()-start, frozen_baseline_exact=True,
        model_sha256={p: sha256(p) for p in ['mmdet3d/models/sparsedetectors/query_memory.py',
            'mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py',
            'mmdet3d/datasets/causal_ego.py', __file__]},
        limitations='Old trained weights under changed input; not a clean-trained performance claim')
    (root / (args.protocol + '_report.json')).write_text(json.dumps(report, indent=2))
    for name, rows in metrics.items():
        print(args.protocol, name, 'IoU', [round(r['iou'],4) for r in rows],
              'mIoU', [round(r['miou'],4) for r in rows], flush=True)


if __name__ == '__main__':
    main()
