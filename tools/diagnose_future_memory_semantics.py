"""CPU-only, bounded analysis of existing temporal dumps (no model inference).

Uses the real dataset index mapping and the evaluator's unmasked 18-class
confusion matrix. Reports class omission versus class confusion separately.
This is an observational comparison, not a branch ablation or causal proof.
"""
import argparse
import json
from pathlib import Path

import numpy as np


CLASSES = ['others', 'barrier', 'bicycle', 'bus', 'car',
           'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
           'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
           'terrain', 'manmade', 'vegetation', 'free']


def confusion(gt, pred):
    gt, pred = np.asarray(gt).reshape(-1), np.asarray(pred).reshape(-1)
    valid = (gt >= 0) & (gt < 18)
    if np.any((pred[valid] < 0) | (pred[valid] >= 18)):
        raise ValueError('Prediction label outside [0,17]')
    return np.bincount(18 * gt[valid].astype(np.int64) + pred[valid],
                       minlength=324).reshape(18, 18)


def summarize(hist):
    gt, pred, tp = hist.sum(1), hist.sum(0), np.diag(hist)
    union = gt + pred - tp
    iou = np.divide(tp, union, out=np.zeros(18, dtype=float), where=union > 0)
    otp, ofp, ofn = hist[:17, :17].sum(), hist[17, :17].sum(), hist[:17, 17].sum()
    occupied_gt = gt[:17].sum()
    rows = {}
    for c, name in enumerate(CLASSES[:17]):
        rows[name] = dict(gt=int(gt[c]), predicted=int(pred[c]),
                         tp=int(tp[c]), fp=int(pred[c]-tp[c]),
                         fn=int(gt[c]-tp[c]), iou=float(iou[c]*100),
                         missed_as_free=int(hist[c, 17]),
                         confused_as_other_occupied=int(hist[c, :17].sum()-tp[c]),
                         predicted_on_free=int(hist[17, c]))
    return dict(iou=100*float(otp/max(otp+ofp+ofn, 1)),
                miou=100*float(iou[:17][union[:17] > 0].mean()),
                precision=100*float(otp/max(otp+ofp, 1)),
                recall=100*float(otp/max(otp+ofn, 1)),
                semantic_accuracy_on_common_occupied=100*float(tp[:17].sum()/max(otp, 1)),
                occupied_gt=int(occupied_gt), occupied_predicted=int(pred[:17].sum()),
                classes=rows, confusion=hist.tolist())


def transitions(gt, before, after):
    """Disjoint GT-occupied changes; not an intervention on model branches."""
    occupied = gt != 17
    old_free, new_free = before == 17, after == 17
    old_right, new_right = before == gt, after == gt
    common = occupied & ~old_free & ~new_free
    return dict(
        recovered_correct=int((occupied & old_free & new_right).sum()),
        recovered_wrong_class=int((occupied & old_free & ~new_free & ~new_right).sum()),
        lost_correct=int((occupied & old_right & new_free).sum()),
        lost_wrong_class=int((occupied & ~old_free & ~old_right & new_free).sum()),
        correct_to_wrong_class=int((common & old_right & ~new_right).sum()),
        wrong_to_correct_class=int((common & ~old_right & new_right).sum()),
        false_positive_added=int((~occupied & old_free & ~new_free).sum()),
        false_positive_removed=int((~occupied & ~old_free & new_free).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--dump', action='append', required=True, help='NAME=directory; first is reference')
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error('--samples must be positive (bounded analysis only)')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    import torch
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.utils import patch_config
    torch.set_num_threads(1)
    cfg = patch_config(Config.fromfile(args.config))
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    roots = dict(item.split('=', 1) for item in args.dump)
    if len(roots) != len(args.dump):
        raise ValueError('Dump names must be unique')
    ref = next(iter(roots))
    indices = np.unique(np.linspace(0, len(dataset)-1,
                                   min(args.samples, len(dataset)), dtype=int))
    hist = {name: np.zeros((4, 18, 18), dtype=np.int64) for name in roots}
    changes = {name: [{} for _ in range(4)] for name in roots if name != ref}
    manifest = []
    for count, index in enumerate(indices):
        ni = dataset.temp2nusc_map[index]
        info = dataset.data_infos[ni]
        relative = Path(info['scene_name']) / (info['token'] + '.npy')
        arrays = {name: np.load(Path(root)/relative, mmap_mode='r')
                  for name, root in roots.items()}
        for name, array in arrays.items():
            if array.shape != (1, 4, 200, 200, 16):
                raise ValueError(f'{name}: unexpected dump shape {array.shape}')
        for h in range(4):
            future_info = dataset.data_infos[ni+2*h]
            assert future_info['scene_name'] == info['scene_name']
            with np.load(Path(future_info['occ_path'])/'labels.npz') as labels:
                gt = labels['semantics']
            if np.any((gt < 0) | (gt >= 18)):
                raise ValueError('Unexpected GT labels: inspect evaluator masking first')
            before = arrays[ref][0, h]
            for name, array in arrays.items():
                pred = array[0, h]
                hist[name][h] += confusion(gt, pred)
                if name != ref:
                    for key, value in transitions(gt, before, pred).items():
                        changes[name][h][key] = changes[name][h].get(key, 0) + value
        manifest.append(dict(index=int(index), scene=info['scene_name'], token=info['token']))
        if (count+1) % 32 == 0:
            print(f'CPU dump diagnostic: {count+1}/{len(indices)}', flush=True)
    result = dict(config=args.config, dumps=roots, reference=ref,
                  samples=len(indices), sampling='uniform dataset indices, deterministic, no class selection',
                  scenes=len({m['scene'] for m in manifest}), manifest=manifest,
                  metrics={name: [summarize(h) for h in hs] for name, hs in hist.items()},
                  transitions_from_reference=changes,
                  limitations='Saved output comparison only; different checkpoints, no branch causal attribution; subset not full validation.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
    for name, hs in result['metrics'].items():
        for h, m in enumerate(hs):
            print(name, h, {k: round(m[k], 4) for k in
                            ('iou', 'miou', 'precision', 'recall', 'semantic_accuracy_on_common_occupied')})
    print(output)


if __name__ == '__main__':
    main()
