"""Tiny CPU forward probes of V2 loss/decoder alignment; no gradients or model build.

These establish mechanisms, not their contribution to real validation metrics.
The production loss and decoder are called without modifying their source.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from mmdet3d.models.sparsedetectors.query_memory import sparse_soft_voxel_iou_loss
from mmdet3d.models.sparsedetectors.opus_head import OPUSHead


def probe():
    torch.set_num_threads(1)
    with torch.no_grad():
        semantic = torch.full((1, 1, 1, 17), -10.)
        semantic[..., 7] = 10.
        occupancy = torch.full((1, 1, 1, 1), 10.)
        gt = torch.full((1, 4, 4, 4), 17)
        gt[0, 1, 1, 1] = 7
        coordinates = [[.6, .6, .6], [.59, .6, .6], [.61, .6, .6],
                       [.401, .401, .401], [.799, .799, .799]]
        spatial = []
        for xyz in coordinates:
            points = torch.tensor(xyz).reshape(1, 1, 1, 3)/1.6
            kw = dict(pc_range=(0., 0., 0., 1.6, 1.6, 1.6),
                      voxel_size=(.4, .4, .4))
            current = sparse_soft_voxel_iou_loss(semantic, occupancy, points, gt, **kw)
            # Analytical alternative: shifting ONLY the splatting coordinate
            # by half a voxel is equivalent to a centre-indexed grid. Not used
            # for actual evaluation or any training/update in this script.
            centered = sparse_soft_voxel_iou_loss(
                semantic, occupancy, points - .2/1.6, gt, **kw)
            spatial.append(dict(metric=xyz, hard_voxel=(torch.tensor(xyz)/.4).floor().tolist(),
                                actual_loss=float(current), centre_indexed_probe_loss=float(centered)))
        assert all(r['hard_voxel'] == [1., 1., 1.] for r in spatial)
        assert spatial[1]['actual_loss'] < spatial[0]['actual_loss'] < spatial[2]['actual_loss']
        assert spatial[0]['centre_indexed_probe_loss'] < spatial[1]['centre_indexed_probe_loss']

        head = SimpleNamespace(pc_range=torch.tensor([0., 0., 0., 1.6, 1.6, 1.6]),
                               voxel_num=torch.tensor([4, 4, 4]),
                               voxel_size=torch.tensor([.4, .4, .4]), num_classes=17,
                               test_cfg=dict(score_thr=[.35]*15+[.25, .30], padding=False))
        points = torch.tensor([[[[.25, .25, .25], [.75, .75, .75]]]])
        outputs = []
        for cls, suppression in ((4, True), (15, True), (15, False), (16, True)):
            logits = torch.full((1, 1, 2, 17), -10.)
            logits[..., cls] = torch.logit(torch.tensor(.4))
            pred = dict(cls_scores=logits, refine_pts=points.clone())
            kwargs = {} if suppression else dict(thre1=None, thre2=None)
            decoded = OPUSHead.get_occ(head, pred, **kwargs)
            outputs.append(dict(top_class=cls, occupancy_probability=.4,
                                class_spread_suppression=suppression,
                                nonempty_voxels=int((decoded != 17).sum())))
        assert [r['nonempty_voxels'] for r in outputs] == [2, 0, 2, 0]
    return dict(soft_voxel_coordinates=spatial, actual_decoder=outputs,
                limitations='Synthetic forward mechanisms only; no estimate of real-data causal effect; no backward/optimizer/GPU use.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    result = probe()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
