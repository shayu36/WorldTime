"""Focused checks for the optional centre-aligned sparse SoftVoxel loss."""
import torch

from mmdet3d.models.sparsedetectors.query_memory import sparse_soft_voxel_iou_loss


def _loss(xyz, gt=None, **extra):
    sem = torch.full((1, 1, 1, 17), -10.)
    sem[..., 7] = 10.
    occ = torch.full((1, 1, 1, 1), 10.)
    pts = torch.as_tensor(xyz, dtype=torch.float32).reshape(1, 1, 1, 3)/1.6
    if gt is None:
        gt = torch.full((1, 4, 4, 4), 17)
        gt[0, 1, 1, 1] = 7
    return sparse_soft_voxel_iou_loss(
        sem, occ, pts, gt,
        pc_range=(0., 0., 0., 1.6, 1.6, 1.6),
        voxel_size=(.4, .4, .4),
        voxel_center_offset=.5,
        renormalize_valid_corners=True, **extra)


def test_centre_is_optimal_and_neighbours_symmetric():
    center = _loss((.6, .6, .6)).item()
    left = _loss((.59, .6, .6)).item()
    right = _loss((.61, .6, .6)).item()
    assert center < left
    assert center < right
    assert abs(left - right) < 1e-6


def test_boundary_stays_finite_and_out_of_bounds_safe():
    empty = torch.full((1, 4, 4, 4), 17)
    for xyz in ((0., 0., 0.), (1.599, 1.599, 1.599),
                (-1., -1., -1.), (2., 2., 2.)):
        assert torch.isfinite(_loss(xyz, gt=empty))
    assert _loss((-.01, .6, .6)).item() == _loss((-1., -1., -1.)).item()
    edge_gt = torch.full((1, 4, 4, 4), 17)
    edge_gt[0, 0, 0, 0] = 7
    assert abs(_loss((0., 0., 0.), gt=edge_gt).item() -
               _loss((.2, .2, .2), gt=edge_gt).item()) < 1e-6


def test_centre_loss_has_position_gradient():
    sem = torch.full((1, 1, 1, 17), -10.)
    sem[..., 7] = 10.
    occ = torch.full((1, 1, 1, 1), 10.)
    gt = torch.full((1, 4, 4, 4), 17)
    gt[0, 1, 1, 1] = 7
    point_metric = torch.tensor([[[[.61, .6, .6]]]], requires_grad=True)
    pts = point_metric/1.6
    loss = sparse_soft_voxel_iou_loss(
        sem, occ, pts, gt,
        pc_range=(0., 0., 0., 1.6, 1.6, 1.6),
        voxel_size=(.4, .4, .4),
        voxel_center_offset=.5, renormalize_valid_corners=True)
    loss.backward()
    assert point_metric.grad is not None
    assert torch.isfinite(point_metric.grad).all()
    assert point_metric.grad.abs().sum() > 0
    assert point_metric.grad[0, 0, 0, 0] > 0  # descent moves toward .6
