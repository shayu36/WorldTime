# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info


def _parse_sparseworld_result(result):
    if not isinstance(result, dict):
        raise TypeError(
            'SparseWorld evaluation expects a dictionary result, got '
            f'{type(result).__name__}')

    pred_traj = None
    if result.get('semantic_occ_4s', None) is not None:
        occ = np.stack([
            result['semantic_occ_0s'][0],
            result['semantic_occ_2s'][0],
            result['semantic_occ_4s'][0],
            result['semantic_occ_6s'][0],
        ], axis=0).astype(np.uint8)
        if 'pred_traj' not in result:
            raise KeyError(
                'SparseWorld temporal result is missing "pred_traj"')
        pred_traj = result['pred_traj'].cpu()
    elif result.get('semantic_occ_0s', None) is not None:
        occ = np.stack([
            result['semantic_occ_0s'][0],
        ], axis=0).astype(np.uint8)
    elif result.get('semantic_occ', None) is not None:
        occ = result['semantic_occ']
    else:
        raise KeyError(
            'SparseWorld result has no supported occupancy prediction key')
    return occ, pred_traj


def _interleave_result_parts(part_list, size):
    ordered_results = []
    for result_group in zip(*part_list):
        ordered_results.extend(result_group)
    return ordered_results[:size]


def _collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    if tmpdir is None:
        max_len = 512
        dir_tensor = torch.full(
            (max_len,), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir_tensor = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir_tensor)] = tmpdir_tensor
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)

    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    if rank != 0:
        return None

    part_list = [
        mmcv.load(osp.join(tmpdir, f'part_{part_rank}.pkl'))
        for part_rank in range(world_size)
    ]
    ordered_results = _interleave_result_parts(part_list, size)
    shutil.rmtree(tmpdir)
    return ordered_results


def _collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8,
        device='cuda')
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)

    shape_max = torch.stack(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    dist.all_gather(part_recv_list, part_send)

    if rank != 0:
        return None
    part_list = [
        pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
        for recv, shape in zip(part_recv_list, shape_list)
    ]
    return _interleave_result_parts(part_list, size)


def _format_collected_results(ordered_results):
    has_traj = [pred_traj is not None
                for _, pred_traj in ordered_results]
    if any(has_traj) and not all(has_traj):
        raise RuntimeError(
            'SparseWorld evaluation received inconsistent trajectory outputs '
            'across samples')

    ordered_occ = [occ for occ, _ in ordered_results]
    if has_traj and all(has_traj):
        ordered_traj = [pred_traj for _, pred_traj in ordered_results]
        return [ordered_occ, ordered_traj]
    return [ordered_occ]


def single_gpu_test(model, data_loader):
    model.eval()
    result_part = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        result_part.append(_parse_sparseworld_result(result))
        prog_bar.update()
    return _format_collected_results(result_part)


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    model.eval()
    result_part = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)

    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        result_part.append(_parse_sparseworld_result(result))
        if rank == 0:
            for _ in range(world_size):
                prog_bar.update()

    if gpu_collect:
        ordered_results = _collect_results_gpu(result_part, len(dataset))
    else:
        ordered_results = _collect_results_cpu(
            result_part, len(dataset), tmpdir)
    if rank != 0:
        return None
    return _format_collected_results(ordered_results)
