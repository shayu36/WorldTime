# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
from tqdm import tqdm
from collections import OrderedDict
# import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
import mmdet3d
# print(mmdet3d.__path__)
from mmdet3d.apis import single_gpu_test,  multi_gpu_test_temporal


from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

from mmdet.datasets import replace_ImageToTensor
from mmdet.apis import set_random_seed
from mmdet3d.utils import patch_config

if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint',help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--eval',
        type=str,
        default=['segm'],
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--load_interval',
        type=int,
        default=1,
        )
    parser.add_argument(
        '--dump_dir', help='directory where results will be saved')
    parser.add_argument(
        '--dump-only',
        action='store_true',
        help='write temporal predictions to dump_dir without gathering all '
             'predictions in host memory or running dataset evaluation.')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--no-aavt',
        action='store_true',
        help='Do not align after view transformer.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def _dump_query_memory_diagnostics(model, args, cfg):
    """Aggregate and dump STAC-QM diagnostics collected during eval."""
    try:
        base = model.module if hasattr(model, 'module') else model
        diags = getattr(base, 'query_memory_diagnostics', None)
        if not diags:
            return
        import json
        import numpy as np
        keys = ['has_candidate', 'support_conf', 'candidate_count',
                'topk_candidate_count', 'avg_distance', 'avg_age', 'avg_gate']
        stats = {}
        for key in keys:
            vals = []
            for d in diags:
                v = d.get(key)
                if isinstance(v, torch.Tensor):
                    vals.append(v.float().reshape(-1))
                elif isinstance(v, (list, tuple)) and v and isinstance(
                        v[0], torch.Tensor):
                    vals.extend([x.float().reshape(-1) for x in v])
                elif isinstance(v, (int, float, bool)):
                    vals.append(torch.tensor([float(v)]))
            if not vals:
                continue
            cat = torch.cat(vals).cpu().numpy()
            stats[key] = dict(
                mean=float(np.mean(cat)), std=float(np.std(cat)),
                min=float(np.min(cat)), max=float(np.max(cat)),
                count=int(cat.size))
        stats['num_samples_with_diag'] = len(diags)
        stats['attention_shape'] = diags[0].get('attention_shape') if diags else None
        out_path = args.out.replace('.pkl', '_memory_diag.json') \
            if args.out else 'work_dirs/memory_diag.json'
        with open(out_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f'=== STAC-QM diagnostics dumped to {out_path} ===')
        for key, s in stats.items():
            if isinstance(s, dict) and 'mean' in s:
                print(f'{key}: mean={s["mean"]:.4f} std={s["std"]:.4f} '
                      f'min={s["min"]:.4f} max={s["max"]:.4f} n={s["count"]}')
    except Exception as e:
        print(f'[WARN] failed to dump query memory diagnostics: {e}')


def main():
    args = parse_args()

    assert args.eval or args.dump_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--eval", "--dump_dir"')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    cfg = patch_config(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)


        
    # build the dataloader
    cfg.data.test.load_interval=args.load_interval  # debug用
    
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    
    
    # build the model and load checkpoint
    if not args.no_aavt:
        if '4D' in cfg.model.type:
            cfg.model.align_after_view_transfromation=True
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    
    if args.dump_dir is not None:
        os.makedirs(args.dump_dir, exist_ok=True)

    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        # outputs = multi_gpu_test(model, data_loader, args.dump_dir,
        #                         args.gpu_collect
        #                         )
        outputs = multi_gpu_test_temporal(
            model, data_loader, args.dump_dir, gpu_collect=args.gpu_collect,
            dump_only=args.dump_only)




    if args.dump_only:
        return

    rank, _ = get_dist_info()
    if rank == 0:
        # dump STAC-QM diagnostics if collected
        _dump_query_memory_diagnostics(model, args, cfg)
        kwargs = {} if args.eval_options is None else args.eval_optionsz
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
