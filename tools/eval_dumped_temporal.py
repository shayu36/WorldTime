"""Evaluate temporal occupancy predictions dumped one frame per .npy file.

The normal distributed test path keeps every prediction in a Python list and
can exceed host memory on the nuScenes validation split.  This utility opens
the dumped arrays as read-only memmaps and lets the dataset evaluator consume
one sample at a time.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from mmcv import Config

from mmdet3d.datasets import build_dataset
from mmdet3d.utils import patch_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('dump_dir')
    parser.add_argument('--output', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = patch_config(Config.fromfile(args.config))
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)

    dump_root = Path(args.dump_dir)
    predictions = []
    missing = []
    for dataset_index, nusc_index in enumerate(dataset.temp2nusc_map):
        info = dataset.data_infos[nusc_index]
        path = dump_root / info['scene_name'] / f"{info['token']}.npy"
        if not path.is_file():
            missing.append((dataset_index, str(path)))
            continue
        dumped = np.load(path, mmap_mode='r')
        if dumped.ndim != 5 or dumped.shape[0] != 1:
            raise ValueError(
                f'unexpected dump shape at {path}: {dumped.shape}; '
                'expected [1,4,H,W,D]')
        # The dump writer stores the one-element result list.  The dataset
        # evaluator expects the underlying [4,H,W,D] prediction per sample.
        predictions.append(dumped[0])

    if missing:
        preview = '\n'.join(path for _, path in missing[:10])
        raise FileNotFoundError(
            f'{len(missing)} predictions are missing from {dump_root}; '
            f'first missing paths:\n{preview}')
    if len(predictions) != len(dataset):
        raise RuntimeError(
            f'prediction count {len(predictions)} != dataset length '
            f'{len(dataset)}')

    # NuScenesDatasetOccpancy4DTraj.evaluate unwraps the outer result group.
    metrics = dataset.evaluate([predictions], metric=['segm'])
    print(json.dumps(metrics, indent=2, default=str))
    if args.output:
        with open(args.output, 'w') as handle:
            json.dump(metrics, handle, indent=2, default=str)


if __name__ == '__main__':
    main()
