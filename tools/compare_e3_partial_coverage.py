"""Paired validation comparison of original E3 and fixed-coverage E3.

Uses identical ordered validation frames and resamples scenes, preserving
within-scene temporal correlation. Deltas are partial minus original E3.
"""
import argparse
from pathlib import Path

import numpy as np
from diagnose_future_memory_semantics import summarize
from recallocc_modules_common import digest, read_json, write_json


ORIGINAL = 'E3_current_only_task_time'
PARTIAL = 'E3_current_only_70pct'


def compare(original_root, partial_root, seed=0, bootstrap=1000):
    roots = [Path(original_root), Path(partial_root)]
    studies = [read_json(root/'study.json') for root in roots]
    runs = [read_json(root/name/f'seed_{seed}'/'run.json')
            for root,name in zip(roots,(ORIGINAL,PARTIAL))]
    required = ('base_sha256','checkpoint_selection','eval_manifest_sha256')
    for key in required:
        if studies[0][key] != studies[1][key]:
            raise ValueError(f'Study mismatch: {key}')
    for key in ('epochs','effective_batch_size','world_size','smoke_steps'):
        if studies[0]['spec'][key] != studies[1]['spec'][key]:
            raise ValueError(f'Training budget mismatch: {key}')
    if studies[0]['spec']['smoke_steps']:
        raise ValueError('Paired result requires full training and validation')
    manifests = [read_json(root/'eval_manifest.json') for root in roots]
    if manifests[0] != manifests[1] or digest(roots[0]/'eval_manifest.json') != studies[0]['eval_manifest_sha256']:
        raise ValueError('Ordered validation frames differ')
    evaluations = [read_json(run['evaluation']) for run in runs]
    arrays = []
    for root, run, ev, name in zip(roots,runs,evaluations,(ORIGINAL,PARTIAL)):
        if run['experiment'] != name or run['seed'] != seed or ev['model'] != name or ev['seed'] != seed:
            raise ValueError('Experiment or seed identity mismatch')
        if ev['manifest_sha256'] != studies[0]['eval_manifest_sha256'] or not ev['full_validation']:
            raise ValueError('Evaluation protocol mismatch')
        if ev['checkpoint_sha256'] != digest(run['selected_checkpoint']):
            raise ValueError('Checkpoint mismatch')
        with np.load(Path(run['evaluation']).with_name('confusions.npz')) as z:
            arrays.append(z['hist'].copy())
    if arrays[0].shape != arrays[1].shape or arrays[0].shape != (len(manifests[0]),4,18,18):
        raise ValueError('Confusion tensor shape mismatch')
    def values(hist):
        per_horizon = [summarize(hist[:,h].sum(0)) for h in (1,2,3)]
        return {key: [float(row[key]) for row in per_horizon] for key in ('miou','iou')}
    base, partial = map(values, arrays)
    difference = {key: list(np.asarray(partial[key])-np.asarray(base[key])) for key in base}
    scenes = np.asarray([row['scene'] for row in manifests[0]])
    unique, inverse = np.unique(scenes,return_inverse=True)
    scene_hist = [np.zeros((len(unique),4,18,18),dtype=np.int64) for _ in arrays]
    for target,source in zip(scene_hist,arrays):
        np.add.at(target,inverse,source)
    rng = np.random.default_rng(20261104)
    draws = {key: np.empty((bootstrap,3),dtype=float) for key in ('miou','iou')}
    for b in range(bootstrap):
        indices = rng.integers(0,len(unique),len(unique))
        sampled = [x[indices].sum(0) for x in scene_hist]
        for h in range(3):
            before,after = (summarize(x[h+1]) for x in sampled)
            for key in draws:
                draws[key][b,h] = after[key]-before[key]
    ci = {key: np.quantile(value,[.025,.975],axis=0).tolist() for key,value in draws.items()}
    report = dict(comparison='partial_minus_original_E3',seed=seed,
                  original_run=str((roots[0]/ORIGINAL/f'seed_{seed}'/'run.json').resolve()),
                  partial_run=str((roots[1]/PARTIAL/f'seed_{seed}'/'run.json').resolve()),
                  frames=len(scenes),scenes=len(unique),bootstrap_replicates=bootstrap,
                  units='percentage_points',horizons_seconds=[1,2,3],
                  original=base,partial=partial,difference=difference,
                  paired_scene_bootstrap_95pct_ci=ci,
                  interpretation='Fixed marginal point coverage only; eligible points are not E4 history-supported points.')
    output = roots[1]/'paired_e3_coverage_comparison.json'
    write_json(output,report)
    print(output)
    print('partial - original E3 (percentage points):',difference)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root',default='work_dirs/recallocc-modules-formal-20261102')
    parser.add_argument('--partial-root',required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--bootstrap',type=int,default=1000)
    args=parser.parse_args()
    compare(args.original_root,args.partial_root,args.seed,args.bootstrap)
