"""Summarize paired diagnostic matrices, with paired scene bootstrap intervals.

Intervals describe variability of the selected scenes/frames only, not seed
uncertainty, retraining performance, or the full validation score.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def scores(hist):
    tp = np.diagonal(hist, axis1=-2, axis2=-1)
    union = hist.sum(-1) + hist.sum(-2) - tp
    classes = np.divide(tp, union, out=np.full_like(tp, np.nan, dtype=float), where=union > 0)
    occ_tp = hist[..., :17, :17].sum(axis=(-1, -2))
    occ_fp = hist[..., 17, :17].sum(-1)
    occ_fn = hist[..., :17, 17].sum(-1)
    return dict(iou=100*occ_tp/(occ_tp+occ_fp+occ_fn), miou=100*np.nanmean(classes[..., :17], -1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--output-name', default='summary.json')
    args = parser.parse_args()
    root = Path(args.directory)
    if Path(args.output_name).name != args.output_name:
        parser.error('--output-name must be a filename inside the diagnostic directory')
    target = root / args.output_name
    if target.exists():
        raise FileExistsError(target)
    report = json.loads((root/'report.json').read_text())
    provenance = json.loads((root/'provenance.json').read_text())
    hist = np.load(root/'per_sample_confusions.npz')['hist']
    assert len(hist) == len(provenance['manifest']) == report['samples']
    scenes = sorted({r['scene'] for r in provenance['manifest']})
    scene_index = {scene: i for i, scene in enumerate(scenes)}
    clustered = np.zeros((len(scenes), *hist.shape[1:]), dtype=np.float64)
    for row, entry in zip(hist, provenance['manifest']):
        clustered[scene_index[entry['scene']]] += row
    rng = np.random.default_rng(0)
    counts = rng.multinomial(len(scenes), np.full(len(scenes), 1/len(scenes)), size=args.bootstrap)
    bootstrap_hist = (counts @ clustered.reshape(len(scenes), -1)).reshape(args.bootstrap, *hist.shape[1:])
    observed = scores(hist.sum(0))
    boot = scores(bootstrap_hist)
    names = provenance['variants']
    boot_tp = np.diagonal(bootstrap_hist, axis1=-2, axis2=-1)
    boot_union = bootstrap_hist.sum(-1) + bootstrap_hist.sum(-2) - boot_tp
    boot_class = np.divide(100*boot_tp, boot_union,
                           out=np.full_like(boot_tp, np.nan), where=boot_union > 0)
    summary = dict(samples=len(hist), scenes=len(scenes), bootstrap_replicates=args.bootstrap,
                   ci_scope='Paired scene bootstrap of this fixed sampled-frame subset, not training seeds or full validation',
                   metrics={}, class_deltas_vs_ab={}, class_delta_ci95={})
    for i, name in enumerate(names):
        rows = {}
        for metric in ('iou', 'miou'):
            deltas = observed[metric][i] - observed[metric][0]
            boot_delta = boot[metric][:, i] - boot[metric][:, 0]
            rows[metric] = dict(values=observed[metric][i].tolist(), delta_vs_ab=deltas.tolist(),
                                delta_ci95=np.percentile(boot_delta, [2.5, 97.5], axis=0).T.tolist(),
                                future_mean=float(observed[metric][i, 1:].mean()),
                                future_delta=float(deltas[1:].mean()),
                                future_delta_ci95=np.percentile(boot_delta[:, 1:].mean(-1), [2.5, 97.5]).tolist())
        summary['metrics'][name] = rows
        summary['class_deltas_vs_ab'][name] = [
            {c: round(row['classes'][c]['iou'] - report['metrics']['ab'][h]['classes'][c]['iou'], 6)
             for c in report['classes'][:17]}
            for h, row in enumerate(report['metrics'][name])]
        class_ci = np.nanpercentile(boot_class[:, i] - boot_class[:, 0], [2.5, 97.5], axis=0)
        summary['class_delta_ci95'][name] = [
            {c: class_ci[:, h, j].tolist() for j, c in enumerate(report['classes'][:17])}
            for h in range(4)]
        print(name, {m: dict(future=round(rows[m]['future_mean'], 4),
                            delta=round(rows[m]['future_delta'], 4),
                            ci95=np.round(rows[m]['future_delta_ci95'], 4).tolist())
                     for m in ('iou', 'miou')})
    target.write_text(json.dumps(summary, indent=2))
    print(target)


if __name__ == '__main__':
    main()
