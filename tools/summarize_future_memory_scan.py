"""Select a provisional occupancy-protection setting on held-out scenes.

The split is deterministic by scene, not by frame. This is still a diagnostic
subset, not evidence of full-validation or retraining improvement.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from summarize_future_memory_branches import scores


def scene_partition(scene):
    digest = hashlib.sha256(scene.encode('utf-8')).digest()
    return 'selection' if digest[0] % 2 == 0 else 'holdout'


def evaluate_partition(hist, names):
    observed = scores(hist.sum(0))
    ab = names.index('ab')
    result = {}
    for i, name in enumerate(names):
        result[name] = {}
        for metric in ('iou', 'miou'):
            values = observed[metric][i]
            delta = values - observed[metric][ab]
            result[name][metric] = dict(
                values=values.tolist(), delta_vs_ab=delta.tolist(),
                future_delta=float(delta[1:].mean()),
                targeted_2s_3s_delta=float(delta[2:4].mean()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    args = parser.parse_args()
    root = Path(args.directory)
    output = root / 'scan_selection.json'
    if output.exists():
        raise FileExistsError(output)
    provenance = json.loads((root / 'provenance.json').read_text())
    if provenance['experiment'] != 'scan':
        raise ValueError('Expected a scan diagnostic')
    names = provenance['variants']
    with np.load(root / 'per_sample_confusions.npz') as data:
        hist = data['hist']
    manifest = provenance['manifest']
    if hist.shape[:2] != (len(manifest), len(names)):
        raise ValueError('Confusion dimensions do not match provenance')
    partition = [scene_partition(row['scene']) for row in manifest]
    results = {}
    for label in ('selection', 'holdout'):
        mask = np.array([p == label for p in partition])
        if not mask.any():
            raise ValueError(f'No scenes in {label} partition')
        results[label] = dict(samples=int(mask.sum()),
                              scenes=len({manifest[i]['scene'] for i in np.flatnonzero(mask)}),
                              metrics=evaluate_partition(hist[mask], names))
    # Predeclared decision rule: seek mIoU recovery at 2/3 s, but reject
    # settings whose binary IoU drops by more than 0.10 percentage points.
    eligible = []
    for name in names:
        if not name.startswith('protected_occ_alpha_'):
            continue
        candidate = results['selection']['metrics'][name]
        if candidate['iou']['targeted_2s_3s_delta'] >= -0.10:
            eligible.append(name)
    winner = (max(eligible, key=lambda n: (
        results['selection']['metrics'][n]['miou']['targeted_2s_3s_delta'], n))
              if eligible else None)
    decision = dict(candidate=winner, eligible=eligible,
                    rule='Select by 2s/3s mIoU on selection scenes only, requiring 2s/3s binary IoU delta >= -0.10 pp.',
                    holdout_result=(results['holdout']['metrics'][winner]
                                    if winner is not None else None),
                    status='diagnostic only; full validation required before model change')
    report = dict(split='SHA256(scene) first byte parity', partitions=results,
                  decision=decision)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(decision, indent=2))
    print(output)


if __name__ == '__main__':
    main()
