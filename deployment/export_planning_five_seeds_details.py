"""Export complete per-horizon results without changing training or raw results."""
import argparse
import csv
import datetime
import fcntl
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'work_dirs/memory-planning-five-seeds-1ep-20261122'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(value)
    temp.replace(path)


def csv_text(rows, columns):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def flatten(seed, result):
    rows = []

    def add(domain, aggregation, horizon, metric, value, unit, samples):
        if not math.isfinite(value):
            raise ValueError(f'Nonfinite metric: seed={seed}, {metric}, {horizon}')
        rows.append(dict(seed=seed, domain=domain, aggregation=aggregation,
                         horizon=horizon, metric=metric, value=value, unit=unit,
                         samples=samples, checkpoint_sha256=result['checkpoint_sha256']))

    for metric in ('IoU', 'mIoU'):
        values = result['occupancy'][metric]
        if len(values) != 4:
            raise ValueError('Occupancy requires the four native 0/1/2/3s horizons')
        for h, value in zip(('0s', '1s', '2s', '3s'), values):
            add('occupancy', 'endpoint', h, metric, value, 'percent', result['occupancy_samples'])
        add('occupancy', 'future_mean', '1/2/3s', metric, statistics.mean(values[1:]),
            'percent', result['occupancy_samples'])

    plan = result['planning']
    for source, metric, factor, unit in (
        ('L2', 'L2', 1, 'm'), ('obj_col', 'ObjCollision', 100, 'percent'),
        ('obj_box_col', 'BoxCollision', 100, 'percent'),
    ):
        values = plan['per_step'][source]
        if len(values) != 6:
            raise ValueError('Planning requires six native 0.5/1/1.5/2/2.5/3s steps')
        for h, value in zip(('0.5s', '1s', '1.5s', '2s', '2.5s', '3s'), values):
            add('planning', 'per_step', h, metric, factor*value, unit, result['planning_samples'])
        for aggregation in ('prefix', 'endpoint'):
            values = [plan[aggregation][h][source] for h in ('1s', '2s', '3s')]
            for h, value in zip(('1s', '2s', '3s'), values):
                add('planning', aggregation, h, metric, factor*value, unit, result['planning_samples'])
            add('planning', aggregation+'_mean', '1/2/3s', metric, factor*statistics.mean(values),
                unit, result['planning_samples'])
    return rows


def summarize_rows(rows):
    keys = ('domain', 'aggregation', 'horizon', 'metric', 'unit')
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row['value'])
    return [dict(zip(keys, group), n=len(values), mean=statistics.mean(values),
                 sample_std=statistics.stdev(values) if len(values) > 1 else None)
            for group, values in sorted(groups.items())]


def export(output):
    registry = read(output/'registry.json')
    records, rows = [], []
    for entry in registry['entries']:
        seed = entry['seed']
        folder = output/f'seed_{seed}'
        evaluation = folder/'eval_epoch_1'
        complete = evaluation/'complete.json'
        if not complete.exists():
            records.append(dict(seed=seed, status='pending', results=None))
            continue
        done = read(complete)
        audit = read(folder/'checkpoint_audit.json')
        path = evaluation/'all_metrics.json'
        result = read(path)
        if not (done['status'] == 'PASS' and audit['seed'] == seed and
                done['checkpoint_sha256'] == audit['sha256'] == result['checkpoint_sha256'] and
                audit['epoch'] == result['epoch'] == registry['epochs'] and
                audit['iterations'] == registry['iterations_per_seed'] and
                audit['frozen_unchanged'] and
                result['occupancy_samples'] == result['planning_samples'] == done['samples'] == registry['validation_samples']):
            raise ValueError(f'Completed result provenance mismatch: seed {seed}')
        records.append(dict(seed=seed, status='complete', source=str(path),
                            source_sha256=digest(path), results=result))
        rows.extend(flatten(seed, result))
    count = sum(r['status'] == 'complete' for r in records)
    statistics_rows = summarize_rows(rows)
    document = dict(protocol=registry['protocol'], completed_seeds=count,
        expected_seeds=registry['seeds'], final=count == len(records),
        updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        registry_sha256=digest(output/'registry.json'),
        note='Raw per-seed results are preserved verbatim below. CSV collision values are percentages; raw JSON retains original fractions. Planning prefix and endpoint are separate. Occupancy has only native 0/1/2/3s evaluation, no fabricated intermediate times. Statistics use sample SD (ddof=1); incomplete seeds are not zero-filled.',
        seeds=records, statistics=statistics_rows)
    atomic_text(output/'results_by_horizon.json', json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    atomic_text(output/'results_by_horizon.csv', csv_text(rows, (
        'seed', 'domain', 'aggregation', 'horizon', 'metric', 'value', 'unit', 'samples', 'checkpoint_sha256')))
    atomic_text(output/'statistics_by_horizon.csv', csv_text(statistics_rows, (
        'domain', 'aggregation', 'horizon', 'metric', 'unit', 'n', 'mean', 'sample_std')))
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    with (args.output/'detailed_results.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        while True:
            signature = tuple(sorted(str(p) for p in args.output.glob('seed_*/eval_epoch_1/complete.json')))
            if signature != previous:
                result = export(args.output)
                print(f"Detailed results: {result['completed_seeds']}/5; final={result['final']}", flush=True)
                previous = signature
                if result['final'] or not args.watch:
                    return
            state = args.output/'parallel_state.json'
            if state.exists() and read(state).get('status') == 'failed':
                raise RuntimeError('Training/evaluation queue failed; partial detailed results retained')
            time.sleep(30)


if __name__ == '__main__':
    main()
