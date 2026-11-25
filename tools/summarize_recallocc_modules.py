"""Only module-control v2 identities; real evaluation metrics, no factor interaction."""
import argparse
import csv
from pathlib import Path
from recallocc_modules_common import MODES, read_json, write_json, digest
from summarize_recallocc import COLUMNS, values, stats

HEADERS = ['Experiment', 'Variant', 'Seed'] + COLUMNS


def summarize(root):
    root = Path(root)
    study = read_json(root/'study.json')
    seeds = sorted(set(study['spec']['seeds']) | {
        int(p.name[5:]) for e in MODES for p in (root/e).glob('seed_*')})
    rows, observed = [], {}
    for e in ('E1', *MODES, 'E4'):
        for seed in seeds:
            data = None
            path = root/e/f'seed_{seed}'/'run.json'
            if e in MODES and path.exists():
                run = read_json(path)
                if run.get('design') != 'modules_v2' or run['experiment'] != e or run['seed'] != seed:
                    raise ValueError(f'Wrong experiment identity: {path}')
                if run.get('evaluation'):
                    ev = read_json(run['evaluation'])
                    if (ev['model'], ev['seed'], ev['manifest_sha256'], ev['checkpoint_sha256']) != (
                            e, seed, study['eval_manifest_sha256'], run['selected_checkpoint_sha256']):
                        raise ValueError(f'Evaluation provenance mismatch: {path}')
                    if digest(run['selected_checkpoint']) != ev['checkpoint_sha256']:
                        raise ValueError('Selected checkpoint changed')
                    if ev['full_validation'] != (study['spec']['smoke_steps'] == 0):
                        raise ValueError('Cannot mix smoke and full validation')
                    data = values(ev)
                    observed.setdefault(e, []).append(data)
            rows.append(dict(Experiment=e[:2], Variant=e[3:] if e in MODES else
                ('SparseWorld Baseline' if e=='E1' else 'Full RecallOcc'), Seed=seed,
                **(data or {k:None for k in COLUMNS})))
    report = dict(scope='SMOKE ONLY' if study['spec']['smoke_steps'] else 'formal',
                  units='percent', rows=rows,
                  aggregates={e:stats(observed.get(e, [])) for e in MODES},
                  comparisons=None,
                  comparability='E1/E4 supplied by user; no matched evidence yet. No differences calculated.')
    write_json(root/'results_modules_v2.json', report)
    with (root/'results_modules_v2.csv').open('w') as f:
        writer=csv.DictWriter(f, fieldnames=HEADERS); writer.writeheader(); writer.writerows(rows)
    lines=[report['scope']+'; percentage units; 1/2/3s arithmetic means.', '',
           '| '+' | '.join(HEADERS)+' |', '| '+' | '.join(['---']*len(HEADERS))+' |']
    for r in rows:
        lines.append('| '+' | '.join('MISSING' if r[k] is None else
            f'{r[k]:.4f}' if k in COLUMNS else str(r[k]) for k in HEADERS)+' |')
    (root/'results_modules_v2.md').write_text('\n'.join(lines)+'\n')
    print(root/'results_modules_v2.md')
    return report

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-root', required=True)
    summarize(p.parse_args().work_root)
