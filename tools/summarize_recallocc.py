"""Read real evaluation JSON; per-seed horizon means, sample SD, paired effects."""
import argparse
import csv
import io
import math
from pathlib import Path
import statistics

from recallocc_common import MODES, digest, read_json, write_json

COLUMNS = ['mIoU@1s','mIoU@2s','mIoU@3s','Mean mIoU',
           'IoU@1s','IoU@2s','IoU@3s','Mean IoU']
HEADERS = ['Model','Read mode','Gate mode','Seed']+COLUMNS
CONTRASTS = {'E2-E1': {'E2':1,'E1':-1},'E4-E3': {'E4':1,'E3':-1},
             'E3-E1': {'E3':1,'E1':-1},'E4-E2': {'E4':1,'E2':-1},
             'E4-E3-E2+E1': {'E4':1,'E3':-1,'E2':-1,'E1':1},
             'E4-SparseWorld': {'E4':1,'SparseWorld':-1}}


def values(report):
    if report['units']!='percent' or report['horizons']!=[0,1,2,3]:
        raise ValueError('Expected native percent metrics for horizons [0,1,2,3]')
    result=[]
    for metric in ('miou','iou'):
        xs=[float(row[metric]) for row in report['metrics'][1:]]
        if len(xs)!=3 or not all(math.isfinite(x) and 0<=x<=100 for x in xs):
            raise ValueError('Incomplete/invalid horizon metrics')
        result+=xs+[statistics.mean(xs)]
    return dict(zip(COLUMNS,result))


def stats(rows):
    return {key:dict(n=len(rows),mean=statistics.mean([r[key] for r in rows]) if rows else None,
        sample_std=statistics.stdev([r[key] for r in rows]) if len(rows)>1 else None) for key in COLUMNS}


def summarize(root):
    root=Path(root)
    study=read_json(root/'study.json')
    seeds=sorted(set(study['spec']['seeds']) | {int(p.name[5:]) for e in MODES for p in (root/e).glob('seed_*')})
    rows=[]
    observed={}
    expected_manifest=study['eval_manifest_sha256']
    def load(path):
        report=read_json(path)
        if report['manifest_sha256']!=expected_manifest:
            raise ValueError(f'Evaluation manifest mismatch: {path}')
        if report['full_validation'] != (study['spec']['smoke_steps']==0):
            raise ValueError(f'Mixed smoke/full evaluation: {path}')
        return report
    for experiment,(read,gate) in MODES.items():
        for seed in seeds:
            run=root/experiment/f'seed_{seed}'/'run.json'
            record=read_json(run) if run.exists() else None
            data=None
            if record and record.get('evaluation'):
                report=load(record['evaluation'])
                if report['model']!=experiment or report['seed']!=seed:
                    raise ValueError('Evaluation model/seed mismatch')
                if report['checkpoint_sha256']!=record['selected_checkpoint_sha256']:
                    raise ValueError('Evaluation used a different checkpoint')
                data=values(report)
                observed[(experiment,seed)]=data
            rows.append(dict(zip(HEADERS[:4],[experiment,read,gate,seed]),
                             **(data or {k:None for k in COLUMNS})))
    baseline=root/'SparseWorld.json'
    if baseline.exists():
        report=load(read_json(baseline)['evaluation'])
        if report['model']!='SparseWorld':
            raise ValueError('Invalid baseline record')
        data=values(report)
        observed[('SparseWorld',None)]=data
    else:
        data=None
    rows.append(dict(zip(HEADERS[:4],['SparseWorld','—','—','fixed']),
                     **(data or {k:None for k in COLUMNS})))
    aggregates={e:stats([v for (m,s),v in observed.items() if m==e]) for e in MODES}
    paired={}
    for name,coefficients in CONTRASTS.items():
        per_seed={}
        for seed in seeds:
            parts=[(coefficient,observed.get((model,None if model=='SparseWorld' else seed)))
                   for model,coefficient in coefficients.items()]
            per_seed[str(seed)]=({key:sum(c*v[key] for c,v in parts) for key in COLUMNS}
                                 if all(v is not None for c,v in parts) else None)
        paired[name]=dict(per_seed=per_seed,
                         statistics=stats([v for v in per_seed.values() if v is not None]))
    return dict(units='percent; differences in percentage points',
        scope='SMOKE ONLY' if study['spec']['smoke_steps'] else 'full validation',
        checkpoint_selection=study['checkpoint_selection'],rows=rows,aggregates=aggregates,
        paired_effects=paired,missing=[dict(model=r['Model'],seed=r['Seed']) for r in rows if r['Mean mIoU'] is None])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root',required=True)
    parser.add_argument('--output-dir')
    args=parser.parse_args()
    root=Path(args.work_root)
    output=Path(args.output_dir) if args.output_dir else root/'summary'
    output.mkdir(parents=True,exist_ok=True)
    report=summarize(root)
    write_json(output/'results.json',report)
    buf=io.StringIO()
    writer=csv.DictWriter(buf,fieldnames=HEADERS)
    writer.writeheader();writer.writerows(report['rows'])
    (output/'results.csv').write_text(buf.getvalue())
    lines=[f"RecallOcc — {report['scope']}; unit: %. Checkpoint: {report['checkpoint_selection']}.",
        'Missing entries are not estimated. Standard deviation is sample SD (ddof=1); n<2 has no SD.', '',
        '| '+' | '.join(HEADERS)+' |','| '+' | '.join(['---']*len(HEADERS))+' |']
    for row in report['rows']:
        lines.append('| '+' | '.join('MISSING' if row[k] is None else
            f'{row[k]:.4f}' if k in COLUMNS else str(row[k]) for k in HEADERS)+' |')
    for name,data in [('Across seeds',report['aggregates']),
                      ('Paired effects (percentage points)',{k:v['statistics'] for k,v in report['paired_effects'].items()})]:
        lines+=['',name,'','| Model / contrast | n | Mean mIoU ± sample SD | Mean IoU ± sample SD |',
                '| --- | --- | --- | --- |']
        for label,columns in data.items():
            entries=[]
            for key in ('Mean mIoU','Mean IoU'):
                v=columns[key]
                entries.append('MISSING' if v['mean'] is None else f"{v['mean']:.4f} ± "+
                    (f"{v['sample_std']:.4f}" if v['sample_std'] is not None else 'N/A'))
            lines.append(f"| {label} | {columns['Mean mIoU']['n']} | {' | '.join(entries)} |")
    (output/'results.md').write_text('\n'.join(lines)+'\n')
    print(output/'results.md')

if __name__=='__main__':
    main()
