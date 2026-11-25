import importlib.util
import json
from pathlib import Path
import statistics

import pytest

spec = importlib.util.spec_from_file_location('details', Path(__file__).with_name('export_planning_five_seeds_details.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture_result():
    prefix = {h: dict(L2=v, obj_col=v/100, obj_box_col=v/50)
              for h, v in zip(('1s', '2s', '3s'), (0.1, 0.2, 0.3))}
    endpoint = {h: dict(L2=v, obj_col=v/100, obj_box_col=v/50)
                for h, v in zip(('1s', '2s', '3s'), (0.2, 0.4, 0.6))}
    return dict(checkpoint_sha256='abc', epoch=1, occupancy_samples=4219, planning_samples=4219,
        occupancy=dict(IoU=[25, 23, 22, 21], mIoU=[18, 15, 13, 11]),
        planning=dict(prefix=prefix, endpoint=endpoint,
            per_step=dict(L2=[.1, .2, .3, .4, .5, .6], obj_col=[.001]*6, obj_box_col=[.002]*6)))


def prepare(tmp_path, completed=True):
    (tmp_path/'registry.json').write_text(json.dumps(dict(protocol='test', seeds=[17,23,41,73,101],
        entries=[dict(seed=s) for s in (17,23,41,73,101)], epochs=1,
        iterations_per_seed=9865, validation_samples=4219)))
    folder = tmp_path/'seed_17'
    out = folder/'eval_epoch_1'
    out.mkdir(parents=True)
    result = fixture_result()
    (out/'all_metrics.json').write_text(json.dumps(result))
    (folder/'checkpoint_audit.json').write_text(json.dumps(dict(seed=17, sha256='abc', epoch=1,
        iterations=9865, frozen_unchanged=True)))
    if completed:
        (out/'complete.json').write_text(json.dumps(dict(status='PASS', checkpoint_sha256='abc', samples=4219)))
    return out


def test_all_horizons_units_and_aggregation_are_kept_separate():
    rows = m.flatten(17, fixture_result())
    assert len(rows) == 52
    occ = [r for r in rows if r['domain']=='occupancy' and r['aggregation']=='endpoint' and r['metric']=='mIoU']
    assert [r['horizon'] for r in occ] == ['0s','1s','2s','3s']
    assert [r['value'] for r in occ] == [18,15,13,11]
    point = [r for r in rows if r['aggregation']=='per_step' and r['metric']=='ObjCollision']
    assert len(point)==6 and all(r['value']==.1 and r['unit']=='percent' for r in point)
    l2 = {r['aggregation']:r['value'] for r in rows if r['metric']=='L2' and r['horizon']=='3s'}
    assert l2['prefix']==.3 and l2['endpoint']==.6


def test_sample_standard_deviation():
    rows = m.flatten(17, fixture_result())
    other = dict(rows[0], seed=23, value=rows[0]['value']+2)
    stats = m.summarize_rows([rows[0],other])[0]
    assert stats['n']==2 and stats['sample_std']==pytest.approx(statistics.stdev([25,27]))


def test_partial_export_retains_raw_results_without_zero_filling(tmp_path):
    out = prepare(tmp_path)
    before = (out/'all_metrics.json').read_bytes()
    result = m.export(tmp_path)
    assert result['completed_seeds']==1 and not result['final']
    assert result['seeds'][0]['results']==fixture_result()
    assert all(s['results'] is None for s in result['seeds'][1:])
    assert all(s['n']==1 and s['sample_std'] is None for s in result['statistics'])
    assert (out/'all_metrics.json').read_bytes()==before


def test_requires_completion_marker(tmp_path):
    prepare(tmp_path, completed=False)
    result = m.export(tmp_path)
    assert result['completed_seeds']==0 and result['statistics']==[]


def test_rejects_checkpoint_provenance_mismatch(tmp_path):
    out = prepare(tmp_path)
    result = fixture_result()
    result['checkpoint_sha256']='wrong'
    (out/'all_metrics.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='provenance mismatch'):
        m.export(tmp_path)
