"""Verify published source/result hashes and arithmetic without GPU/data."""
import gzip, hashlib, json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'publication/manifest.json').read_text())
for name,expected in manifest['published_hashes'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
for row in json.loads((root/'results/summary.json').read_text()):
    for metric in ('IoU','mIoU'):
        actual=sum(row[f'{metric}_{h}s'] for h in (1,2,3))/3
        assert abs(actual-row[f'Future_{metric}'])<1e-10,(row,metric)
    assert row['samples']==4219
print('Published files, 4219-sample records and future means: PASS')
