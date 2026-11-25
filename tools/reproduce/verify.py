"""Check release weights, dependencies, external data and registered model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment-only',action='store_true')
    p.add_argument('--model',action='store_true')
    p.add_argument('--data',choices=['train','val','both'])
    a=p.parse_args()
    import torch,mmcv,mmdet,mmseg,torch_scatter
    from mmdet3d.ops.bev_pool_v2 import bev_pool_v2_ext
    from mmdet3d.models.sparsedetectors.csrc import wrapper
    assert torch.__version__=='2.0.1+cu118',torch.__version__
    assert mmcv.__version__=='1.7.0' and mmdet.__version__=='2.28.0' and mmseg.__version__=='0.30.0'
    assert wrapper.MSMV_CUDA,'Build the MSMV CUDA extension'
    print('Environment / both CUDA extensions: PASS')
    if a.environment_only:return
    for rel,entry in json.loads((ROOT/'artifacts/weights_manifest.json').read_text()).items():
        path=ROOT/rel
        if not path.exists() or path.stat().st_size!=entry['bytes']:
            raise RuntimeError(f'Missing full checkpoint {rel}; run git lfs pull')
        assert digest(path)==entry['sha256'],f'Checkpoint SHA256 mismatch: {rel}'
    print('Published checkpoint SHA256: PASS')
    if a.model:
        from mmcv import Config
        from mmdet3d.models import build_model
        from mmcv.runner import build_optimizer
        cfg=Config.fromfile(str(ROOT/'configs/sparseworld/nuscenes-temporal/memanchorworld-joint-reproduce.py'))
        model=build_model(cfg.model)
        state=torch.load(ROOT/'artifacts/seed17/epoch_1.pth',map_location='cpu')['state_dict']
        model.load_state_dict(state,strict=True);model.train()
        optimizer=build_optimizer(model,cfg.optimizer)
        info=model.validate_query_memory_training_setup(optimizer,cfg.optimizer)
        assert info['trainable_count']==1722147
        print('Strict seed17 model load / two trainable modules / frozen Baseline: PASS')
    if a.data:
        manifest=json.loads((ROOT/'artifacts/input_manifest.json').read_text())
        for entry in manifest['files']:
            rel=entry['path']
            if a.data=='val' and '_train' in Path(rel).name:continue
            if a.data=='train' and ('_val' in Path(rel).name or rel.startswith('stp3_val/')):continue
            path=ROOT/rel
            if not path.exists():raise FileNotFoundError(f'Required external input: {rel}; see docs/REPRODUCIBILITY.md')
            assert digest(path)==entry['sha256'],f'Input differs from recorded run: {rel}'
        for split in (['train','val'] if a.data=='both' else [a.data]):
            path=ROOT/f'data/query_memory/sparseworld_epoch56_schema2_{split}'
            assert path.is_dir(),f'Missing fixed cache: {path}'
        print('External input hashes / cache locations: PASS')


if __name__=='__main__':main()
