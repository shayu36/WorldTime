"""Compare live E4 to the independent BEFORE-edit source on real images/cache."""
import argparse
import copy
import importlib.util
from pathlib import Path
import types

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import wrap_fp16_model
from mmdet.apis import set_random_seed
from mmdet.utils import compat_cfg
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.utils import patch_config
from recallocc_common import ROOT, digest, write_json, state_dict


def snapshot(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, dict):
        return {k:snapshot(v) for k,v in value.items() if k not in ('outs',)}
    if isinstance(value, (list,tuple)):
        return [snapshot(x) for x in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--reference-output', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--indices', type=int, nargs='+', default=[30, 2000])
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    reference = Path(args.reference)
    spec = importlib.util.spec_from_file_location('recallocc_before_edit', reference/'query_memory.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    torch.set_num_threads(1)
    set_random_seed(0, deterministic=True)
    cfg = patch_config(compat_cfg(Config.fromfile(str(ROOT/'configs/sparseworld/recallocc/E4.py'))))
    cfg.data.test.test_mode = True
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.align_after_view_transfromation = True
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(torch.utils.data.Subset(dataset, args.indices),
        samples_per_gpu=1, workers_per_gpu=1, num_gpus=1, dist=False, shuffle=False, seed=0)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    wrap_fp16_model(model)
    model.load_state_dict(state_dict(torch.load(args.checkpoint, map_location='cpu')), strict=True)
    model.cuda().eval()
    new_adapter = model.future_memory_adapter_v2
    old_adapter = old.FutureMemoryAdapterV2(**dict(cfg.model.query_memory_cfg.future_memory_adapter_v2)).cuda().eval()
    old_adapter.load_state_dict(new_adapter.state_dict(), strict=True)
    parallel = MMDataParallel(model, device_ids=[0]).eval()
    original = model.forward_backbone
    captured = []
    def capture(this, *a, **kw):
        result = original(*a, **kw)
        captured.append(snapshot({k:v for k,v in result.items() if k in (
            'cls_score','refine_pts','forecast_points_list','forecast_semantics_list',
            'forecast_semantic_logits_list','forecast_occupancy_logits_list',
            'baseline_forecast_semantics_list','baseline_forecast_points_list','pred_trajs_list')}))
        return result
    model.forward_backbone = types.MethodType(capture, model)
    saved = [] if args.capture else torch.load(args.reference_output, map_location='cpu')
    records = []
    diffs = {}
    def compare(a,b,path):
        if torch.is_tensor(a):
            assert a.shape == b.shape, path
            delta = float((a.float()-b.float()).abs().max()) if a.numel() else 0.
            diffs[path] = max(diffs.get(path,0.), delta)
            torch.testing.assert_close(a,b,rtol=0,atol=0,msg=path)
        elif isinstance(a,dict):
            for key in a:
                compare(a[key],b[key],path+'/'+str(key))
        elif isinstance(a,list):
            assert len(a)==len(b)
            for i,(x,y) in enumerate(zip(a,b)):
                compare(x,y,path+'/'+str(i))
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for idx,batch in zip(args.indices,loader):
            outputs=[]
            for adapter in (new_adapter,):
                model.future_memory_adapter_v2=adapter
                seen=[]
                hook=adapter.register_forward_hook(lambda m,a,o:seen.append(snapshot({k:o[k] for k in (
                    'delta_s','delta_o','delta_p','gate_sem','gate_occ','gate_pos','context')})))
                set_random_seed(0,deterministic=True)
                result=parallel(return_loss=False,rescale=True,**batch)
                hook.remove()
                outputs.append(dict(backbone=captured.pop(),adapter=seen,
                    decoded={k:snapshot(v) for k,v in result.items() if k.startswith('semantic_occ_')}))
            if args.capture:
                saved.append(outputs[0])
                compare(outputs[0],outputs[0],'capture')
            else:
                compare(saved[len(records)],outputs[0],'E4')
            records.append(dict(index=idx, all_bitwise_equal=True))
    if args.capture:
        torch.save(saved, args.reference_output)
    write_json(output, dict(reference_source_sha256=digest(reference/'query_memory.py'),
        live_source_sha256=digest(ROOT/'mmdet3d/models/sparsedetectors/query_memory.py'),
        checkpoint=args.checkpoint, checkpoint_sha256=digest(args.checkpoint),
        rtol=0,atol=0,samples=records,max_abs_differences=diffs,
        max_abs_difference=max(diffs.values()),
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
    print(f'E4 old-source vs new-source: {len(records)} real samples, max abs diff={max(diffs.values())}')

if __name__ == '__main__':
    main()
