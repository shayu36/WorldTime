"""CPU forward-only audit of time masks and future-label isolation."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v2_audit_helpers', ROOT / 'tests/test_future_memory_adapter_v2.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def test_real_rap_mask_blocks_later_query_features():
    from mmdet3d.models.sparsedetectors.opus_head import OPUSHead
    from mmdet3d.models.sparsedetectors.opus_transformer import OPUSSelfAttention
    torch.manual_seed(11)
    ids = torch.tensor([0, 2, 1, 6, 3, 0, 4, 5])
    attn = OPUSSelfAttention(embed_dims=8, num_heads=2, dropout=0.,
                            pc_range=[-40,-40,-1,40,40,5.4]).eval()
    attn.init_weights()
    head = SimpleNamespace(init_points=SimpleNamespace(weight=torch.zeros(8,3)),
        ind_stamps_all=ids, num_fu_frames=6,
        transformer=SimpleNamespace(decoder=SimpleNamespace(
            decoder_layers=[SimpleNamespace(self_attn=attn)])))
    OPUSHead.reset_mask(head)
    assert torch.equal(attn.ind_mask < 0, ids[None] > ids[:, None])
    features = torch.randn(1,8,8)
    points = torch.rand(1,8,2,3)
    altered = features.clone(); altered[:,ids > 1] += torch.randn_like(altered[:,ids > 1])*20
    with torch.no_grad():
        a, _ = attn(points, features)
        b, _ = attn(points, altered)
    assert torch.equal(a[:,ids <= 1], b[:,ids <= 1])
    assert not torch.equal(a[:,ids > 1], b[:,ids > 1])


def test_loader_filters_future_equal_time_and_foreign_scene():
    from mmdet3d.datasets.pipelines.loading_query_memory import LoadQueryMemoryFromFiles
    loader = LoadQueryMemoryFromFiles(history_frames=8)
    good = dict(scene_id='a',sample_idx='past',frame_idx=8,timestamp=8.)
    entries = [good, dict(good,sample_idx='future',frame_idx=11,timestamp=11.),
               dict(good,sample_idx='equal',timestamp=10.), dict(good,scene_id='b')]
    result = dict(scene_id='a', sample_idx='now', frame_idx=10, timestamp=10.,
                  query_memory_history_infos=entries)
    assert loader._history_candidates(result) == [good]


def make_runtime():
    QM = helpers.QM
    env = dict(torch=torch, np=np, decode_points_metric=QM.decode_points_metric,
               decode_semantic_occupancy_logits=QM.decode_semantic_occupancy_logits)
    forward = helpers._load_sparseworld_method('forward_backbone', env)
    future = helpers._load_sparseworld_method('_forward_backbone_future_memory_adapter', env)
    class Projection:
        def __init__(self, n): self.n=n
        def __call__(self, x):
            return (x.mean(-1, keepdim=True)*.02).expand(*x.shape[:-1],self.n)
    class Ego:
        def __call__(self, qp, qf, mp, mf): return qf+mf.mean(1,keepdim=True), None
    ids=torch.arange(7)
    model=SimpleNamespace(training=False, num_refines=2,num_fu_frames=6,
        pc_range=torch.tensor([-40.,-40.,-1.,40.,40.,5.4]),
        query_memory_log_diagnostics=False, future_memory_adapter_enabled=True,
        future_memory_adapter_version='v2', plan_head=Projection(8),
        points_scale_branch=Projection(3), ego_cross_attn=Ego(), traj_head=Projection(2),
        position_encoder=Projection(8), cls_branch=Projection(34),
        reg_branch=Projection(6),vel_branch=Projection(4))
    model.refine_points=lambda p,d: p+.01*d.reshape_as(p)
    model._refine_future_memory_points_v2=lambda p,d,active_mask=None: QM.apply_metric_position_residual(
        p,d.reshape_as(p),model.pc_range,active_mask)
    model.future_memory_adapter_v2=QM.FutureMemoryAdapterV2(embed_dims=8,num_classes=17,
        num_points=2,num_heads=2,coarse_topk_geometry=2,coarse_topk_semantic=2,point_topk=2,
        gate_mode='task_split', horizon_gate_bias=True).eval()
    memory=helpers._memory(C=8, classes=17, points=2, count=2)
    model._query_memory_context=lambda *a: memory
    def head(x, metas):
        signal=x.mean()
        return dict(query_feat=(torch.arange(56).reshape(1,7,8).float()/50+signal),
                    all_refine_pts=[torch.ones(1,7,2,3)*.5],
                    all_cls_scores=[torch.ones(1,7,2,17)*signal])
    class Head:
        ind_stamps_all=ids
        def __call__(self, x, metas): return head(x, metas)
    model.pts_bbox_head=Head()
    model.extract_feat=lambda img, metas: img
    model.simple_test_online=lambda metas,img: head(img, metas)
    model._forward_backbone_future_memory_adapter=lambda *a: future(model,*a)
    def run(**kw):
        return forward(model,torch.ones(1,1,1,1,1),
                       [{'ego2lidar':np.eye(4,dtype=np.float32)}],**kw)
    run.model=model
    return run


def test_future_labels_states_and_poses_do_not_change_forward_predictions():
    torch.manual_seed(7)
    run=make_runtime()
    kwargs=dict(temporal_ego_states={0:torch.ones(1,1,21),1:torch.zeros(1,1,21)},
                temporal_trajs=torch.zeros(1,6,2), temporal_semantics={'gt':torch.zeros(3)},
                temporal2ego=torch.eye(4),temporal_ego2global=torch.eye(4),
                temporal_agent_boxes=torch.zeros(2,7),temporal_agent_feats=torch.zeros(2,10))
    changed=copy.deepcopy(kwargs)
    for k in changed:
        if k=='temporal_ego_states': changed[k][1]=torch.full((1,1,21),float('nan'))
        else: changed[k]=object()  # unusable as model input
    with torch.no_grad():
        a,b=run(**kwargs),run(**changed)
    for k in ['forecast_points_list','forecast_semantics_list','pred_trajs_list']:
        assert all(torch.equal(x,y) for x,y in zip(a[k],b[k]))
    changed['temporal_ego_states'][0]*=2
    with torch.no_grad(): c=run(**changed)
    assert not torch.equal(a['pred_trajs_list'][0],c['pred_trajs_list'][0])


def test_training_future_trajectory_changes_supervision_mask_not_predictions():
    from mmdet3d.models.sparsedetectors.sparseworld_4d_traj import SparseWorld4DTraj
    run=make_runtime()
    model=run.model
    model.training=True
    model.future_memory_adapter_v2_activation_checkpoint=False
    model.trans_points=lambda *a: SparseWorld4DTraj.trans_points(model,*a)
    current={0:torch.ones(1,1,21)}
    with torch.no_grad():
        a=run(temporal_ego_states=current,temporal_trajs=torch.zeros(1,6,2))
        b=run(temporal_ego_states=current,temporal_trajs=torch.ones(1,6,2)*100)
    for k in ['forecast_points_list','forecast_semantics_list','pred_trajs_list']:
        assert all(torch.equal(x,y) for x,y in zip(a[k],b[k]))
    assert any(not torch.equal(x,y) for x,y in zip(
        a['forecast_points_mask_list'],b['forecast_points_mask_list']))


def test_adapter_output_changes_do_not_modify_planning_trajectory():
    run=make_runtime()
    inputs=dict(temporal_ego_states={0:torch.ones(1,1,21)})
    with torch.no_grad():
        a=run(**inputs)
        run.model.future_memory_adapter_v2.delta_o_head[-1].bias.fill_(2.)
        run.model.future_memory_adapter_v2.delta_p_head[-1].bias.fill_(.5)
        b=run(**inputs)
    assert not torch.equal(a['forecast_semantics_list'][1],b['forecast_semantics_list'][1])
    assert not torch.equal(a['forecast_points_list'][1],b['forecast_points_list'][1])
    assert all(torch.equal(x,y) for x,y in zip(a['pred_trajs_list'],b['pred_trajs_list']))


def test_planning_uses_cumulative_predictions_and_token_alignment():
    spec=importlib.util.spec_from_file_location('planning_export',ROOT/'tools/eval_future_memory_planning.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    increments=np.ones((1,6,2),dtype=np.float32)
    traj=module.cumulative_trajectory(increments)
    assert np.array_equal(traj[0,:,0],np.arange(1,7))
    got=module.merge_trajectories([[('b',traj)],[('a',traj),('b',traj)]],['a','b'])
    assert list(got)==['a','b']
    with pytest.raises(ValueError,match='coverage'):
        module.merge_trajectories([[('a',traj)]],['a','b'])
    with pytest.raises(ValueError,match='disagrees'):
        module.merge_trajectories([[('a',traj)],[('a',traj+1)]],['a'])
    with pytest.raises(ValueError,match='finite'):
        module.cumulative_trajectory(np.full((1,6,2),np.nan))


def test_legacy_planning_metric_cpu_forward():
    import sys
    sys.path.insert(0,str(ROOT/'AD-MLP/deps/stp3'))
    from stp3.planning_metrics import PlanningMetric
    metric=PlanningMetric(6).cpu()
    with torch.no_grad():
        metric.update(torch.zeros(1,6,3),torch.zeros(1,6,3),torch.zeros(1,6,200,200,dtype=torch.bool))
    assert all(float(x.sum())==0 for x in metric.compute().values())
