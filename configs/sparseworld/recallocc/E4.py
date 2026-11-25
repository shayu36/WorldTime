"""E4: point history read; task_horizon gate; Memory always enabled."""
_base_ = ['./common.py']
recallocc = dict(experiment='E4')
work_dir = 'work_dirs/recallocc/E4'
model = dict(query_memory_cfg=dict(future_memory_adapter_v2=dict(
    memory_read_mode='point', memory_gate_mode='task_horizon')))
