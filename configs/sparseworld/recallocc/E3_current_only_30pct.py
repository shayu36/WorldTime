"""Diagnostic E3: fixed 30% current-point residual coverage, no Query Memory."""
_base_ = ['./E3_current_only_task_time.py']

recallocc = dict(experiment='E3_current_only_30pct',
                 variant='current_only_task_time_30pct', design='modules_v2')
work_dir = 'work_dirs/recallocc_modules_v2/E3_current_only_30pct/seed_0'
model = dict(query_memory_cfg=dict(future_memory_adapter_v2=dict(
    ablation_mode='current_only_task_time', current_point_support_fraction=0.30)))
