"""D5 contrast: same T1 continuation, only target-age slot 0 is readable."""

_base_ = ['./sparseworld-traj-memory-future-adapter-v2-internal-d5-3slots.py']

work_dir = 'work_dirs/future-memory-v2-internal-d5-1slot-from-t1e3'
model = dict(query_memory_cfg=dict(
    future_memory_adapter_v2=dict(history_slot_limit=1)))
