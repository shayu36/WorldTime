"""Round 1: Adapter-only, Query-ID aligned, stop-gradient temporal features."""
_base_ = ['./sparseworld-traj-memory-future-adapter-v2-t1-temporal-control.py']

work_dir = 'work_dirs/t1-temporal-from-best3'
model = dict(query_memory_cfg=dict(future_memory_adapter_v2=dict(
    temporal_adapter_enabled=True,
    # sigmoid(-2) ~= .119 at zero gate input; learned point gates also see
    # current adapted features and the normalized previous Query state.
    temporal_gate_bias=-2.0,
    # Bound each feature update component in normalized-feature units (not
    # metres or logits). The final projection starts exactly at zero.
    temporal_max_delta=1.0)))
