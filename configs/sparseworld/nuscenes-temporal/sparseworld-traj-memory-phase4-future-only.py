_base_ = ['./sparseworld-traj-memory-phase3.py']

# Phase 4 candidate: preserve the raw OPUS observation Query for 0s and apply
# STAC-QM/refiner only to scheduled future Query groups.  This is intentionally
# a one-variable routing change from Phase 3; keep the frozen epoch-56 feature
# boundary, optimizer, cache, and batch policy unchanged.
model = dict(
    query_memory_cfg=dict(
        memory_phase3_future_only=True,
        log_diagnostics=False))

data = dict(samples_per_gpu=4, workers_per_gpu=2)

load_from = 'ckpts/epoch_56.pth'
resume_from = None
