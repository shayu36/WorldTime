"""RecallOcc module ablation v2: E2_full_retrieval_concat."""
_base_ = ["./common.py"]
recallocc = dict(experiment="E2_full_retrieval_concat", variant="full_retrieval_concat", design="modules_v2")
work_dir = "work_dirs/recallocc_modules_v2/E2_full_retrieval_concat/seed_0"
model = dict(query_memory_cfg=dict(future_memory_adapter_v2=dict(ablation_mode="full_retrieval_concat")))
