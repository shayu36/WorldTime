#!/usr/bin/env bash
set -uo pipefail
cd /data/jxy/projects
export PYTHONPATH=/data/jxy/projects
export OMP_NUM_THREADS=1
/data/jxy/projects/env/bin/python -u tools/run_memory_planning_continue.py --work-dir /data/jxy/projects/work_dirs/memory-planning-fusion-from-t1-e3 --job-dir /data/jxy/projects/work_dirs/memory-planning-fusion-from-t1-e3/continue_3epochs --config configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-planning-fusion.py --resume /data/jxy/projects/work_dirs/memory-planning-fusion-from-t1-e3/epoch_1.pth --additional-epochs 3 2>&1 | tee -a /data/jxy/projects/work_dirs/memory-planning-fusion-from-t1-e3/continue_3epochs/controller.log
run_status=${PIPESTATUS[0]}
printf "%s\n" "$run_status" > /data/jxy/projects/work_dirs/memory-planning-fusion-from-t1-e3/continue_3epochs/exit_code
exit "$run_status"
