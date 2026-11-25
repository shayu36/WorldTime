CONFIG=$1
GPUS=$2

MASTER_PORT=29500

export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
# Keep peer-to-peer communication enabled.
# export NCCL_P2P_DISABLE=1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
/data/jxy/projects/env/bin/torchrun \
    --nproc_per_node=$GPUS \
    --master_port=$MASTER_PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --work-dir ./work_dirs/our-nusc-base \
    --launcher pytorch \
    --validate ${@:3}