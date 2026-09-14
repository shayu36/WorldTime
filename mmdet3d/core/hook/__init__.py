# Copyright (c) OpenMMLab. All rights reserved.
from .ema import MEGVIIEMAHook
from .utils import is_parallel
from .sequentialcontrol import SequentialControlHook
from .syncbncontrol import SyncbnControlHook
from .meanteacher import MeanTeacher
from .set_epoch_info_hook import CustomSetEpochInfoHook
from .query_memory_training_hook import (
    QueryMemoryConnectivityOptimizerHook,
    QueryMemoryJointConnectivityOptimizerHook,
    QueryMemoryPhase2ConnectivityOptimizerHook,
    QueryMemoryPhase3ConnectivityOptimizerHook,
)

__all__ = ['MEGVIIEMAHook', 'is_parallel', 'SequentialControlHook',
           'SyncbnControlHook', 'MeanTeacher', 'CustomSetEpochInfoHook',
           'QueryMemoryConnectivityOptimizerHook',
           'QueryMemoryJointConnectivityOptimizerHook',
           'QueryMemoryPhase2ConnectivityOptimizerHook',
           'QueryMemoryPhase3ConnectivityOptimizerHook']
