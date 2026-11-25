"""Existing runner and strict audit, plus runtime no-memory guards for E3."""
from pathlib import Path
import runpy
from mmcv.runner import HOOKS
from train_recallocc import RecallOccAuditHook


def forbidden(*args, **kwargs):
    raise AssertionError('current_only_task_time executed a historical Memory path')


@HOOKS.register_module()
class RecallOccModulesAuditHook(RecallOccAuditHook):
    def before_run(self, runner):
        super().before_run(runner)
        model = runner.model.module
        if model.future_memory_adapter_v2.ablation_mode == 'current_only_task_time':
            from mmdet3d.datasets.pipelines.loading_query_memory import LoadQueryMemoryFromFiles
            LoadQueryMemoryFromFiles.__call__ = forbidden
            model._query_memory_context = forbidden
            model.future_memory_adapter_v2._reshape_memory = forbidden
            model.future_memory_adapter_v2._coarse = forbidden
            self.record(dict(event='memory_execution_guards_installed'))

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('train.py')), run_name='__main__')
