"""Run the normal trainer with bounded, explicit internal-ablation checks."""
import json
from pathlib import Path
import runpy

import torch
from mmcv.runner import HOOKS, Hook, get_dist_info


@HOOKS.register_module()
class InternalAblationAuditHook(Hook):
    def __init__(self, source_checkpoint, probe=False):
        self.source_checkpoint = source_checkpoint
        self.probe = probe
        self.steps = 0

    def before_run(self, runner):
        source = torch.load(self.source_checkpoint, map_location='cpu')
        # MMCV IterBasedRunner.resume does not restore meta (unlike the epoch
        # runner). Restore the source scaler for the matched short probes.
        if self.probe and 'fp16' in source['meta']:
            runner.meta['fp16'] = source['meta']['fp16']
        self.source_epoch = source['meta']['epoch']
        self.source_iter = source['meta']['iter']
        model = runner.model.module
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert trainable and all(n.startswith('future_memory_adapter_v2.')
                                 for n in trainable)
        model.validate_query_memory_training_setup(optimizer=runner.optimizer)
        rank, _ = get_dist_info()
        self.path = Path(runner.work_dir) / f'audit_rank{rank}.jsonl'
        self.stream = self.path.open('a')
        self.stream.write(json.dumps(dict(event='start', rank=rank,
            source_epoch=self.source_epoch, source_iter=self.source_iter,
            trainable_tensors=len(trainable), probe=self.probe)) + '\n')
        self.stream.flush()
        torch.cuda.reset_peak_memory_stats()

    def before_train_iter(self, runner):
        if self.probe:
            # A short IterBasedRunner probe uses a fresh, common sampler but
            # the same frozen temporal allocation and source epoch in both arms.
            runner.model.module.set_epoch(self.source_epoch)

    def after_train_iter(self, runner):
        self.steps += 1
        if self.steps > 6 and self.steps % 50:
            return
        model = runner.model.module
        for name, module in model.named_children():
            if name != 'future_memory_adapter_v2':
                assert not module.training, name
        norms = {}
        for name, param in model.named_parameters():
            if not name.startswith('future_memory_adapter_v2.'):
                assert not param.requires_grad and param.grad is None, name
            elif param.grad is not None:
                assert torch.isfinite(param.grad).all(), name
                norms[name] = float(param.grad.detach().float().norm())
        loss = float(runner.outputs['loss'].detach())
        if not torch.isfinite(torch.tensor(loss)):
            raise RuntimeError(f'Nonfinite internal-ablation loss: {loss}')
        record = dict(event='step', step=self.steps, iteration=runner.iter + 1,
                      loss=loss, baseline_frozen=True, gradient_norms=norms,
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        self.stream.write(json.dumps(record) + '\n')
        self.stream.flush()

    def after_run(self, runner):
        self.stream.write(json.dumps(dict(event='complete', steps=self.steps,
                                          iteration=runner.iter)) + '\n')
        self.stream.close()


if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('train.py')), run_name='__main__')
