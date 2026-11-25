"""Bounded allocator and telemetry around the joint occupancy/planning entry points."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ('future_memory_adapter_v2.', 'memory_planning_fusion.')


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def fingerprint(model, selected):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        if selected(name):
            x = value.detach().cpu().contiguous()
            h.update((name + str(x.shape) + str(x.dtype)).encode())
            h.update(x.numpy().tobytes())
    return h.hexdigest()


class ProbeComplete(Exception):
    pass


def main():
    kind = sys.argv.pop(1)
    if kind not in ('train', 'predict'):
        raise ValueError(kind)
    sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
    os.chdir(ROOT)
    import torch
    device = int(os.getenv('LOCAL_RANK', '0'))
    rank = int(os.getenv('RANK', '0'))
    fraction = float(os.environ.get('JOINT_MEMORY_FRACTION', '.90'))
    if not 0 < fraction <= .90:
        raise ValueError('Joint run may not exceed 90% of one GPU')
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    path = Path(os.environ['JOINT_TELEMETRY_DIR'])
    path.mkdir(parents=True, exist_ok=True)
    probe_steps = int(os.getenv('JOINT_PROBE_STEPS', '0'))
    if probe_steps and probe_steps != 40:
        raise ValueError('Bounded performance probe is exactly 40 steps')
    import statistics

    if kind == 'train':
        from mmcv.runner import EpochBasedRunner, Hook

        class Telemetry(Hook):
            def before_run(self, runner):
                self.model = runner.model.module
                names = [n for n, p in self.model.named_parameters() if p.requires_grad]
                assert len(names) == 77 and all(n.startswith(PREFIXES) for n in names)
                rates = {n: g['lr'] for g in runner.optimizer.param_groups for p in g['params']
                         for n, q in self.model.named_parameters() if p is q}
                assert all(abs(lr - (2e-5 if n.startswith(PREFIXES[0]) else 1e-4)) < 1e-12
                           for n, lr in rates.items()), rates
                assert len(runner.optimizer.state) == 0, 'Fresh optimizer required'
                self.start_frozen = fingerprint(self.model, lambda n: not n.startswith(PREFIXES))
                self.start_v2 = fingerprint(self.model, lambda n: n.startswith(PREFIXES[0]))
                self.start_fusion = fingerprint(self.model, lambda n: n.startswith(PREFIXES[1]))
                self.samples = []
                self.losses = []
                torch.cuda.reset_peak_memory_stats()
                self.previous = time.perf_counter()
                write(path / f'initial_rank_{rank}.json', dict(
                    seed=runner.meta['seed'], frozen_sha256=self.start_frozen,
                    fusion_sha256=self.start_fusion, V2_sha256=self.start_v2, optimizer_states=0,
                    trainable_prefixes=PREFIXES, learning_rates=rates, parameter_tensors=len(names),
                    world_size=torch.distributed.get_world_size(), batch_per_gpu=1,
                    allocator_fraction=fraction, pid=os.getpid()))

            def report(self, runner, status):
                frozen = fingerprint(self.model, lambda n: not n.startswith(PREFIXES))
                assert frozen == self.start_frozen, 'Frozen state changed'
                assert all(p.grad is None for n, p in self.model.named_parameters()
                           if not n.startswith(PREFIXES))
                v2 = fingerprint(self.model, lambda n: n.startswith(PREFIXES[0]))
                assert v2 != self.start_v2, 'No V2 update observed'
                fusion = fingerprint(self.model, lambda n: n.startswith('memory_planning_fusion.'))
                assert fusion != self.start_fusion, 'No planning update observed'
                values = self.samples[10:]
                write(path / f'result_rank_{rank}.json', dict(
                    status=status, seed=runner.meta['seed'], steps=len(self.samples),
                    warmup_steps=10, measured_steps=len(values),
                    mean_seconds=statistics.mean(values), median_seconds=statistics.median(values),
                    per_step_seconds=self.samples if probe_steps else self.samples[-20:],
                    all_losses_finite=all(__import__('math').isfinite(v) for v in self.losses),
                    frozen_unchanged=True, fusion_changed=True, V2_changed=True,
                    fusion_initial_sha256=self.start_fusion, fusion_final_sha256=fusion,
                    peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20,
                    allocator_fraction=fraction,
                    checkpoint_is_formal=not bool(probe_steps)))

            def after_train_iter(self, runner):
                torch.cuda.synchronize(device)
                now = time.perf_counter()
                self.samples.append(now-self.previous)
                self.previous = now
                loss = float(runner.outputs['loss'].detach())
                if not __import__('math').isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                self.losses.append(loss)
                if len(self.samples) % 20 == 0:
                    write(path / f'progress_rank_{rank}.json', dict(
                        steps=len(self.samples), seed=runner.meta['seed'],
                        recent_seconds=statistics.mean(self.samples[-20:]), loss=loss))
                if probe_steps and len(self.samples) == probe_steps:
                    self.report(runner, 'probe_complete')
                    torch.distributed.barrier()
                    raise ProbeComplete()

            def after_run(self, runner):
                self.report(runner, 'epoch_complete')

        original_run = EpochBasedRunner.run

        def registered_run(runner, *args, **kwargs):
            # Added after the AMP optimizer hook, with no change to the update.
            runner.register_hook(Telemetry(), priority='LOWEST')
            return original_run(runner, *args, **kwargs)
        EpochBasedRunner.run = registered_run
        target = ROOT / 'tools/train.py'
    else:
        target = ROOT / 'tools/eval_memory_planning_joint.py'
    sys.argv[0] = str(target)
    try:
        runpy.run_path(str(target), run_name='__main__')
    except ProbeComplete:
        assert probe_steps == 40
        print('Bounded probe complete; no checkpoint saved', flush=True)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
