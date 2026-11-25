"""The existing tools/train.py runner, with RecallOcc strict-load/RNG/audit hooks."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import types

import torch
from mmcv.runner import HOOKS, Hook, get_dist_info
from mmdet.apis import set_random_seed
from recallocc_common import PREFIX, read_json, state_dict, tensor_hashes, validate_state


@HOOKS.register_module()
class RecallOccAuditHook(Hook):
    def __init__(self, source_checkpoint, seed, run_manifest):
        self.source_checkpoint = source_checkpoint
        self.seed = seed
        self.run_manifest = run_manifest

    def before_run(self, runner):
        self.rank, self.world = get_dist_info()
        model = runner.model.module
        checkpoint = torch.load(os.environ.get('RECALLOCC_LOAD_SOURCE', self.source_checkpoint), map_location='cpu')
        state = state_dict(checkpoint)
        validate_state(model, state)
        # MMCV's legacy runner loader uses strict=False. Verify every loaded
        # value, then a strict load: missing, extra and shape errors are fatal.
        actual = model.state_dict()
        differing = [k for k in state if not torch.equal(actual[k].detach().cpu(),
                                                        state[k].to(actual[k].dtype))]
        if differing:
            raise RuntimeError(f'Runner did not load exact checkpoint: {differing}')
        model.load_state_dict(state, strict=True)
        model.validate_query_memory_training_setup(optimizer=runner.optimizer)
        if 'fp16' in checkpoint.get('meta', {}):
            runner.meta['fp16'] = checkpoint['meta']['fp16']
            # Our priority 45 follows the native AMP optimizer hook (40),
            # so restore its scaler explicitly, including IterBasedRunner.
            for hook in runner.hooks:
                if hasattr(hook, 'loss_scaler'):
                    hook.loss_scaler.load_state_dict(checkpoint['meta']['fp16']['loss_scaler'])
        self.frozen_hashes = tensor_hashes({k:v for k,v in model.state_dict().items()
                                          if not k.startswith(PREFIX)})
        self.stream = (Path(runner.work_dir)/f'audit_rank{self.rank}.jsonl').open('a')
        self.batch_tokens = []
        self.batch_hash = None
        original = model.train_step
        def capture(this, data, *args, **kwargs):
            meta = data['img_metas']
            meta = meta.data if hasattr(meta, 'data') else meta
            while isinstance(meta, (list, tuple)) and meta and isinstance(meta[0], (list,tuple)):
                meta = meta[0]
            self.batch_tokens = [str(x.get('sample_idx')) for x in meta]
            if runner.iter < 6:
                img = data['img']
                img = img.data if hasattr(img, 'data') and not torch.is_tensor(img) else img
                while isinstance(img, (list, tuple)):
                    img = img[0]
                self.batch_hash = hashlib.sha256(img.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            return original(data, *args, **kwargs)
        model.train_step = types.MethodType(capture, model)
        run = read_json(self.run_manifest)
        runner.meta['recallocc'] = dict(experiment=run['experiment'], seed=self.seed,
            config_sha256=run['config_sha256'], source_sha256=run['source_sha256'],
            base_sha256=run['base_sha256'], checkpoint_selection=run['checkpoint_selection'])
        torch.cuda.reset_peak_memory_stats()
        self.record(dict(event='start', rank=self.rank, world_size=self.world,
            strict_loaded=True, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            trainable_tensors=sum(p.requires_grad for p in model.parameters())))
        # Model/module construction consumes different amounts of RNG. Reset
        # after loading, before dataloader iteration; never claim seed alone
        # aligned initialization (the canonical checkpoint does that).
        set_random_seed(self.seed + self.rank, deterministic=True)

    def record(self, data):
        self.stream.write(json.dumps(data, allow_nan=False)+'\n')
        self.stream.flush()

    def before_train_epoch(self, runner):
        set_random_seed(self.seed + runner.epoch*100003 + self.rank, deterministic=True)

    def before_train_iter(self, runner):
        # Align model augmentation + semantic dropout across arms even after
        # epoch-boundary resume. Worker RNG/sampler use the common cfg.seed.
        set_random_seed(self.seed + runner.iter*100003 + self.rank, deterministic=True)

    def after_train_iter(self, runner):
        if runner.iter >= 6 and (runner.iter+1) % 50:
            return
        model = runner.model.module
        for name, module in model.named_children():
            if name != 'future_memory_adapter_v2':
                assert not module.training, name
        norms, absent = {}, []
        for name, param in model.named_parameters():
            if not name.startswith(PREFIX):
                assert not param.requires_grad and param.grad is None, name
            elif param.grad is None:
                absent.append(name)
            else:
                assert torch.isfinite(param.grad).all(), name
                norms[name] = float(param.grad.detach().float().norm())
        if absent:
            raise RuntimeError(f'Unused trainable parameters: {absent}')
        loss = float(runner.outputs['loss'].detach())
        if not torch.isfinite(torch.tensor(loss)):
            raise RuntimeError(f'Nonfinite loss {loss}')
        model._assert_memory_finetune_temporal_state()
        self.record(dict(event='step', iteration=runner.iter+1, loss=loss,
            tokens=self.batch_tokens, image_sha256=self.batch_hash, gradient_norms=norms,
            frozen_eval=True, tass_unchanged=True,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))

    def after_run(self, runner):
        actual = tensor_hashes({k:v for k,v in runner.model.module.state_dict().items()
                                if not k.startswith(PREFIX)})
        assert actual == self.frozen_hashes, 'Frozen weights/buffers changed'
        self.record(dict(event='complete', frozen_parameters_and_buffers_unchanged=True,
                         iterations=runner.iter, epochs=runner.epoch))
        self.stream.close()


if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('train.py')), run_name='__main__')
