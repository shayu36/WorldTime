"""Explicit, non-overwriting shared V2 -> task/horizon gate migration."""
import argparse
import hashlib
from collections import OrderedDict
from pathlib import Path

import torch


def migrate_shared_state(state, prefix='future_memory_adapter_v2.'):
    """Retain every trained tensor; clone only the shared gate, add zero bias."""
    result = OrderedDict(state)
    extra = [prefix + f'gate_{task}_head.{part}'
             for task in ('occ', 'pos') for part in ('weight', 'bias')]
    extra.append(prefix + 'task_horizon_bias')
    if any(key in result for key in extra):
        raise ValueError('Source already contains split/horizon gates')
    weight = state[prefix + 'gate_head.weight']
    bias = state[prefix + 'gate_head.bias']
    for task in ('occ', 'pos'):
        result[prefix + f'gate_{task}_head.weight'] = weight.clone()
        result[prefix + f'gate_{task}_head.bias'] = bias.clone()
    result[prefix + 'task_horizon_bias'] = bias.new_zeros(3, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('output')
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if source == output or output.exists():
        raise FileExistsError('Migration output must be a new file')
    sha = hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(block)
    checkpoint = torch.load(source, map_location='cpu')
    state = OrderedDict((key.removeprefix('module.'), value)
                        for key, value in checkpoint['state_dict'].items())
    migrated = migrate_shared_state(state)
    # Fresh optimizer/schedule by design. Do not carry old epoch/iter counters.
    meta = dict(source_checkpoint=str(source), source_sha256=sha.hexdigest(),
                source_epoch=checkpoint.get('meta', {}).get('epoch'),
                migration='shared gate -> three task gates + zero horizon bias')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        torch.save(dict(state_dict=migrated, meta=meta), stream)
    print(meta)
    print(f'Created {output}; original tensors unchanged; optimizer omitted')


if __name__ == '__main__':
    main()
