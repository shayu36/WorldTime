"""Generate NEW past-only ego artifact; no GPU, GT loading, cache or training."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import pickle

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'causal_ego', ROOT / 'mmdet3d/datasets/causal_ego.py')
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-info', default='data/nuscenes/bevdetv2-nuscenes_infos_train.pkl')
    parser.add_argument('--val-info', default='data/nuscenes/bevdetv2-nuscenes_infos_val.pkl')
    parser.add_argument('--output-dir', default='data/ego_causal_v1')
    args = parser.parse_args()
    target = Path(args.output_dir)
    if target.exists():
        parser.error('Output directory exists; refusing to overwrite: ' + str(target))
    records, source_hashes, counts = {}, {}, {}
    split_tokens, split_scenes = {}, {}
    for split, name in [('train', args.train_info), ('val', args.val_info)]:
        source_hashes[split] = dict(path=name, sha256=ce.file_sha256(name))
        with open(name, 'rb') as stream:
            infos = pickle.load(stream)['infos']
        scenes = defaultdict(list)
        # Whitelist pose metadata, never gt_trajectory/annotations/cmd/a/v.
        fields = ('token', 'scene_token', 'timestamp', 'ego2global_rotation',
                  'ego2global_translation', 'lidar2ego_rotation', 'lidar2ego_translation')
        for info in infos:
            record = {key: info[key] for key in fields}
            scenes[record['scene_token']].append(record)
        del infos
        split_tokens[split] = set()
        split_scenes[split] = set(scenes)
        for scene in scenes.values():
            scene.sort(key=lambda r: r['timestamp'])
            for i, current in enumerate(scene):
                token = current['token']
                if token in records:
                    raise ValueError('Duplicate token across inputs: ' + token)
                records[token] = ce.build_record(current, scene[max(0, i-4):i])
                split_tokens[split].add(token)
        counts[split] = dict(samples=len(split_tokens[split]), scenes=len(scenes))
        print(split, counts[split], flush=True)
    if split_scenes['train'] & split_scenes['val'] or split_tokens['train'] & split_tokens['val']:
        raise ValueError('Train/val split overlap')
    manifest = dict(
        schema_version=ce.SCHEMA, protocol=ce.PROTOCOL, inputs=source_hashes,
        counts=counts, fields=list(ce.FIELDS), network_channels=21,
        units=dict(a='ego-frame xyz m/s^2', v='ego-frame xyz m/s',
                   x='negative cumulative past lidar-step dx/dy/dyaw: m/m/rad',
                   cmd='unknown=[0,0,0], not straight=[0,1,0]'),
        generator_sha256=ce.file_sha256(__file__),
        implementation_sha256=ce.file_sha256(ROOT / 'mmdet3d/datasets/causal_ego.py'))
    artifact = dict(schema_version=ce.SCHEMA, protocol=ce.PROTOCOL,
                    manifest=manifest, records=records)
    target.mkdir(parents=True, exist_ok=False)
    path = target / 'ego_inputs.pkl'
    with path.open('xb') as stream:
        pickle.dump(artifact, stream, protocol=4)
    manifest['artifact_sha256'] = ce.file_sha256(path)
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
