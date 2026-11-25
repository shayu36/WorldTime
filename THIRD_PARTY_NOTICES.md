# Third-party source and binary notices

This repository builds on SparseWorld (https://github.com/MSunDYY/SparseWorld),
MMDetection3D, MMDetection, MMCV, MMSegmentation, BEVDet and OPUS.
Original copyright and source notices in their files are retained. Added release
packaging does not relicense upstream material or grant dataset rights.

The exact planning evaluator under `AD-MLP/deps/stp3/stp3/planning_metrics.py`
retains its OpenDriveLab/UniAD header and was taken from the server's ST-P3/AD-MLP
evaluation tree. Its hash is recorded by the released Baseline and seed17 results.
The two required `utils` files are retained as source, not replaced with a new metric.
`utils/tools.py` includes the NVIDIA lift-splat-shoot source notice and its license
reference https://github.com/nv-tlabs/lift-splat-shoot. Consult that upstream license
before redistribution or use beyond its granted terms. This release does not claim
that all third-party source is under one blanket project license.

The bundled `mmcv_full-1.7.0-cp39-cp39-linux_x86_64.whl` is MMCV (Apache-2.0);
its LICENSE and LICENSES.md are included inside the wheel. It is provided because
the official cu118/torch2.0 binary index does not provide version 1.7.0.
All other Python packages are installed from their upstream distributions under
their own licenses. No nuScenes/Occ3D data, trajectory labels, images or Memory
cache is redistributed here.

License texts referenced above are retained under `third_party/licenses/`.
