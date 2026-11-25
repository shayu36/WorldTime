#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python -c 'import sys; assert sys.version_info[:2] == (3,9), "Use Python 3.9"'
python -m pip install 'pip==25.3' 'setuptools==69.5.1' 'wheel==0.45.1'
python -m pip install 'torch==2.0.1+cu118' 'torchvision==0.15.2+cu118' --index-url https://download.pytorch.org/whl/cu118
# The published wheel exactly matches this run's mmcv._ext (Py3.9/Torch2.0.1/cu118).
# Official cu118/torch2.0 indexes do not contain mmcv-full 1.7.0.
python -m pip install artifacts/wheels/mmcv_full-1.7.0-cp39-cp39-linux_x86_64.whl 'numpy==1.23.4' 'opencv-python-headless==4.11.0.86' 'yapf==0.40.1'
python -m pip install 'torch-scatter==2.1.2' --only-binary=torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
python -m pip install -c requirements/constraints-py39-cu118.txt -r requirements.txt
python -m pip install -e . --no-build-isolation --no-deps
python -m pip check
python tools/reproduce/verify.py --environment-only
