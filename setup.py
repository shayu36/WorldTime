"""Build the two repository-specific CUDA extensions against installed PyTorch."""
import os
from pathlib import Path
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent.resolve()
os.chdir(ROOT)
bev = 'mmdet3d/ops/bev_pool_v2'
msmv = 'mmdet3d/models/sparsedetectors/csrc/msmv_sampling'
setup(name='memanchorworld', version='0.1.0', python_requires='>=3.9,<3.10',
      description='Memory-adapted SparseWorld with supervised planning fusion',
      packages=find_packages(include=['mmdet3d', 'mmdet3d.*']),
      ext_modules=[
          CUDAExtension('mmdet3d.ops.bev_pool_v2.bev_pool_v2_ext',
                        [f'{bev}/src/bev_pool.cpp', f'{bev}/src/bev_pool_cuda.cu'],
                        extra_compile_args={'cxx':['-O3'], 'nvcc':['-O3']}),
          CUDAExtension('mmdet3d.models.sparsedetectors.csrc._msmv_sampling_cuda',
                        [f'{msmv}/msmv_sampling.cpp',f'{msmv}/msmv_sampling_forward.cu',f'{msmv}/msmv_sampling_backward.cu'],
                        include_dirs=[str(ROOT/msmv)],
                        extra_compile_args={'cxx':['-O3'], 'nvcc':['-O3']})],
      cmdclass={'build_ext':BuildExtension})
