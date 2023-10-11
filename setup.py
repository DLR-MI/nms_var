from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name='nms_variance', packages=['nms_variance'],
      package_dir={'': 'src'},
      ext_modules=[CUDAExtension('nms_variance.details', ['src/nms.cpp', 'src/nms_kernel.cu'])],
      cmdclass={'build_ext': BuildExtension})
