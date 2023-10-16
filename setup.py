from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='nms_with_variance', packages=['nms_with_variance'],
      package_dir={'': 'src'},
      ext_modules=[
            cpp_extension.CUDAExtension('nms_with_variance.details',
                                        ['src/nms.cpp', 'src/cuda/nms_var_kernel.cu'])
      ],
      cmdclass={
            'build_ext': cpp_extension.BuildExtension
      })
