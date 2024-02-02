import os
import subprocess
from setuptools import setup

__version__ = '0.1.0'


def check_torch_cuda():
    args = {"channels": [],
            "packages": []}
    has_cuda = False
    has_pytorch = False

    # check if PyTorch is installed and has CUDA support
    try:
        import torch
        if not torch.cuda.is_available():
            raise ModuleNotFoundError
        has_pytorch = True
    except ModuleNotFoundError:
        print("PyTorch is not installed! Attempting to install in virtual env...")
        args["channels"].extend(["-c", "pytorch", "-c", "nvidia"])
        args["packages"].extend(["pytorch", "torchvision", "torchaudio", "pytorch-cuda"])
    # check if nvcc (part of cuda-toolkit) is installed
    try:
        subprocess.call(args=["nvcc", "--version"])
        has_cuda = True
    except FileNotFoundError:
        print("CUDA is not installed. Attempting to install in virtual env...")
        args["channels"].extend(["-c", "nvidia"])
        args["packages"].extend(["cuda-toolkit"])

    # Check if the $CONDA_PREFIX environment variable is set.
    # If it's not set, check if PyTorch and CUDA are globally available, if not exit.
<<<<<<< HEAD
    if not os.environ.get('CONDA_PREFIX'):
=======
    if not os.environ['CONDA_PREFIX']:
>>>>>>> 1a0dbe8cae0b1220ed5c5a121101e0bac8e5f2be
        print(
            'Please specify the CONDA_PREFIX environment variable. '
            'It should point to you environment folder, e.g. "/home/<user>/conda/env/<env_name>"')
        if not has_cuda:
            print('Cuda-Toolkit not found. Please install from: '
                  'https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html')
            exit(-1)
        if not has_pytorch:
            print('PyTorch with CUDA not found. Please install from: '
                  'https://pytorch.org/get-started/locally/')
            exit(-1)
        return
    # If it's set, then we can try to install missing components through conda.
    else:
        # install the packages in the conda environment
        conda_args = ["conda", "install", "--yes"]
        args["packages"].extend(args["channels"])
        conda_args.extend(args["packages"])
        subprocess.call(args=conda_args)


def main():
    PACKAGE_NAME = 'nms_var'

    # check if PyTorch & Cuda-Toolkit is installed
    check_torch_cuda()

    # Set up the dirs
    setup_dir = os.path.dirname(os.path.realpath(__file__))
    src_dir = os.path.join(setup_dir, os.path.join('src', PACKAGE_NAME))

    # Pull the repo for Pybind11
    pybind11_dir = os.path.join(os.path.join(src_dir, os.path.join('extern', 'pybind11')))
    if not os.path.exists(pybind11_dir):
        subprocess.call(args=["git", "clone", "https://github.com/pybind/pybind11.git", "{}".format(pybind11_dir)])

    from torch.utils import cpp_extension

    setup(name=PACKAGE_NAME,
          version=__version__,
          author='Felix Sattler',
          author_email='felix.sattler@dlr.de',
          description="Non maximum suppression for PyTorch with added variance computation over scores and bounding boxes.",
          packages=[PACKAGE_NAME],
          package_dir={'': 'src'},
          ext_modules=[
              cpp_extension.CUDAExtension(name='{}.nms_with_variance_impl'.format(PACKAGE_NAME),
                                          sources=['src/nms.cpp',
                                                   'src/cuda/nms_var_kernel.cu'],
                                          extra_compile_args={'cxx': ['-g'],
                                                              'nvcc': ['-O2']})
          ],
          cmdclass={
              'build_ext': cpp_extension.BuildExtension
          },
          url='https://github.com/DLR-MI/nms_var',
          python_requires='>=3.8',
          keywords=['nms', 'variance', 'object detection']
          )


if __name__ == "__main__":
    main()
