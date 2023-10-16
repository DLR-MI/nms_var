# Implementation of NMS (non maximum suppression) with variance as a PyTorch extension.

This repository includes an implementation of NMS with variance based on PyTorch ***. 
CPU and CUDA backends are supported and automatically chosen based on the device of the provided tensors.

This code is based on the official [Torchvision](https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cuda/nms_kernel.cu) implementation of NMS and includes code from an older implementation by [Grégoire Payen de La Garanderie](https://github.com/gdlg/pytorch_nms/blob/master/src/nms_kernel.cu).

## Installation

You can install it either via Python directly or using pip.

### Python
```
python setup.py install --install-lib <your_install_location>
```
### PIP
```
pip install --target=<your_custom_target> <repository_link>
```

## Usage

After building, you can use this package just as any other ordinary package. See ```__init__.py```for docs.

```
from nms_with_variance import nms_with_variance

kept_indices, num_kept_indices, var_per_kept = nms_with_variance(boxes, scores, overlap=.5, top_k=200)
```

