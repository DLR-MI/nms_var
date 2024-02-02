# NMSVar: Non-maximum suppression with variance 

This repository includes an implementation of NMS with variance for PyTorch.
Computing the variance over possible bounding box and score candidates adds information about the uncertainty of the NMS process.
This becomes important when using the predicted bounding box in subsequent steps (e.g. tracking) to compensate for high uncertainty predictions.
There is only a very slight performance penalty when computing variances due to native C++ and CUDA code.

**Note**: Currently only PyTorch with CUDA is supported as backend. Stay tuned for a CPU version!

This code is based on the official [Torchvision](https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cuda/nms_kernel.cu) implementation of NMS and includes code from an older implementation by [Grégoire Payen de La Garanderie](https://github.com/gdlg/pytorch_nms/blob/master/src/nms_kernel.cu).

## Installation
You can install the extension using PIP. If you do not want to use Conda (or Mamba), then the installation script automatically searches if PyTorch and CUDA are installed.
If you call PIP from within Conda all required packages are installed automatically.

### PIP
```bash
pip install git+https://github.com/DLR-MI/nms_var.git
```

## Usage

```python
import nms_var

# Returns the kept indices (not sorted) and the variance per kept index
kept_indices, var_per_kept = nms_var.nms(boxes, scores, overlap=.5, top_k=200)
```

## Test
You can test the implementation using a script provided in this repo.
The test compares the predicted bounding boxes and timings against the official Torchvision implementation.
```bash
git clone https://github.com/DLR-MI/nms_var.git nms_var
python nms_var/test/test_nms.py
```
