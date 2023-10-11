project(PYTORCH_NMS_VARIANCE CXX CUDA)
cmake_minimum_required(VERSION 3.16)

add_library(pytorch_nms_variance src/nms_kernel.cu)