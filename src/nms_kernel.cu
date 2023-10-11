/* Copyright (c) 2018, Grégoire Payen de La Garanderie, Durham University
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * 
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * 
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>

// Hard-coded maximum. Increase if needed.
#define MAX_COL_BLOCKS 1000

#define DIVUP(m, n) (((m)+(n)-1) / (n))
int64_t const threadsPerBlock = sizeof(unsigned long long) * 8;

// The functions below originates from Fast R-CNN
// See https://github.com/rbgirshick/py-faster-rcnn
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License
// Written by Shaoqing Ren

template<typename scalar_t>
__device__ inline scalar_t devIoU(scalar_t const *const a, scalar_t const *const b) {
    scalar_t left = max(a[0], b[0]), right = min(a[2], b[2]);
    scalar_t top = max(a[1], b[1]), bottom = min(a[3], b[3]);
    scalar_t width = max(right - left, 0.f), height = max(bottom - top, 0.f);
    scalar_t interS = width * height;
    scalar_t Sa = (a[2] - a[0]) * (a[3] - a[1]);
    scalar_t Sb = (b[2] - b[0]) * (b[3] - b[1]);
    return interS / (Sa + Sb - interS);
}

template<typename scalar_t>
__global__ void nms_kernel(const int64_t n_boxes, const scalar_t nms_overlap_thresh,
                           const scalar_t *dev_boxes, const int64_t *idx, int64_t *dev_mask) {
    const int64_t row_start = blockIdx.y;
    const int64_t col_start = blockIdx.x;

    const int row_size =
            min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size =
            min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

    __shared__ scalar_t block_boxes[threadsPerBlock * 4];
    if (threadIdx.x < col_size) {
        block_boxes[threadIdx.x * 4 + 0] =
                dev_boxes[idx[(threadsPerBlock * col_start + threadIdx.x)] * 4 + 0];
        block_boxes[threadIdx.x * 4 + 1] =
                dev_boxes[idx[(threadsPerBlock * col_start + threadIdx.x)] * 4 + 1];
        block_boxes[threadIdx.x * 4 + 2] =
                dev_boxes[idx[(threadsPerBlock * col_start + threadIdx.x)] * 4 + 2];
        block_boxes[threadIdx.x * 4 + 3] =
                dev_boxes[idx[(threadsPerBlock * col_start + threadIdx.x)] * 4 + 3];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
        const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
        const scalar_t *cur_box = dev_boxes + idx[cur_box_idx] * 4;
        int i = 0;
        unsigned long long t = 0;
        int start = 0;
        if (row_start == col_start) {
            start = threadIdx.x + 1;
        }
        for (i = start; i < col_size; i++) {
            if (devIoU(cur_box, block_boxes + i * 4) > nms_overlap_thresh) {
                t |= 1ULL << i;
            }
        }
        const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
        dev_mask[cur_box_idx * col_blocks + col_start] = t;
    }
}


__global__ void
nms_collect(const int64_t boxes_num, const int64_t col_blocks, int64_t top_k, const int64_t *idx, const int64_t *mask,
            int64_t *keep, int64_t *parent_object_index, int64_t *num_to_keep) {
    int64_t remv[MAX_COL_BLOCKS];
    int64_t num_to_keep_ = 0;

    for (int i = 0; i < col_blocks; i++) {
        remv[i] = 0;
    }

    for (int i = 0; i < boxes_num; ++i) {
        parent_object_index[i] = 0;
    }

    for (int i = 0; i < boxes_num; i++) {
        int nblock = i / threadsPerBlock;
        int inblock = i % threadsPerBlock;


        if (!(remv[nblock] & (1ULL << inblock))) {
            int64_t idxi = idx[i];
            keep[num_to_keep_] = idxi;
            const int64_t *p = &mask[0] + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++) {
                remv[j] |= p[j];
            }
            for (int j = i; j < boxes_num; j++) {
                int nblockj = j / threadsPerBlock;
                int inblockj = j % threadsPerBlock;
                if (p[nblockj] & (1ULL << inblockj))
                    parent_object_index[idx[j]] = num_to_keep_ + 1;
            }
            parent_object_index[idx[i]] = num_to_keep_ + 1;

            num_to_keep_++;

            if (num_to_keep_ == top_k)
                break;
        }
    }

    // Initialize the rest of the keep array to avoid uninitialized values.
    for (int i = num_to_keep_; i < boxes_num; ++i)
        keep[i] = 0;

    *num_to_keep = min(top_k, num_to_keep_);
}

template<typename scalar_t>
__global__ void indexed_variance(const int64_t parent_object_index_num,
                                 const int64_t *parent_object_index,
                                 const scalar_t *dev_boxes,
                                 scalar_t *mean) {

}


#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

#define PARENT_INDEX(x) ((x) - 1)

std::vector<at::Tensor> nms_cuda_forward(
        at::Tensor boxes,
        at::Tensor idx,
        float nms_overlap_thresh,
        unsigned long top_k) {

    const auto boxes_num = boxes.size(0);

    const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

    AT_ASSERTM(col_blocks < MAX_COL_BLOCKS,
               "The number of column blocks must be less than MAX_COL_BLOCKS. Increase the MAX_COL_BLOCKS constant if needed.");

    auto longOptions = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong);
    auto mask = at::empty({boxes_num * col_blocks}, longOptions);

    dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
                DIVUP(boxes_num, threadsPerBlock));
    dim3 threads(threadsPerBlock);

    CHECK_CONTIGUOUS(boxes);
    CHECK_CONTIGUOUS(idx);
    CHECK_CONTIGUOUS(mask);

    AT_DISPATCH_FLOATING_TYPES(boxes.type(), "nms_cuda_forward", ([&] {
        nms_kernel<<<blocks, threads>>>(boxes_num,
                                        (scalar_t) nms_overlap_thresh,
                                        boxes.data_ptr<scalar_t>(),
                                        idx.data_ptr<int64_t>(),
                                        mask.data_ptr<int64_t>());
    }));

    auto keep = at::empty({boxes_num}, longOptions);
    auto parent_object_index = at::empty({boxes_num}, longOptions);
    auto num_to_keep = at::empty({}, longOptions);

    nms_collect<<<1, 1>>>(boxes_num, col_blocks, top_k,
                          idx.data_ptr<int64_t>(),
                          mask.data_ptr<int64_t>(),
                          keep.data_ptr<int64_t>(),
                          parent_object_index.data_ptr<int64_t>(),
                          num_to_keep.data_ptr<int64_t>());

    auto num_to_keep_cpu = num_to_keep.to(torch::kCPU);
    auto parent_object_index_cpu = parent_object_index.to(torch::kCPU);
    auto boxes_cpu = boxes.to(torch::kCPU);

    const auto num_to_keep_size = num_to_keep_cpu.data_ptr<int64_t>()[0];
    const auto floatOptions = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat);
    auto samples_N = torch::zeros(num_to_keep_size, torch::TensorOptions().device(torch::kCPU).dtype(torch::kLong));
    auto mean = torch::zeros(num_to_keep_size * 4, floatOptions);
    auto variance = torch::zeros(num_to_keep_size * 4, floatOptions);

    auto parent_obj_idx_ptr = parent_object_index_cpu.data_ptr<int64_t>();
    auto boxes_ptr = boxes_cpu.data_ptr<float>();
    auto sample_ptr = samples_N.data_ptr<int64_t>();
    auto mean_ptr = mean.data_ptr<float>();
    auto var_ptr = variance.data_ptr<float>();

    // Compute number of samples
    for (int i = 0; i < parent_object_index_cpu.size(0); i++) {
        sample_ptr[PARENT_INDEX(parent_obj_idx_ptr[i])] += 1;
    }

    //FIXME the last part can also be done in CUDA for maximum acceleration
    // Compute mean
    for (int i = 0; i < parent_object_index_cpu.size(0); i++) {
        const int index = PARENT_INDEX(parent_obj_idx_ptr[i]) * 4;
        float inv_N = 1.0f / sample_ptr[PARENT_INDEX(parent_obj_idx_ptr[i])];
        inv_N = std::isinf(inv_N) ? 0.0f : inv_N;

        mean_ptr[index + 0] += 0.5f * (boxes_ptr[i * 4 + 0] + boxes_ptr[i * 4 + 2]) * inv_N;
        mean_ptr[index + 1] += 0.5f * (boxes_ptr[i * 4 + 1] + boxes_ptr[i * 4 + 3]) * inv_N;
        mean_ptr[index + 2] += (boxes_ptr[i * 4 + 2] - boxes_ptr[i * 4 + 0]) * inv_N;
        mean_ptr[index + 3] += (boxes_ptr[i * 4 + 3] - boxes_ptr[i * 4 + 1]) * inv_N;
    }

    // Compute variance
    const float correction = 1.0f;
    for (int i = 0; i < parent_object_index_cpu.size(0); i++) {
        const int index = PARENT_INDEX(parent_obj_idx_ptr[i]) * 4;
        float inv_N = 1.0f / fmaxf(0.0f, sample_ptr[PARENT_INDEX(parent_obj_idx_ptr[i])] - correction);
        inv_N = std::isinf(inv_N) ? 0.0f : inv_N;

        auto xavg = mean_ptr[index + 0] - 0.5f * (boxes_ptr[i * 4 + 0] + boxes_ptr[i * 4 + 2]);
        auto yavg = mean_ptr[index + 1] - 0.5f * (boxes_ptr[i * 4 + 1] + boxes_ptr[i * 4 + 3]);
        auto x2x1 = mean_ptr[index + 2] - (boxes_ptr[i * 4 + 2] - boxes_ptr[i * 4 + 0]);
        auto y2y1 = mean_ptr[index + 3] - (boxes_ptr[i * 4 + 3] - boxes_ptr[i * 4 + 1]);
        var_ptr[index + 0] += xavg * xavg * inv_N;
        var_ptr[index + 1] += yavg * yavg * inv_N;
        var_ptr[index + 2] += x2x1 * x2x1 * inv_N;
        var_ptr[index + 3] += y2y1 * y2y1 * inv_N;
    }

    return {keep, num_to_keep, parent_object_index, variance};
}

