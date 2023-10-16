#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>

/* Based on the official PyTorch implementation of NMS using CUDA from:
https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cuda/nms_kernel.cu
 * Official code was changed by Felix Sattler 2023 to include:
 * - Mean and variance calculations for every kept bounding boxes over all overlapping candidates
 * - Coalesced memory accesses for the nms map kernel in CUDA & minor tweaks
 * - Modifications to compute the number of overlapping candidates per kept bounding box index
*/

// Hard-coded maximum. Increase if needed.
#define MAX_COL_BLOCKS 1000

#define DIVUP(m, n) (((m)+(n)-1) / (n))
#define PARENT_INDEX(x) ((x) - 1)

int64_t const threadsPerBlock = sizeof(unsigned long long) * 8;
int64_t const threadsPerBlockLinear = 256;

template<typename T, typename Ts>
__device__ inline bool devIoU(const T &a, const T &b, const float threshold) {
    Ts left = max(a.x, b.x), right = min(a.z, b.z);
    Ts top = max(a.y, b.y), bottom = min(a.w, b.w);
    Ts width = max(right - left, (Ts) 0), height = max(bottom - top, (Ts) 0);
    using acc_T = at::acc_type<Ts, /*is_cuda=*/true>;
    acc_T interS = (acc_T) width * height;
    acc_T Sa = ((acc_T) a.z - a.x) * (a.w - a.y);
    acc_T Sb = ((acc_T) b.z - b.x) * (b.w - b.y);
    return (interS / (Sa + Sb - interS)) > threshold;
}

template<typename T>
__global__ void nms_map_impl(const int64_t n_boxes,
                             const T nms_overlap_thresh,
                             const T *dev_boxes,
                             const int64_t *idx,
                             int64_t *dev_mask) {
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    if (row_start > col_start)
        return;

    const int row_size =
            min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size =
            min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

    // We can coalesce this load into a single float4 or double4
    __shared__ float4 block_boxes[threadsPerBlock];
    if (threadIdx.x < col_size) {
        block_boxes[threadIdx.x] = *reinterpret_cast<const float4 *>(&dev_boxes[
                idx[(threadsPerBlock * col_start + threadIdx.x)] * 4]);
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
        const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
        const float4 cur_box = *reinterpret_cast<const float4 *>(dev_boxes + idx[cur_box_idx] * 4);
        int i = 0;
        unsigned long long t = 0;
        int start = 0;
        if (row_start == col_start) {
            start = threadIdx.x + 1;
        }
        for (i = start; i < col_size; i++) {
            if (devIoU<float4, float>(cur_box, block_boxes[i]/* + i * 4*/, nms_overlap_thresh)) {
                t |= 1ULL << i;
            }
        }
        const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
        dev_mask[cur_box_idx * col_blocks + col_start] = t;
    }
}

template<>
__global__ void nms_map_impl<double>(const int64_t n_boxes,
                                     const double nms_overlap_thresh,
                                     const double *dev_boxes,
                                     const int64_t *idx,
                                     int64_t *dev_mask) {
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    if (row_start > col_start)
        return;

    const int row_size =
            min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size =
            min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

    // We can coalesce this load into a single float4 or double4
    __shared__ double4 block_boxes[threadsPerBlock];
    if (threadIdx.x < col_size) {
        block_boxes[threadIdx.x] = *reinterpret_cast<const double4 *>(&dev_boxes[
                idx[(threadsPerBlock * col_start + threadIdx.x)] * 4]);
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
        const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
        const double4 cur_box = *reinterpret_cast<const double4 *>(dev_boxes + idx[cur_box_idx] * 4);
        int i = 0;
        unsigned long long t = 0;
        int start = 0;
        if (row_start == col_start) {
            start = threadIdx.x + 1;
        }
        for (i = start; i < col_size; i++) {
            if (devIoU<double4, double>(cur_box, block_boxes[i]/* + i * 4*/, nms_overlap_thresh)) {
                t |= 1ULL << i; // cap to 64 for debugging
            }
        }
        const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
        dev_mask[cur_box_idx * col_blocks + col_start] = t;
    }
}

/* The original code of this function was taken from:
 * https://github.com/gdlg/pytorch_nms/blob/master/src/nms_kernel.cu
 * Copyright (c) 2018, Grégoire Payen de La Garanderie, Durham University
 * It was modified by Felix Sattler, 2023 including the following modifications:
 * - index changed to avoid double visits of diagonal elements
 * - summing loop to collect how often a kept index is referenced
 */
__global__ void
nms_reduce_impl(const int boxes_num,
                const int col_blocks,
                int top_k,
                const int64_t *dets,
                const int64_t *idx,
                int64_t *keep,
                int64_t *parent_object_index,
                int64_t *parent_ref_count,
                int64_t *num_to_keep) {

    int64_t remv[MAX_COL_BLOCKS] = {0};
    int num_to_keep_ = 0;

    for (int i = 0; i < boxes_num; i++) {
        int nblock = i / threadsPerBlock;
        int inblock = i % threadsPerBlock;

        if (!(remv[nblock] & (1ULL << inblock))) {
            keep[num_to_keep_] = idx[i];

            const int64_t *p = &dets[0] + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++) {
                remv[j] |= p[j];
            }

            // 'i + 1' since 'i' is visited below anyway (i.e. self-intersection)
            for (int j = i + 1; j < boxes_num; j++) {
                int nblockj = j / threadsPerBlock;
                int inblockj = j % threadsPerBlock;

                if (p[nblockj] & (1ULL << inblockj)) {
                    parent_object_index[idx[j]] = num_to_keep_ + 1;
                }
            }
            parent_object_index[idx[i]] = num_to_keep_ + 1;
            num_to_keep_++;

            if (num_to_keep_ == top_k)
                break;
        }
    }

    // collect the number of times each parent is referenced
    for (int i = 0; i < boxes_num; i++) {
        parent_ref_count[PARENT_INDEX(parent_object_index[i])] += 1;
    }

    *num_to_keep = min(top_k, num_to_keep_);
}

template<typename T>
__global__ void nms_mean_impl(const int64_t parent_object_num,
                              const T *dev_boxes,
                              const int64_t *parent_object_index,
                              const int64_t *parent_object_count,
                              T *mean_per_parent) {
    __shared__ float4 mean_accm[threadsPerBlockLinear];  //local block memory cache

    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= parent_object_num) {
        return;
    }

    float inv_N = 1.0f / static_cast<float>(parent_object_count[PARENT_INDEX(parent_object_index[i])]);
    inv_N = isinf(inv_N) ? 0.0f : inv_N;

    // coalesced loads using float4 vector types
    const float4 boxes = *reinterpret_cast<const float4 *>(&dev_boxes[i * 4]);
    mean_accm[threadIdx.x] = {
            0.5f * (boxes.x + boxes.z) * inv_N,
            0.5f * (boxes.y + boxes.w) * inv_N,
            (boxes.z - boxes.x) * inv_N,
            (boxes.w - boxes.y) * inv_N
    };

    __syncthreads();

    // write (this is done by one thread)
    if (threadIdx.x == 0) {
        for (int j = 0; j < blockDim.x; j++) {
            const int k = j + blockIdx.x * blockDim.x;
            if (k < parent_object_num) {
                mean_per_parent[PARENT_INDEX(parent_object_index[k]) * 4 + 0] += mean_accm[j].x;
                mean_per_parent[PARENT_INDEX(parent_object_index[k]) * 4 + 1] += mean_accm[j].y;
                mean_per_parent[PARENT_INDEX(parent_object_index[k]) * 4 + 2] += mean_accm[j].z;
                mean_per_parent[PARENT_INDEX(parent_object_index[k]) * 4 + 3] += mean_accm[j].w;
            }
        }
    }
}

// Template specialization required for double4 vector types
template<>
__global__ void nms_mean_impl<double>(const int64_t parent_object_num,
                                      const double *dev_boxes,
                                      const int64_t *parent_object_index,
                                      const int64_t *parent_object_count,
                                      double *mean_per_parent) {
    __shared__ double4 mean_accm[threadsPerBlockLinear];  //local block memory cache

    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= parent_object_num) {
        return;
    }

    double inv_N = 1.0 / static_cast<double>(parent_object_count[PARENT_INDEX(parent_object_index[i])]);
    inv_N = isinf(inv_N) ? 0.0 : inv_N;

    const double4 boxes = *reinterpret_cast<const double4 *>(&dev_boxes[i * 4]);
    mean_accm[threadIdx.x] = {
            0.5 * (boxes.x + boxes.z) * inv_N,
            0.5 * (boxes.y + boxes.w) * inv_N,
            (boxes.z - boxes.x) * inv_N,
            (boxes.w - boxes.y) * inv_N
    };

    __syncthreads();

    // write (this is done by one thread)
    if (threadIdx.x == 0) {
        for (int j = 0; j < blockDim.x; j++) {
            const int k = j + blockIdx.x * blockDim.x;
            if (k < parent_object_num) {
                mean_per_parent[PARENT_INDEX(parent_object_index[k]) * 4 + 0] += mean_accm[j].x;
                mean_per_parent[PARENT_INDEX(parent_object_index[k]) * 4 + 1] += mean_accm[j].y;
                mean_per_parent[PARENT_INDEX(parent_object_index[k]) * 4 + 2] += mean_accm[j].z;
                mean_per_parent[PARENT_INDEX(parent_object_index[k]) * 4 + 3] += mean_accm[j].w;
            }
        }
    }
}

template<typename T>
__global__ void nms_var_impl(const int64_t parent_object_num,
                             const T *dev_boxes,
                             const int64_t *parent_object_index,
                             const int64_t *parent_object_count,
                             const T *mean_per_parent,
                             T *var_per_parent) {
    __shared__ float4 var_accm[threadsPerBlockLinear];  //local block memory cache

    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= parent_object_num) {
        return;
    }

    float inv_N = 1.0f / (static_cast<float>(parent_object_count[PARENT_INDEX(parent_object_index[i])]) - 1.0f);
    inv_N = isinf(inv_N) ? 0.0f : inv_N;

    const float4 boxes = *reinterpret_cast<const float4 *>(&dev_boxes[i * 4]);
    const float4 mean = *reinterpret_cast<const float4 *>(&mean_per_parent[PARENT_INDEX(parent_object_index[i]) * 4]);
    float4 tmp = {mean.x - 0.5f * (boxes.x + boxes.z),
                  mean.y - 0.5f * (boxes.y + boxes.w),
                  mean.z - (boxes.z - boxes.x),
                  mean.w - (boxes.w - boxes.y)};

    var_accm[threadIdx.x] = {tmp.x * tmp.x * inv_N,
                             tmp.y * tmp.y * inv_N,
                             tmp.z * tmp.z * inv_N,
                             tmp.w * tmp.w * inv_N};

    __syncthreads();

    // write (this is done by one thread)
    if (threadIdx.x == 0) {
        for (int j = 0; j < blockDim.x; j++) {
            const int k = j + blockIdx.x * blockDim.x;
            if (k < parent_object_num) {
                var_per_parent[PARENT_INDEX(parent_object_index[k]) * 4 + 0] += var_accm[j].x;
                var_per_parent[PARENT_INDEX(parent_object_index[k]) * 4 + 1] += var_accm[j].y;
                var_per_parent[PARENT_INDEX(parent_object_index[k]) * 4 + 2] += var_accm[j].z;
                var_per_parent[PARENT_INDEX(parent_object_index[k]) * 4 + 3] += var_accm[j].w;
            }
        }
    }
}

template<>
__global__ void nms_var_impl<double>(const int64_t parent_object_num,
                                     const double *dev_boxes,
                                     const int64_t *parent_object_index,
                                     const int64_t *parent_object_count,
                                     const double *mean_per_parent,
                                     double *var_per_parent) {
    __shared__ double4 var_accm[threadsPerBlockLinear];  //local block memory cache

    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= parent_object_num) {
        return;
    }

    double inv_N = 1.0 / (static_cast<double>(parent_object_count[PARENT_INDEX(parent_object_index[i])]) - 1.0f);
    inv_N = isinf(inv_N) ? 0.0 : inv_N;

    const double4 boxes = *reinterpret_cast<const double4 *>(&dev_boxes[i * 4]);
    const double4 mean = *reinterpret_cast<const double4 *>(&mean_per_parent[PARENT_INDEX(parent_object_index[i]) * 4]);
    double4 tmp = {mean.x - 0.5 * (boxes.x + boxes.z),
                   mean.y - 0.5 * (boxes.y + boxes.w),
                   mean.z - (boxes.z - boxes.x),
                   mean.w - (boxes.w - boxes.y)};

    var_accm[threadIdx.x] = {tmp.x * tmp.x * inv_N,
                             tmp.y * tmp.y * inv_N,
                             tmp.z * tmp.z * inv_N,
                             tmp.w * tmp.w * inv_N};

    __syncthreads();

    // write (this is done by one thread)
    if (threadIdx.x == 0) {
        for (int j = 0; j < blockDim.x; j++) {
            const int k = j + blockIdx.x * blockDim.x;
            if (k < parent_object_num) {
                var_per_parent[PARENT_INDEX(parent_object_index[k]) * 4 + 0] += var_accm[j].x;
                var_per_parent[PARENT_INDEX(parent_object_index[k]) * 4 + 1] += var_accm[j].y;
                var_per_parent[PARENT_INDEX(parent_object_index[k]) * 4 + 2] += var_accm[j].z;
                var_per_parent[PARENT_INDEX(parent_object_index[k]) * 4 + 3] += var_accm[j].w;
            }
        }
    }
}

std::vector<at::Tensor> nms_var_impl_cuda_forward(
        const at::Tensor &dets,
        const at::Tensor &scores,
        float nms_overlap_thresh,
        unsigned long top_k) {

    TORCH_CHECK(dets.is_cuda(), "dets must be a CUDA tensor")
    TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor")

    TORCH_CHECK(
            dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D")
    TORCH_CHECK(
            dets.size(1) == 4,
            "boxes should have 4 elements in dimension 1, got ",
            dets.size(1))
    TORCH_CHECK(
            scores.dim() == 1,
            "scores should be a 1d tensor, got ",
            scores.dim(),
            "D")
    TORCH_CHECK(
            dets.size(0) == scores.size(0),
            "boxes and scores should have same number of elements in ",
            "dimension 0, got ",
            dets.size(0),
            " and ",
            scores.size(0))

    if (dets.numel() == 0) {
        return {at::empty({0}, dets.options().dtype(at::kLong))};
    }

    auto idx = std::get<1>(scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));

    int dets_num = dets.size(0);

    const int col_blocks = DIVUP(dets_num, threadsPerBlock);

    AT_ASSERTM(col_blocks < MAX_COL_BLOCKS,
               "The number of column blocks must be less than MAX_COL_BLOCKS. Increase the MAX_COL_BLOCKS constant if needed.");

    auto longOptions = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong);
    auto mask = at::empty({dets_num * col_blocks}, dets.options().dtype(at::kLong));
    auto keep = at::zeros({dets_num}, longOptions);
    auto parent_object_index = at::zeros({dets_num}, longOptions);
    auto parent_object_ref_count = at::zeros({dets_num}, longOptions);
    auto num_to_keep = at::empty({}, longOptions);

    dim3 blocks(col_blocks, col_blocks);
    dim3 threads(threadsPerBlock);

    AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "nms_cuda_forward", ([&] {
        nms_map_impl<scalar_t><<<blocks, threads>>>(dets_num,
                                                    (scalar_t) nms_overlap_thresh,
                                                    dets.data_ptr<scalar_t>(),
                                                    idx.data_ptr<int64_t>(),
                                                    mask.data_ptr<int64_t>());
    }));

    nms_reduce_impl<<<1, 1>>>(dets_num, col_blocks, top_k,
                              mask.data_ptr<int64_t>(),
                              idx.data_ptr<int64_t>(),
                              keep.data_ptr<int64_t>(),
                              parent_object_index.data_ptr<int64_t>(),
                              parent_object_ref_count.data_ptr<int64_t>(),
                              num_to_keep.data_ptr<int64_t>());


    auto parent_object_mean = torch::zeros(num_to_keep.item<int>() * 4,
                                           torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));
    auto parent_object_var = torch::zeros(num_to_keep.item<int>() * 4,
                                          torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));

    blocks = {static_cast<unsigned int>(DIVUP(parent_object_index.size(0), threadsPerBlockLinear)), 1, 1};
    threads = {threadsPerBlockLinear, 1, 1};

    AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "nms_mean_impl", ([&] {
        nms_mean_impl<scalar_t><<<blocks, threads>>>(parent_object_index.size(0),
                                                     dets.data_ptr<scalar_t>(),
                                                     parent_object_index.data_ptr<int64_t>(),
                                                     parent_object_ref_count.data_ptr<int64_t>(),
                                                     parent_object_mean.data_ptr<scalar_t>());
    }));

    AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "indexed_var", ([&] {
        nms_var_impl<scalar_t><<<blocks, threads>>>(parent_object_index.size(0),
                                                    dets.data_ptr<scalar_t>(),
                                                    parent_object_index.data_ptr<int64_t>(),
                                                    parent_object_ref_count.data_ptr<int64_t>(),
                                                    parent_object_mean.data_ptr<scalar_t>(),
                                                    parent_object_var.data_ptr<scalar_t>());
    }));

    return {keep, num_to_keep, parent_object_var};
}
