//
// Created by satt_fe on 10/16/23.
//

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include <cmath>

template<typename T>
inline bool devIoU(T const *const a, T const *const b, const float threshold) {
    T left = max(a[0], b[0]), right = min(a[2], b[2]);
    T top = max(a[1], b[1]), bottom = min(a[3], b[3]);
    T width = max(right - left, (T) 0), height = max(bottom - top, (T) 0);
    using acc_T = at::acc_type<T, /*is_cuda=*/true>;
    acc_T interS = (acc_T) width * height;
    acc_T Sa = ((acc_T) a[2] - a[0]) * (a[3] - a[1]);
    acc_T Sb = ((acc_T) b[2] - b[0]) * (b[3] - b[1]);
    return (interS / (Sa + Sb - interS)) > threshold;
}

template<typename T>
static void nms_kernel(const int64_t n_boxes,
                const T nms_overlap_thresh,
                const T *dev_boxes,
                const int64_t *idx,
                int64_t *dev_mask) {
    const int64_t row_start = blockIdx.y;
    const int64_t col_start = blockIdx.x;

    // we can try to use a ulonglong2 instead of ulong to support 128 threads

    const int row_size =
            min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size =
            min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

    __shared__ T block_boxes[threadsPerBlock * 4];
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
    __syncthreads(); // barrier

    if (threadIdx.x < row_size) {
        const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
        const T *cur_box = dev_boxes + idx[cur_box_idx] * 4;
        int i = 0;
        //unsigned long long t[2] = {0, 0};
        unsigned long long t = 0;
        int start = 0;
        if (row_start == col_start) {
            start = threadIdx.x + 1;
        }
        for (i = start; i < col_size; i++) {
            if (devIoU(cur_box, block_boxes + i * 4, nms_overlap_thresh)) {
                t |= 1ULL << i; // cap to 64 for debugging
                //t[i / ULLBYTES] |= 1ULL << i;
            }
        }
        const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
        //memcpy(&dev_mask[cur_box_idx * col_blocks + col_start], t, sizeof(unsigned long long) * 2);
        dev_mask[cur_box_idx * col_blocks + col_start] = t;//t[0];
    }
}

static void nms_collect(const int64_t boxes_num,
                        const int64_t col_blocks,
                        int64_t top_k,
                        const int64_t *dets,
                        const int64_t *idx,
                        int64_t *keep,
                        int64_t *parent_object_index,
                        int64_t *parent_ref_count,
                        int64_t *num_to_keep) {

    int64_t remv[MAX_COL_BLOCKS] = {0};
    int64_t num_to_keep_ = 0;

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

                parent_object_index[idx[j]] = p[nblockj] & (1ULL << inblockj) ? num_to_keep_ + 1 : 0;

                /*
                if (p[nblockj] & (1ULL << inblockj)) {
                    parent_object_index[idx[j]] = num_to_keep_ + 1;
                }*/
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

std::vector<at::Tensor> nms_impl_forward(
        at::Tensor &dets,
        at::Tensor &scores,
        float nms_overlap_thresh,
        unsigned long top_k) {

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

    auto num_to_keep_cpu = num_to_keep.to(torch::kCPU);
    auto parent_object_index_cpu = parent_object_index.to(torch::kCPU);
    auto boxes_cpu = dets.to(torch::kCPU);

    const auto floatOptions = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat);
    const auto num_to_keep_size = num_to_keep_cpu.data_ptr<int64_t>()[0];

    /* This... */
    auto samples_N = torch::zeros(num_to_keep_size, torch::TensorOptions().device(torch::kCPU).dtype(torch::kLong));
    auto mean = torch::zeros(num_to_keep_size * 4, floatOptions);
    auto variance = torch::zeros(num_to_keep_size * 4, floatOptions);

    auto parent_obj_idx_ptr = parent_object_index_cpu.data_ptr<int64_t>();
    auto boxes_ptr = boxes_cpu.data_ptr<float>();
    auto sample_ptr = samples_N.data_ptr<int64_t>();
    auto mean_ptr = mean.data_ptr<float>();
    auto var_ptr = variance.data_ptr<float>();
    /* ...until here is irrelevant fr GPU*/

    // Compute number of samples, can be integrated with NMS computation on CPU
    for (int i = 0; i < parent_object_index_cpu.size(0); i++) {
        sample_ptr[PARENT_INDEX(parent_obj_idx_ptr[i])] += 1;
    }

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

    return {keep, num_to_keep, parent_object_index, variance, var_per_parent};
}