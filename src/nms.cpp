#include <torch/extension.h>
#include <torch/types.h>
/*
std::vector<at::Tensor> nms_impl_cpu_forward(
        at::Tensor &dets,
        at::Tensor &scores,
        float nms_overlap_thresh,
        unsigned long top_k);
*/
std::vector<at::Tensor> nms_var_impl_cuda_forward(
        const at::Tensor &dets,
        const at::Tensor &scores,
        float nms_overlap_thresh,
        unsigned long top_k);

std::vector<at::Tensor> nms_var_forward(
        at::Tensor &dets,
        at::Tensor &scores,
        float thresh,
        unsigned long top_k) {
    return nms_var_impl_cuda_forward(dets, scores, thresh, top_k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nms_with_variance", &nms_var_forward, "NMS with variance calculation, forward pass");
}

