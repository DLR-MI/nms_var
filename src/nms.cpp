#include <torch/extension.h>
#include <torch/types.h>

std::vector<at::Tensor> nms_var_forward(
        const at::Tensor &dets,
        const at::Tensor &scores,
        float nms_overlap_thresh,
        unsigned long top_k);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nms_with_variance", &nms_var_forward, "NMS with variance calculation, forward pass");
}

