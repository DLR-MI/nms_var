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
#include <torch/types.h>
#include <iostream>

std::vector<at::Tensor> nms_cuda_forward(
        at::Tensor &dets,
        at::Tensor &scores,
        float nms_overlap_thresh,
        unsigned long top_k);

std::vector<at::Tensor> nms_forward(
        at::Tensor &dets,
        at::Tensor &scores,
        float thresh,
        unsigned long top_k) {

    TORCH_CHECK(dets.is_cuda(), "dets must be a CUDA tensor");
    TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");

    TORCH_CHECK(
            dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
    TORCH_CHECK(
            dets.size(1) == 4,
            "boxes should have 4 elements in dimension 1, got ",
            dets.size(1));
    TORCH_CHECK(
            scores.dim() == 1,
            "scores should be a 1d tensor, got ",
            scores.dim(),
            "D");
    TORCH_CHECK(
            dets.size(0) == scores.size(0),
            "boxes and scores should have same number of elements in ",
            "dimension 0, got ",
            dets.size(0),
            " and ",
            scores.size(0))

    return nms_cuda_forward(dets, scores, thresh, top_k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nms_forward", &nms_forward, "NMS");
}

