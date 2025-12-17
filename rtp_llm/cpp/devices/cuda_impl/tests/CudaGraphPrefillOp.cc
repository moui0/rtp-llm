#include "rtp_llm/cpp/devices/cuda_impl/tests/CudaGraphPrefillOp.h"
namespace cuda_graph {
using namespace rtp_llm;

void CudaGraphPrefillOp::init(py::object       py_instance,
                              int64_t          max_context_batch_size,
                              int64_t          hidden_size,
                              int64_t          max_seq_len,
                              int64_t          tokens_per_block,
                              int64_t          max_prefill_cuda_graph_len,
                              std::vector<int> prefill_capture_seq_lens) {
    cuda_graph_runner_ = createCudaGraphRunner(std::move(py_instance),
                                               max_context_batch_size,
                                               hidden_size,
                                               max_seq_len,
                                               tokens_per_block,
                                               prefill_capture_seq_lens);
    cuda_graph_runner_->setMaxPrefillCudaGraphLen(max_prefill_cuda_graph_len);
    cuda_graph_runner_->initCapture();
}

int CudaGraphPrefillOp::getCurrentRealGraphSize() {
    return cuda_graph_runner_->getCurrentRealGraphBs();
}

CudaGraphRunnerPtr CudaGraphPrefillOp::createCudaGraphRunner(py::object       py_instance,
                                                             int64_t          max_context_batch_size,
                                                             int64_t          hidden_size,
                                                             int64_t          max_seq_len,
                                                             int64_t          tokens_per_block,
                                                             std::vector<int> prefill_capture_seq_lens) {
    DeviceInitParams params;
    DeviceBase*      device                              = rtp_llm::DeviceFactory::getDefaultDevice();
    params.hw_kernel_config.enable_cuda_graph            = true;
    params.fifo_scheduler_config.max_context_batch_size  = max_context_batch_size;
    params.hw_kernel_config.enable_cuda_graph_debug_mode = true;
    params.hidden_size                                   = hidden_size;
    params.max_seq_len                                   = max_seq_len;
    params.tokens_per_block                              = tokens_per_block;
    params.hw_kernel_config.prefill_capture_seq_lens     = prefill_capture_seq_lens;
    auto               runner_ptr            = device->getDeviceGraphRunner(params, std::move(py_instance), 0, true);
    CudaGraphRunnerPtr cuda_graph_runner_ptr = dynamic_cast<CudaGraphRunner*>(runner_ptr);
    cuda_graph_runner_ptr->setModelDataType(torch::scalarTypeToTypeMeta(torch::kBFloat16));
    return cuda_graph_runner_ptr;
}

PYBIND11_MODULE(libtest_cuda_graph_prefill_ops, m) {
    py::class_<cuda_graph::CudaGraphPrefillOp>(m, "CudaGraphPrefillOp")
        .def(py::init<>())
        .def("init",
             &CudaGraphPrefillOp::init,
             py::arg("py_instance"),
             py::arg("max_context_batch_size"),
             py::arg("hidden_size"),
             py::arg("max_seq_len"),
             py::arg("tokens_per_block"),
             py::arg("max_prefill_cuda_graph_len"),
             py::arg("prefill_capture_seq_lens"))
        .def("forward", &cuda_graph::CudaGraphPrefillOp::forward)
        .def("getCurrentRealGraphSize", &cuda_graph::CudaGraphPrefillOp::getCurrentRealGraphSize);
    // buildInputs is now implemented in Python (CudaGraphPrefill.py)
}

}  // namespace cuda_graph
