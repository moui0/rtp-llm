#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphRunner.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include <cuda_runtime_api.h>

using namespace torch_ext;
namespace rtp_llm {

GraphBase* CudaDevice::getDeviceGraphRunner(const DeviceInitParams& params,
                                            py::object              py_instance,
                                            int                     kv_cache_block_offset,
                                            bool                    is_prefill_cuda_graph_mode) {
    if (!graph_runner_) {
        graph_runner_ = new CudaGraphRunner(
            params, std::move(py_instance), kv_cache_block_offset, this, is_prefill_cuda_graph_mode);
    }
    return graph_runner_;
}

CudaGraphRunner::CudaGraphRunner(const DeviceInitParams& params,
                                 py::object              py_instance,
                                 int                     kv_cache_block_offset,
                                 DeviceBase*             device,
                                 bool                    is_prefill_cuda_graph_mode):
    GraphBase(params, std::move(py_instance), kv_cache_block_offset, device, is_prefill_cuda_graph_mode),
    capture_stream_(CudaGraphUtils::getStreamFromPool()) {}

CudaGraphRunner::~CudaGraphRunner() {
    RTP_LLM_LOG_INFO("Release CudaGraphRunner .....");
    py::gil_scoped_acquire gil;
    py_instance_.release();
    RTP_LLM_LOG_INFO("Release CudaGraphRunner Successfully");
}

void CudaGraphRunner::deviceSpecificSync() {
    cudaDeviceSynchronize();
}

std::unique_ptr<void, std::function<void(void*)>> CudaGraphRunner::createStreamLife(void* capture_stream) {
    auto* stream_life = new CudaGraphStreamLife(capture_stream_, device_);
    return std::unique_ptr<void, std::function<void(void*)>>(
        stream_life, [](void* ptr) { delete static_cast<CudaGraphStreamLife*>(ptr); });
}

void* CudaGraphRunner::getDeviceStream() {
    return &capture_stream_;
}

void CudaGraphRunner::setCaptureFlag(bool flag) {
    CaptureCheck::in_cuda_graph_capture = flag;
}

}  // namespace rtp_llm
