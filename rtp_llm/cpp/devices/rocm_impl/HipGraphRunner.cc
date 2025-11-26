#include "rtp_llm/cpp/devices/rocm_impl/HipGraphRunner.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#include <hip/hip_runtime_api.h>

using namespace torch_ext;
namespace rtp_llm {

GraphBase* ROCmDevice::getDeviceGraphRunner(const DeviceInitParams& params,
                                            py::object              py_instance,
                                            int                     kv_cache_block_offset,
                                            bool                    is_prefill_hip_graph_mode) {
    if (!graph_runner_) {
        graph_runner_ =
            new HipGraphRunner(params, std::move(py_instance), kv_cache_block_offset, this, is_prefill_hip_graph_mode);
    }
    return graph_runner_;
}

HipGraphRunner::HipGraphRunner(const DeviceInitParams& params,
                               py::object              py_instance,
                               int                     kv_cache_block_offset,
                               DeviceBase*             device,
                               bool                    is_prefill_hip_graph_mode):
    GraphBase(params, std::move(py_instance), kv_cache_block_offset, device, is_prefill_hip_graph_mode),
    capture_stream_(HipGraphUtils::getStreamFromPool()) {}

HipGraphRunner::~HipGraphRunner() {
    RTP_LLM_LOG_INFO("Release HipGraphRunner .....");
    py::gil_scoped_acquire gil;
    py_instance_.release();
    RTP_LLM_LOG_INFO("Release HipGraphRunner Successfully");
}

void HipGraphRunner::deviceSpecificSync() {
    hipDeviceSynchronize();
}

std::unique_ptr<void, std::function<void(void*)>> HipGraphRunner::createStreamLife(void* capture_stream) {
    auto* stream_life = new HipGraphStreamLife(capture_stream_, device_);
    return std::unique_ptr<void, std::function<void(void*)>>(
        stream_life, [](void* ptr) { delete static_cast<HipGraphStreamLife*>(ptr); });
}

void* HipGraphRunner::getDeviceStream() {
    return &capture_stream_;
}

void HipGraphRunner::setCaptureFlag(bool flag) {
    CaptureCheck::in_hip_graph_capture = flag;
}

bool HipGraphRunner::shouldUsePinnedCPUMemory() {
    // ROCm 不使用 pinned CPU memory
    return false;
}

bool HipGraphRunner::isCudaGraph() {
    // HipGraphRunner 是 HIP graph，不是 CUDA graph
    return false;
}

}  // namespace rtp_llm