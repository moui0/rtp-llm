#pragma once
#include "rtp_llm/cpp/devices/GraphBase.h"
#include "rtp_llm/cpp/devices/rocm_impl/HipGraphUtils.h"
#include <ATen/hip/HIPGraph.h>

namespace rtp_llm {

class HipGraphRunner: public GraphBase {
public:
    HipGraphRunner(const DeviceInitParams& params,
                   py::object              py_instance,
                   int                     kv_cache_block_offset,
                   DeviceBase*             device,
                   bool                    is_prefill_hip_graph_mode = false);

    ~HipGraphRunner() override;

    // 实现设备特定的虚函数
    void                                              deviceSpecificSync() override;
    std::unique_ptr<void, std::function<void(void*)>> createStreamLife(void* capture_stream) override;
    void*                                             getDeviceStream() override;
    void                                              setCaptureFlag(bool flag) override;

    // ROCm 特定：不使用 pinned CPU memory
    bool shouldUsePinnedCPUMemory() override;

    // ROCm 特定：这是 HIP graph 而不是 CUDA graph
    bool isCudaGraph() override;

    // ROCm 特定的成员变量
    at::hip::HIPStream capture_stream_;
};

}  // namespace rtp_llm