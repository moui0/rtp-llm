#pragma once
#include "rtp_llm/cpp/devices/GraphBase.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphUtils.h"
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>

namespace rtp_llm {

class CudaGraphRunner: public GraphBase {
public:
    CudaGraphRunner(const DeviceInitParams& params,
                    py::object              py_instance,
                    int                     kv_cache_block_offset,
                    DeviceBase*             device,
                    bool                    is_prefill_cuda_graph_mode = false);

    ~CudaGraphRunner() override;

    // 实现设备特定的虚函数
    void                                              deviceSpecificSync() override;
    std::unique_ptr<void, std::function<void(void*)>> createStreamLife(void* capture_stream) override;
    void*                                             getDeviceStream() override;
    void                                              setCaptureFlag(bool flag) override;

    // CUDA 特定的成员变量
    at::cuda::CUDAStream capture_stream_;
};
}  // namespace rtp_llm