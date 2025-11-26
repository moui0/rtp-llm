#include "rtp_llm/cpp/devices/GraphBase.h"

namespace rtp_llm {

// Constructor implementations
GraphBase::GraphBase(py::object py_instance): py_instance_(std::move(py_instance)) {}

GraphBase::GraphBase(const DeviceInitParams& params,
                     py::object              py_instance,
                     int                     kv_cache_block_offset,
                     DeviceBase*             device,
                     bool                    is_prefill_mode):
    enable_graph_(params.hw_kernel_config.enable_cuda_graph),
    is_prefill_graph_mode_(is_prefill_mode),
    enable_graph_debug_mode_(params.hw_kernel_config.enable_cuda_graph_debug_mode),
    concurrency_limit_(params.concurrency_config.concurrency_limit),
    hidden_size_(params.hidden_size),
    max_seq_len_(params.max_seq_len),
    seq_size_per_block_(params.tokens_per_block),
    kv_cache_block_offset_(kv_cache_block_offset),
    device_(device),
    py_instance_(std::move(py_instance)) {

    py::gil_scoped_acquire gil;
    if (!py_instance_ || py_instance_.is_none()) {
        throw std::runtime_error("GraphRunner constructor: Python instance is null or none.");
    }
    py_forward_method_     = py_instance_.attr("forward");
    py_fill_params_method_ = py_instance_.attr("fill_params");

    RTP_LLM_LOG_INFO(
        "Initialize GraphRunner with parameters below: enable_graph=%d, concurrency_limit=%d, debug_mode=%d, "
        "hidden_size=%d, max_seq_len=%d, seq_size_per_block=%d, kv_cache_offset=%d, prefill_mode=%d",
        enable_graph_,
        concurrency_limit_,
        enable_graph_debug_mode_,
        hidden_size_,
        max_seq_len_,
        seq_size_per_block_,
        kv_cache_block_offset_,
        is_prefill_graph_mode_);
}

// Device-specific virtual functions
void GraphBase::initCapture() {
    if (enable_graph_) {
        RTP_LLM_LOG_INFO("Graph capture is enabled");
        if (is_prefill_graph_mode_) {
            RTP_LLM_LOG_INFO("Graph capture for embedding");
            // for embedding model which is prefill-only, the `input_ids` shape should be: [bs, max_seq_len_].
            // we will do mask for extra tokens in attention mechanism.
            num_tokens_per_bs_ = max_seq_len_;
        }

        // Capture
        capture_range_ = GraphUtils::getBatchSizesToCapture(concurrency_limit_);
        max_bs_        = *(std::max_element(capture_range_.begin(), capture_range_.end()));
        max_num_token_ = max_bs_ * num_tokens_per_bs_;

        PyModelInputs inputs;

        // Setup attention inputs using the extracted function
        initCaptureAttentionInputs(inputs, max_bs_, num_tokens_per_bs_);

        // Setup BertEmbedding inputs using the extracted function
        initCaptureBertEmbeddingInputs(inputs, max_bs_, max_num_token_);

        torch::Tensor output;
        capture_mem_hold_ = CaptureMemoryHold(output, inputs, kv_cache_block_offset_, is_prefill_graph_mode_);
        initKernelInternalMemory();

        // get real output data type
        auto py_outputs_obj     = py_forward_method_(capture_mem_hold_.py_model_inputs_);
        auto outputs            = py_outputs_obj.cast<PyModelOutputs>();
        auto options_cuda_float = torch::TensorOptions()
                                      .dtype(outputs.hidden_states.dtype().toScalarType())
                                      .device(torch::kCUDA)
                                      .requires_grad(false);
        output = torch::zeros({max_num_token_, hidden_size_}, options_cuda_float);
        capture_mem_hold_.setHiddenStates(output);
        capture();
    } else {
        initKernelInternalMemory();
        RTP_LLM_LOG_INFO("Graph capture is not enabled, skipping initialization");
    }
}

void GraphBase::setPositionEncoding(torch::Tensor position_encoding) {
    position_encoding_ = position_encoding;
}

void GraphBase::setTokenTypeEmbedding(torch::Tensor token_type_embedding) {
    token_type_embedding_ = token_type_embedding;
}

void GraphBase::setInputEmbeddingScalar(float input_embedding_scalar) {
    input_embedding_scalar_ = input_embedding_scalar;
}

void GraphBase::setModelDataType(caffe2::TypeMeta data_type) {
    model_data_type_ = data_type;
}

void GraphBase::setParamsPtr(int bs, const PyModelOutputs& outputs) {
    // Default implementation with recycle checking
    if (outputs.params_ptr->check_recycle()) {
        graph_instances_[bs].mem_hold_.params_ptr = ParamsBasePtr(outputs.params_ptr.get(), [&](ParamsBase* ptr) {});
    } else {
        graph_instances_[bs].mem_hold_.params_ptr = outputs.params_ptr;
    }
}

bool GraphBase::shouldUsePinnedCPUMemory() {
    // 默认实现 使用pinned CPU内存
    return true;
}

bool GraphBase::isCudaGraph() {
    // 默认实现 假设是 CUDA graph
    return true;
}

void GraphBase::initCaptureAttentionInputs(PyModelInputs& inputs, int max_bs, int num_tokens_per_bs) {
    auto options_cpu_int32  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).requires_grad(false);
    auto options_cuda_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);

    inputs.attention_inputs.is_prefill = is_prefill_graph_mode_;
    // input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
    inputs.input_ids = torch::zeros({max_num_token_}, options_cuda_int32);

    // Device-aware tensor allocation
    bool use_pinned_cpu                 = shouldUsePinnedCPUMemory();
    auto input_lengths_options          = use_pinned_cpu ? options_cpu_int32 : options_cuda_int32;
    auto sequence_lengths_options       = use_pinned_cpu ? options_cpu_int32 : options_cuda_int32;
    auto kv_cache_block_id_host_options = options_cpu_int32;
    auto padding_offset_options         = use_pinned_cpu ? options_cpu_int32 : options_cuda_int32;

    // input_lengths [batch_size, int32]
    inputs.attention_inputs.input_lengths = torch::full({int(max_bs_)}, num_tokens_per_bs_, input_lengths_options);

    // sequence_lengths [batch_size, int32]
    inputs.attention_inputs.sequence_lengths = torch::ones({int(max_bs_)}, sequence_lengths_options);
    if (use_pinned_cpu) {
        inputs.attention_inputs.sequence_lengths = inputs.attention_inputs.sequence_lengths.pin_memory();
    }

    // kv_cache_block_id_device [batch_size, block_num]
    inputs.attention_inputs.kv_cache_block_id_device = torch::zeros(
        {int(max_bs_), ((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_)}, options_cuda_int32);

    // prefix_lengths [batch_size, int32] (for attention `prepare`)
    if (isCudaGraph()) {
        inputs.attention_inputs.prefix_lengths = torch::full({int(max_bs_)}, num_tokens_per_bs_, input_lengths_options);
    } else {
        inputs.attention_inputs.prefix_lengths = torch::full({int(max_bs_)}, 0, input_lengths_options);
    }

    // kv_cache_block_id_host [batch_size, block_num]
    inputs.attention_inputs.kv_cache_block_id_host =
        torch::zeros({int(max_bs_), ((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_)},
                     kv_cache_block_id_host_options);

    // padding_offset [max_num_token_, int32] (for attention padding)
    inputs.attention_inputs.padding_offset = torch::zeros({int(max_seq_len_ * max_bs_)}, padding_offset_options);
    if (use_pinned_cpu) {
        inputs.attention_inputs.padding_offset = inputs.attention_inputs.padding_offset.pin_memory();
    }

    inputs.attention_inputs.dtype = model_data_type_;
}

void GraphBase::initCaptureBertEmbeddingInputs(PyModelInputs& inputs, int max_bs, int max_num_token) {
    auto options_cuda_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);

    // Initialize BertEmbeddingInputs for capture
    // combo_position_ids: empty tensor for capture (will be filled during actual forward)
    inputs.bert_embedding_inputs.combo_position_ids = torch::zeros({max_seq_len_ * max_bs}, options_cuda_int32);

    // position_encoding: from weights
    inputs.bert_embedding_inputs.position_encoding = position_encoding_;

    // combo_tokens_type_ids: empty tensor for capture (will be filled during actual forward)
    inputs.bert_embedding_inputs.combo_tokens_type_ids = torch::zeros({max_seq_len_ * max_bs}, options_cuda_int32);

    // token_type_embedding: from weights
    inputs.bert_embedding_inputs.token_type_embedding = token_type_embedding_;

    // input_embedding_scalar: fixed value
    inputs.bert_embedding_inputs.input_embedding_scalar = input_embedding_scalar_;
}

// Main interface methods
PyModelOutputs GraphBase::forward(PyModelInputs& inputs) {
    PyModelOutputs outputs;
    // decode or embedding model only
    if (canRun(inputs)) {
        RTP_LLM_LOG_INFO("Replay Start");
        prepareInputs(inputs);
        replay(current_real_graph_bs_);

        if (is_prefill_graph_mode_) {
            // In embedding mode, extract valid parts from padded decoder_layer_hidden_states_
            auto& hidden_states = graph_instances_[current_real_graph_bs_].mem_hold_.decoder_layer_hidden_states_;
            // create output tensor
            outputs.hidden_states = hidden_states;
            auto input_lengths    = inputs.attention_inputs.input_lengths.data_ptr<int32_t>();

            // calculate valid tokens num
            int32_t total_valid_tokens = 0;
            for (int i = 0; i < current_batch_size_; i++) {
                total_valid_tokens += input_lengths[i];
            }

            // Extract valid hidden states using the extracted function
            extractValidHiddenStates(outputs, inputs, total_valid_tokens);
        } else {
            outputs.hidden_states =
                graph_instances_[current_real_graph_bs_].mem_hold_.decoder_layer_hidden_states_.slice(
                    0, 0, seq_len_sum_);
        }
        RTP_LLM_LOG_INFO("Replay End");
    } else {
        auto py_outputs_obj = normalForward(inputs);
        // Cast the Python object to PyModelOutputs and extract hidden states
        outputs = py_outputs_obj.cast<PyModelOutputs>();
    }

    return outputs;
}

bool GraphBase::canRun(PyModelInputs& inputs) {
    if (!enable_graph_ || (inputs.attention_inputs.is_prefill && !is_prefill_graph_mode_)) {
        return false;
    }
    return tryGetRealGraphBatchSize(inputs);
}

int GraphBase::getCurrentRealGraphBs() {
    return current_real_graph_bs_;
}

bool GraphBase::tryGetRealGraphBatchSize(PyModelInputs& inputs) {
    int cuda_graph_bs   = inputs.attention_inputs.input_lengths.size(0);
    current_batch_size_ = cuda_graph_bs;
    RTP_LLM_LOG_INFO("canRun judge for batch size: %d", cuda_graph_bs);
    bool is_bs_supported   = (cuda_graph_bs <= max_bs_);
    auto it                = std::lower_bound(capture_range_.begin(), capture_range_.end(), current_batch_size_);
    current_real_graph_bs_ = *it;
    RTP_LLM_CHECK_WITH_INFO(it != capture_range_.end(), "batch size used in replay: %d", current_real_graph_bs_);
    seq_len_sum_ = inputs.attention_inputs.input_lengths.sum(0).item<int>();
    RTP_LLM_LOG_INFO("can run graph: %d", is_bs_supported);
    return is_bs_supported;
}

void GraphBase::extractValidHiddenStates(PyModelOutputs&      outputs,
                                         const PyModelInputs& inputs,
                                         int32_t              total_valid_tokens) {
    auto& hidden_states = graph_instances_[current_real_graph_bs_].mem_hold_.decoder_layer_hidden_states_;
    GraphUtils::extractValidHiddenStates(
        outputs, inputs, total_valid_tokens, hidden_states, current_batch_size_, num_tokens_per_bs_);
}

py::object GraphBase::normalForward(PyModelInputs& inputs) {
    return py_forward_method_(inputs);
}

void GraphBase::initKernelInternalMemory() {
    BufferPtr cu_seqlens_buf = device_->allocateBuffer({DataType::TYPE_INT32, {max_bs_ + 1}, AllocationType::HOST});
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens = Buffer2torchTensor(cu_seqlens_buf, false);
    RTP_LLM_CHECK_WITH_INFO(capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.is_pinned(),
                            "capture_mem_hold_ sequence_lengths is not pinned memory");
}

void GraphBase::captureOneBatchSize(int bs) {
    auto inputs = graph_instances_[bs].mem_hold_.py_model_inputs_;

    // WarmUp twice
    RTP_LLM_LOG_INFO("WarmUp for batch size %d start.", bs);
    py_forward_method_(inputs);
    py_forward_method_(inputs);
    RTP_LLM_LOG_INFO("WarmUp for batch size %d successfully.", bs);

    {
        auto  stream_life         = createStreamLife(getDeviceStream());
        auto& graph               = graph_instances_[bs].graph_;
        auto  output_dot_filename = "";

        if (enable_graph_debug_mode_) {
            graph.enable_debug_mode();
            output_dot_filename = "cuda_graph_visualization.dot";
        }

        RTP_LLM_LOG_INFO("Capture for batch size %d begin.", bs);
        graph.capture_begin();

        setCaptureFlag(true);

        auto py_outputs_obj = py_forward_method_(inputs);
        auto outputs        = py_outputs_obj.cast<PyModelOutputs>();
        graph_instances_[bs].mem_hold_.decoder_layer_hidden_states_.copy_(outputs.hidden_states);

        graph.capture_end();
        RTP_LLM_LOG_INFO("Capture for batch size %d end.", bs);

        setCaptureFlag(false);

        setParamsPtr(bs, outputs);

        if (enable_graph_debug_mode_) {
            graph.debug_dump(output_dot_filename);
        }
    }
}

void GraphBase::prepareInputs(PyModelInputs& inputs) {
    auto& py_model_inputs_ = graph_instances_[current_real_graph_bs_].mem_hold_.py_model_inputs_;
    py_model_inputs_.attention_inputs.input_lengths.slice(0, 0, current_batch_size_) =
        inputs.attention_inputs.input_lengths;
    py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, current_batch_size_ + 1) =
        inputs.attention_inputs.cu_seqlens.slice(0, 0, current_batch_size_ + 1);

    if (!is_prefill_graph_mode_) {
        py_model_inputs_.input_ids.fill_(0);
        py_model_inputs_.input_ids.slice(0, 0, inputs.input_ids.size(0)) = inputs.input_ids;
        py_model_inputs_.attention_inputs.sequence_lengths.slice(0, 0, current_batch_size_) =
            inputs.attention_inputs.sequence_lengths;
        GraphUtils::copySmallerIntoLarger(inputs.attention_inputs.kv_cache_block_id_device,
                                          py_model_inputs_.attention_inputs.kv_cache_block_id_device);
        graph_instances_[current_real_graph_bs_].mem_hold_.params_ptr->fillParams(
            inputs.attention_inputs.sequence_lengths,
            inputs.attention_inputs.input_lengths,
            inputs.attention_inputs.kv_cache_block_id_host,
            current_batch_size_,
            seq_size_per_block_);
    } else {
        preparePrefillInputs(inputs, py_model_inputs_);
    }
}

void GraphBase::capture() {
    RTP_LLM_LOG_INFO("Capture Start");
    int  capture_range_size = capture_range_.size();
    bool use_pinned_cpu     = shouldUsePinnedCPUMemory();

    for (int i = 0; i <= capture_range_size - 1; i++) {
        int           bs = capture_range_[i];
        PyModelInputs inputs;
        inputs.input_ids        = capture_mem_hold_.py_model_inputs_.input_ids.slice(0, 0, bs * num_tokens_per_bs_);
        auto options_cpu_int32  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).requires_grad(false);
        auto options_cuda_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);

        // Device-aware tensor options
        auto input_lengths_options    = use_pinned_cpu ? options_cpu_int32 : options_cuda_int32;
        auto sequence_lengths_options = use_pinned_cpu ? options_cpu_int32 : options_cuda_int32;

        // input_lengths [batch_size, int32]
        inputs.attention_inputs.input_lengths = torch::full({int(bs)}, num_tokens_per_bs_, input_lengths_options);

        // sequence_lengths [batch_size, int32] (decode only)
        inputs.attention_inputs.sequence_lengths = torch::ones({int(bs)}, sequence_lengths_options);
        if (use_pinned_cpu) {
            inputs.attention_inputs.sequence_lengths = inputs.attention_inputs.sequence_lengths.pin_memory();
        }

        // kv_cache_block_id_device [batch_size, block_num]
        inputs.attention_inputs.kv_cache_block_id_device = torch::zeros(
            {int(bs), ((max_seq_len_ + seq_size_per_block_ - 1) / seq_size_per_block_)}, options_cuda_int32);
        inputs.attention_inputs.kv_cache_block_id_host =
            capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_host.slice(0, 0, bs);
        // pinned memory
        inputs.attention_inputs.cu_seqlens =
            capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, bs + 1);

        if (isCudaGraph()) {
            inputs.attention_inputs.prefix_lengths = capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths;
        } else {
            inputs.attention_inputs.prefix_lengths =
                capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths.slice(0, 0, bs);
        }
        inputs.attention_inputs.dtype = capture_mem_hold_.py_model_inputs_.attention_inputs.dtype;
        inputs.attention_inputs.padding_offset =
            capture_mem_hold_.py_model_inputs_.attention_inputs.padding_offset.slice(0, 0, bs * num_tokens_per_bs_);
        // Copy BertEmbeddingInputs from capture_mem_hold_
        inputs.bert_embedding_inputs = capture_mem_hold_.py_model_inputs_.bert_embedding_inputs;
        graph_instances_[bs].mem_hold_ =
            CaptureMemoryHold(capture_mem_hold_.decoder_layer_hidden_states_.slice(0, 0, bs * num_tokens_per_bs_),
                              inputs,
                              kv_cache_block_offset_,
                              is_prefill_graph_mode_);
        captureOneBatchSize(bs);
        RTP_LLM_LOG_INFO("replay start check for %d", bs);
        replay(bs);
        deviceSpecificSync();
        RTP_LLM_LOG_INFO("replay end check for %d", bs);
        RTP_LLM_LOG_INFO("capture success for batch size: %d", bs);
    }
    RTP_LLM_LOG_INFO("Capture End");
}

void GraphBase::preparePrefillInputs(PyModelInputs& inputs, PyModelInputs& py_model_inputs_) {
    auto input_lengths_ptr  = inputs.attention_inputs.input_lengths.data_ptr<int32_t>();
    auto padding_offset_ptr = py_model_inputs_.attention_inputs.padding_offset.data_ptr<int32_t>();

    int32_t cum_offset = 0, index = 0;
    for (int32_t i = 0; i < current_batch_size_; i++) {
        index           = i * num_tokens_per_bs_;
        int32_t seq_len = input_lengths_ptr[i];
        for (int32_t j = 0; j < seq_len; j++) {
            padding_offset_ptr[index++] = cum_offset;
        }
        cum_offset += num_tokens_per_bs_ - seq_len;
    }

    py_model_inputs_.input_ids.fill_(0);
    auto lengths   = inputs.attention_inputs.input_lengths.data_ptr<int>();
    int  start_idx = 0;

    for (int i = 0; i < current_batch_size_; i++) {
        int dst_start = i * num_tokens_per_bs_;
        int dst_end   = dst_start + lengths[i];
        int src_start = start_idx;
        int src_end   = src_start + lengths[i];

        py_model_inputs_.input_ids.slice(0, dst_start, dst_end) = inputs.input_ids.slice(0, src_start, src_end);

        if (inputs.bert_embedding_inputs.position_encoding.numel() > 0) {
            py_model_inputs_.bert_embedding_inputs.combo_position_ids.slice(0, dst_start, dst_end) =
                inputs.bert_embedding_inputs.combo_position_ids.slice(0, src_start, src_end);
            py_model_inputs_.bert_embedding_inputs.combo_tokens_type_ids.slice(0, dst_start, dst_end) =
                inputs.bert_embedding_inputs.combo_tokens_type_ids.slice(0, src_start, src_end);
        }
        start_idx += lengths[i];
    }
}

// Device-specific virtual functions with default empty implementations
void GraphBase::replay(int bs) {
    // 子类应该实现此函数来 replay graph
    graph_instances_[bs].graph_.replay();
}

void GraphBase::deviceSpecificSync() {
    // 子类应该实现设备特定的同步操作
    // 例如: cudaDeviceSynchronize() 或 hipDeviceSynchronize()
}

std::unique_ptr<void, std::function<void(void*)>> GraphBase::createStreamLife(void* capture_stream) {
    // 子类应该实现设备特定的 stream life 管理
    // 例如: 返回 CudaGraphStreamLife 或 HipGraphStreamLife
    return std::unique_ptr<void, std::function<void(void*)>>(nullptr, [](void*) {});
}

void* GraphBase::getDeviceStream() {
    // 子类应该返回设备特定的 stream 指针
    return nullptr;
}

void GraphBase::setCaptureFlag(bool flag) {
    // 子类应该实现设备特定的 capture flag 设置
    // 例如: CaptureCheck::in_cuda_graph_capture = flag;
}

}  // namespace rtp_llm
