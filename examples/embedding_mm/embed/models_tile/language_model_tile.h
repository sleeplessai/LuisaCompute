#pragma once
// embedding_mm engine -- Gemma4TextModel + PLE + Mean Pool + Projection + MRL Head (Tile backend).

#include <cstdint>
#include <luisa/core/basic_types.h>
#include <luisa/core/bfloat16.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>

#include "embedding_mm/embed/executor.h"
#include "embedding_mm/embed/ops_tile/ops_attention.h"
#include "embedding_mm/embed/ops_tile/ops_embed.h"
#include "embedding_mm/embed/ops_tile/ops_gemm.h"
#include "embedding_mm/embed/ops_tile/ops_mlp.h"
#include "embedding_mm/embed/ops_tile/ops_norm.h"
#include "embedding_mm/embed/ops_tile/ops_pool.h"
#include "embedding_mm/embed/ops_tile/ops_rope.h"
#include "embedding_mm/embed/ops_tile/ops_tile_common.h"
#include "embedding_mm/embed/models_tile/model_tile_common.h"
#include "embedding_mm/embed/package.h"

namespace luisa::embed_mm {

struct TextLayerWeightsTile {
    compute::Buffer<float> input_norm;
    compute::Buffer<luisa::bfloat16> q_proj;
    compute::Buffer<luisa::bfloat16> k_proj;
    compute::Buffer<luisa::bfloat16> v_proj;
    compute::Buffer<float> q_norm;
    compute::Buffer<float> k_norm;
    compute::Buffer<luisa::bfloat16> out_proj;
    compute::Buffer<float> post_attn_norm;
    compute::Buffer<float> pre_ffn_norm;
    compute::Buffer<luisa::bfloat16> gate_proj;
    compute::Buffer<luisa::bfloat16> up_proj;
    compute::Buffer<luisa::bfloat16> down_proj;
    compute::Buffer<float> post_ffn_norm;
    compute::Buffer<luisa::bfloat16> ple_gate;
    compute::Buffer<luisa::bfloat16> ple_proj;
    compute::Buffer<float> ple_norm;
    float layer_scalar{1.0f};
};

class LanguageModelTile {
public:
    explicit LanguageModelTile(compute::Device &device) noexcept : _device(device) {}

    void load_weights(const Package &package,
                      compute::CommandList &upload_cmds,
                      luisa::vector<luisa::vector<float>> &fp32_storage,
                      luisa::vector<luisa::vector<luisa::bfloat16>> &bf16_storage);

    void compile_shaders() noexcept {}

    void record_encode(
        compute::CommandList &cmd_list,
        compute::BufferView<uint32_t> input_ids,
        compute::BufferView<float> attention_mask,
        compute::BufferView<float> out_embeddings,
        uint32_t B,
        uint32_t S,
        uint32_t target_dim,
        const MultimodalInputs &mm_inputs);

private:
    void ensure_workspace(uint32_t B, uint32_t S);

    compute::Device &_device;
    ops_tile::TileShaderCache _cache;

    // Weight buffers
    compute::Buffer<luisa::bfloat16> _word_embeddings;
    compute::Buffer<luisa::bfloat16> _ple_proj;
    compute::Buffer<float> _ple_norm;
    compute::Buffer<float> _final_norm;
    compute::Buffer<luisa::bfloat16> _embedding_proj;
    luisa::vector<TextLayerWeightsTile> _layers;

    // Precomputed RoPE tables
    compute::Buffer<float> _cos_sliding;
    compute::Buffer<float> _sin_sliding;
    compute::Buffer<float> _cos_full;
    compute::Buffer<float> _sin_full;

    // Workspace activations
    compute::Buffer<float> _h0;
    compute::Buffer<float> _h_norm;
    compute::Buffer<float> _ple_all;
    compute::Buffer<float> _q_buf;
    compute::Buffer<float> _k_buf;
    compute::Buffer<float> _v_buf;
    compute::Buffer<float> _attn_ctx;
    compute::Buffer<float> _attn_out;
    compute::Buffer<float> _attn_norm;
    compute::Buffer<float> _ffn_norm;
    compute::Buffer<float> _gate_buf;
    compute::Buffer<float> _up_buf;
    compute::Buffer<float> _mlp_act;
    compute::Buffer<float> _mlp_out;
    compute::Buffer<float> _mlp_norm;
    compute::Buffer<float> _ple_gate_buf;
    compute::Buffer<float> _ple_gated;
    compute::Buffer<float> _ple_out;
    compute::Buffer<float> _ple_norm_buf;
    compute::Buffer<float> _pooled;
    compute::Buffer<float> _proj_out;

    size_t _max_tokens{0};
    size_t _max_batch{0};
    size_t _max_s{0};
};

} // namespace luisa::embed_mm
