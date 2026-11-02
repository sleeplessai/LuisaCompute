#pragma once
// embedding_mm engine -- Gemma4VisionModel + embed_vision connector (Tile backend).

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

struct VisionLayerWeightsTile {
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
};

class VisionTowerTile {
public:
    explicit VisionTowerTile(compute::Device &device) noexcept : _device(device) {}

    void load_weights(const Package &package,
                      compute::CommandList &upload_cmds,
                      luisa::vector<luisa::vector<float>> &fp32_storage,
                      luisa::vector<luisa::vector<luisa::bfloat16>> &bf16_storage);

    void compile_shaders() noexcept {}

    void record_encode(
        compute::CommandList &cmd_list,
        compute::BufferView<float> pixel_values,
        compute::BufferView<int32_t> position_ids,
        compute::BufferView<float> out_soft_tokens,
        uint32_t B = 1u,
        uint32_t num_patches_w = 48u,
        uint32_t num_patches_h = 48u,
        uint32_t max_patches = 2520u,
        uint32_t num_soft_tokens = 256u,
        Executor::VisionDebugActivations *debug_acts = nullptr);

private:
    void ensure_workspace(uint32_t max_patches, uint32_t num_soft_tokens);

    compute::Device &_device;
    ops_tile::TileShaderCache _cache;

    // Weights
    compute::Buffer<luisa::bfloat16> _patch_proj;
    compute::Buffer<float> _pos_table;
    compute::Buffer<float> _cos_table;
    compute::Buffer<float> _sin_table;
    luisa::vector<VisionLayerWeightsTile> _layers;
    compute::Buffer<luisa::bfloat16> _connector_proj;

    // Activations
    compute::Buffer<float> _scaled_pixels;
    compute::Buffer<float> _h0;
    compute::Buffer<float> _mask;
    compute::Buffer<float> _h_norm;
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
    compute::Buffer<float> _pooled_tokens;
    compute::Buffer<float> _pooled_norm;

    size_t _max_patches{0};
    size_t _max_soft_tokens{0};
};

} // namespace luisa::embed_mm
