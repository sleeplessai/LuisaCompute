#pragma once
// embedding_mm engine -- Gemma4AudioModel + embed_audio connector (Tile backend).

#include <cstdint>
#include <luisa/core/basic_types.h>
#include <luisa/core/bfloat16.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>

#include "embedding_mm/embed/executor.h"
#include "embedding_mm/embed/ops_tile/ops_attention.h"
#include "embedding_mm/embed/ops_tile/ops_conv.h"
#include "embedding_mm/embed/ops_tile/ops_gemm.h"
#include "embedding_mm/embed/ops_tile/ops_mlp.h"
#include "embedding_mm/embed/ops_tile/ops_norm.h"
#include "embedding_mm/embed/ops_tile/ops_rope.h"
#include "embedding_mm/embed/ops_tile/ops_tile_common.h"
#include "embedding_mm/embed/models_tile/model_tile_common.h"
#include "embedding_mm/embed/package.h"

namespace luisa::embed_mm {

struct AudioLayerWeightsTile {
    // FeedForward 1
    compute::Buffer<float> ffn1_pre_norm;
    compute::Buffer<luisa::bfloat16> ffn1_linear1;
    compute::Buffer<luisa::bfloat16> ffn1_linear2;
    compute::Buffer<float> ffn1_post_norm;

    // Self Attention
    compute::Buffer<float> norm_pre_attn;
    compute::Buffer<luisa::bfloat16> q_proj;
    compute::Buffer<luisa::bfloat16> k_proj;
    compute::Buffer<luisa::bfloat16> v_proj;
    compute::Buffer<float> per_dim_scale;
    compute::Buffer<luisa::bfloat16> relative_k_proj;
    compute::Buffer<luisa::bfloat16> out_proj;
    compute::Buffer<float> norm_post_attn;

    // LightConv1D
    compute::Buffer<float> lconv_pre_norm;
    compute::Buffer<luisa::bfloat16> lconv_linear_start;
    compute::Buffer<float> lconv_depthwise; // [1024, 5]
    compute::Buffer<float> lconv_norm;
    compute::Buffer<luisa::bfloat16> lconv_linear_end;

    // FeedForward 2
    compute::Buffer<float> ffn2_pre_norm;
    compute::Buffer<luisa::bfloat16> ffn2_linear1;
    compute::Buffer<luisa::bfloat16> ffn2_linear2;
    compute::Buffer<float> ffn2_post_norm;

    compute::Buffer<float> norm_out;

    // Static clip bounds: 10 operations x [min, max]
    float clip_bounds[10][4]{};
};

class AudioTowerTile {
public:
    explicit AudioTowerTile(compute::Device &device) noexcept : _device(device) {}

    void load_weights(const Package &package,
                      compute::CommandList &upload_cmds,
                      luisa::vector<luisa::vector<float>> &fp32_storage,
                      luisa::vector<luisa::vector<luisa::bfloat16>> &bf16_storage);

    void compile_shaders() noexcept {}

    void record_encode(
        compute::CommandList &cmd_list,
        compute::BufferView<float> input_features,
        compute::BufferView<float> out_soft_tokens,
        uint32_t num_frames = 307u,
        uint32_t num_mel_bins = 128u,
        uint32_t num_soft_tokens = 77u,
        Executor::AudioDebugActivations *debug_acts = nullptr);

private:
    void ensure_workspace(uint32_t num_frames, uint32_t num_soft_tokens);

    compute::Device &_device;
    ops_tile::TileShaderCache _cache;

    // Subsample weights
    compute::Buffer<float> _conv0_weight;
    compute::Buffer<float> _conv0_norm;
    compute::Buffer<float> _conv1_weight;
    compute::Buffer<float> _conv1_norm;
    compute::Buffer<luisa::bfloat16> _subsample_proj;
    compute::Buffer<float> _subsample_bias;

    // Sinusoidal pos table
    compute::Buffer<float> _pos_embed;

    // 12 Conformer layers
    luisa::vector<AudioLayerWeightsTile> _layers;

    // Output & connector
    compute::Buffer<luisa::bfloat16> _output_proj_weight;
    compute::Buffer<float> _output_proj_bias;
    compute::Buffer<luisa::bfloat16> _connector_proj;

    // Activations
    compute::Buffer<float> _conv0_out;
    compute::Buffer<float> _conv1_flat;
    compute::Buffer<float> _hidden;
    compute::Buffer<float> _tmp1;
    compute::Buffer<float> _tmp2;
    compute::Buffer<float> _ffn_mid;
    compute::Buffer<float> _q;
    compute::Buffer<float> _k;
    compute::Buffer<float> _v;
    compute::Buffer<float> _rel_k;
    compute::Buffer<float> _attn_out;
    compute::Buffer<float> _glu_in;
    compute::Buffer<float> _glu_out;
    compute::Buffer<float> _tower_out;
    compute::Buffer<float> _conn_in;

    size_t _max_frames{0};
    size_t _max_soft_tokens{0};
};

} // namespace luisa::embed_mm
