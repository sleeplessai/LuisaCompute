#pragma once
// embedding_mm engine -- Tile Audio 2D Subsampling & Causal Depthwise 1D Convolution operators.

#include <cstdint>
#include <luisa/core/basic_types.h>
#include <luisa/tile.h>
#include "embed/ops_tile/ops_tile_common.h"

namespace luisa::embed_mm::ops_tile {

// Audio Subsample Conv0: 3x3 stride-2 padding-1 (1 -> 128 channels) + LayerNorm(128) + ReLU
// in_features: [H_in, W_in]
// weight: [128, 3, 3]
// norm_weight: [128]
// out_h0: [128, H_out, W_out]
[[nodiscard]] compute::tile::Kernel make_tile_audio_subsample_conv0_kernel(
    uint32_t H_in, uint32_t W_in, uint32_t H_out, uint32_t W_out, float eps) noexcept;

// Audio Subsample Conv1: 3x3 stride-2 padding-1 (128 -> 32 channels) + LayerNorm(32) + ReLU
// in_h0: [128, H_in, W_in]
// weight: [32, 128, 3, 3]
// norm_weight: [32]
// out_flat: [H_out, W_out, 32] (flattened as [H_out, W_out * 32])
[[nodiscard]] compute::tile::Kernel make_tile_audio_subsample_conv1_kernel(
    uint32_t H_in, uint32_t W_in, uint32_t H_out, uint32_t W_out, float eps) noexcept;

// Causal 1D Depthwise Convolution (kernel_size = 5, left pad = 4):
// in_x: [T, channels], weight: [channels, 5] -> out_y: [T, channels]
[[nodiscard]] compute::tile::Kernel make_tile_audio_depthwise_conv1d_kernel(
    uint32_t T, uint32_t channels) noexcept;

// Cached dispatch helpers
[[nodiscard]] TileCmd dispatch_tile_audio_subsample_conv0(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> in_features,
    compute::BufferView<float> weight,
    compute::BufferView<float> norm_weight,
    compute::BufferView<float> out_h0,
    uint32_t H_in, uint32_t W_in, uint32_t H_out, uint32_t W_out, float eps);

[[nodiscard]] TileCmd dispatch_tile_audio_subsample_conv1(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> in_h0,
    compute::BufferView<float> weight,
    compute::BufferView<float> norm_weight,
    compute::BufferView<float> out_flat,
    uint32_t H_in, uint32_t W_in, uint32_t H_out, uint32_t W_out, float eps);

[[nodiscard]] TileCmd dispatch_tile_audio_depthwise_conv1d(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> in_x,
    compute::BufferView<float> weight,
    compute::BufferView<float> out_y,
    uint32_t T, uint32_t channels);

}// namespace luisa::embed_mm::ops_tile
