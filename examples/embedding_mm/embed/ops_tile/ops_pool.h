#pragma once
// embedding_mm engine -- Tile Pooling & MRL Normalization operators.

#include <cstdint>
#include <luisa/core/basic_types.h>
#include <luisa/tile.h>
#include "embed/ops_tile/ops_tile_common.h"

namespace luisa::embed_mm::ops_tile {

// Vision 3x3 spatial average pooling + scale factor
[[nodiscard]] compute::tile::Kernel make_tile_vision_pool_kernel(
    uint32_t num_patches,
    uint32_t grid_w_soft,
    uint32_t grid_h_soft,
    uint32_t patch_stride_w,
    uint32_t H,
    float scale_factor) noexcept;

// Mask-aware sequence mean pooling: hidden [B, S, H], mask [B, S] -> pooled [B, H]
[[nodiscard]] compute::tile::Kernel make_tile_mean_pool_kernel(
    uint32_t B, uint32_t S, uint32_t H) noexcept;

// Matryoshka Representation Learning (MRL) truncation + L2 normalization:
// in_proj [B, D_proj] -> out_embed [B, target_dim]
[[nodiscard]] compute::tile::Kernel make_tile_mrl_normalize_kernel(
    uint32_t B, uint32_t D_proj, uint32_t target_dim) noexcept;

[[nodiscard]] TileCmd dispatch_tile_vision_pool(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> in_hidden,
    compute::BufferView<float> out_pooled,
    uint32_t num_patches,
    uint32_t grid_w_soft,
    uint32_t grid_h_soft,
    uint32_t patch_stride_w,
    uint32_t H,
    float scale_factor);

[[nodiscard]] TileCmd dispatch_tile_mean_pool(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> hidden,
    compute::BufferView<float> mask,
    compute::BufferView<float> pooled,
    uint32_t B, uint32_t S, uint32_t H);

[[nodiscard]] TileCmd dispatch_tile_mrl_normalize(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> in_proj,
    compute::BufferView<float> out_embed,
    uint32_t B, uint32_t D_proj, uint32_t target_dim);

}// namespace luisa::embed_mm::ops_tile
