#pragma once
// embedding_mm engine -- Tile MLP activations, gating, clamping, and residual operators.

#include <cstdint>
#include <luisa/core/basic_types.h>
#include <luisa/tile.h>
#include "embed/ops_tile/ops_tile_common.h"

namespace luisa::embed_mm::ops_tile {

[[nodiscard]] compute::tile::Kernel make_tile_geglu_kernel(uint32_t N) noexcept;

[[nodiscard]] compute::tile::Kernel make_tile_residual_add_kernel(uint32_t N) noexcept;
[[nodiscard]] compute::tile::Kernel make_tile_residual_add_inplace_kernel(uint32_t N) noexcept;

[[nodiscard]] compute::tile::Kernel make_tile_ple_gate_kernel(
    uint32_t layer_idx, uint32_t total_tokens, uint32_t num_layers, uint32_t H) noexcept;

[[nodiscard]] compute::tile::Kernel make_tile_scale_add_kernel(uint32_t N, float scale) noexcept;
[[nodiscard]] compute::tile::Kernel make_tile_scale_add_inplace_kernel(uint32_t N, float scale) noexcept;

[[nodiscard]] compute::tile::Kernel make_tile_audio_clamp_kernel(
    uint32_t N, float min_v, float max_v) noexcept;
[[nodiscard]] compute::tile::Kernel make_tile_audio_clamp_inplace_kernel(
    uint32_t N, float min_v, float max_v) noexcept;

[[nodiscard]] compute::tile::Kernel make_tile_audio_silu_kernel(uint32_t N) noexcept;
[[nodiscard]] compute::tile::Kernel make_tile_audio_silu_inplace_kernel(uint32_t N) noexcept;

[[nodiscard]] compute::tile::Kernel make_tile_audio_glu_kernel(uint32_t T, uint32_t H) noexcept;

[[nodiscard]] compute::tile::Kernel make_tile_audio_add_scale_kernel(uint32_t N, float scale) noexcept;
[[nodiscard]] compute::tile::Kernel make_tile_audio_add_scale_inplace_kernel(uint32_t N, float scale) noexcept;

// Cached dispatch helpers
[[nodiscard]] TileCmd dispatch_tile_geglu(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> gate,
    compute::BufferView<float> up,
    compute::BufferView<float> out,
    uint32_t N);

[[nodiscard]] TileCmd dispatch_tile_residual_add(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> x,
    compute::BufferView<float> residual,
    compute::BufferView<float> out,
    uint32_t N);

[[nodiscard]] TileCmd dispatch_tile_ple_gate(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> gate,
    compute::BufferView<float> ple_all,
    compute::BufferView<float> out,
    uint32_t layer_idx,
    uint32_t total_tokens,
    uint32_t num_layers,
    uint32_t H);

[[nodiscard]] TileCmd dispatch_tile_scale_add(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> x,
    compute::BufferView<float> residual,
    compute::BufferView<float> out,
    float scale,
    uint32_t N);

[[nodiscard]] TileCmd dispatch_tile_audio_clamp(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> x,
    compute::BufferView<float> y,
    float min_v,
    float max_v,
    uint32_t N);

[[nodiscard]] TileCmd dispatch_tile_audio_silu(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> x,
    compute::BufferView<float> y,
    uint32_t N);

[[nodiscard]] TileCmd dispatch_tile_audio_glu(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> in_x,
    compute::BufferView<float> out_y,
    uint32_t T,
    uint32_t H);

[[nodiscard]] TileCmd dispatch_tile_audio_add_scale(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> x,
    compute::BufferView<float> residual,
    compute::BufferView<float> y,
    float scale,
    uint32_t N);

}// namespace luisa::embed_mm::ops_tile
