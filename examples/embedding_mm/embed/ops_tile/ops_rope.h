#pragma once
// embedding_mm engine -- Tile Positional Encoding operators (1D RoPE, 2D Axial RoPE, Audio Relative Pos).

#include <cstdint>
#include <luisa/core/basic_types.h>
#include <luisa/core/stl/vector.h>
#include <luisa/tile.h>
#include "embed/ops_tile/ops_tile_common.h"

namespace luisa::embed_mm::ops_tile {

// Host table builders for precomputed cos/sin tables (since Tile IR uses table-driven trig for RoPE)
void build_rope_cos_sin_tables(
    uint32_t max_seq_len,
    uint32_t head_dim,
    float rope_theta,
    luisa::vector<float> &out_cos,
    luisa::vector<float> &out_sin);

void build_axial_rope_cos_sin_tables(
    uint32_t pos_table_size,
    uint32_t quarter_dim,
    float rope_theta,
    luisa::vector<float> &out_cos,
    luisa::vector<float> &out_sin);

void build_audio_rel_pos_table(
    uint32_t pos_len,
    uint32_t hidden_size,
    luisa::vector<float> &out_pos_embed);

// 1D RoPE (out-of-place and in-place) using precomputed cos_table [max_S, half_dim] and sin_table [max_S, half_dim]
[[nodiscard]] compute::tile::Kernel make_tile_rope_kernel(
    uint32_t B, uint32_t S, uint32_t num_heads, uint32_t head_dim, uint32_t table_seq_len) noexcept;

[[nodiscard]] compute::tile::Kernel make_tile_rope_inplace_kernel(
    uint32_t B, uint32_t S, uint32_t num_heads, uint32_t head_dim, uint32_t table_seq_len) noexcept;

// 2D Vision Axial RoPE (out-of-place and in-place) using precomputed cos_table [pos_table_size, 16] and sin_table [pos_table_size, 16]
[[nodiscard]] compute::tile::Kernel make_tile_axial_rope_kernel(
    uint32_t num_patches, uint32_t num_heads, uint32_t head_dim, uint32_t pos_table_size = 10240u) noexcept;

[[nodiscard]] compute::tile::Kernel make_tile_axial_rope_inplace_kernel(
    uint32_t num_patches, uint32_t num_heads, uint32_t head_dim, uint32_t pos_table_size = 10240u) noexcept;

// Cached dispatch helpers
[[nodiscard]] TileCmd dispatch_tile_rope(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> qk_in,
    compute::BufferView<float> qk_out,
    compute::BufferView<float> cos_table,
    compute::BufferView<float> sin_table,
    uint32_t B, uint32_t S, uint32_t num_heads, uint32_t head_dim, uint32_t table_seq_len, uint64_t table_tag);

[[nodiscard]] TileCmd dispatch_tile_axial_rope(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> qk_in,
    compute::BufferView<float> qk_out,
    compute::BufferView<uint32_t> pos_ids,
    compute::BufferView<float> cos_table,
    compute::BufferView<float> sin_table,
    uint32_t num_patches, uint32_t num_heads, uint32_t head_dim, uint32_t pos_table_size = 10240u);

}// namespace luisa::embed_mm::ops_tile
