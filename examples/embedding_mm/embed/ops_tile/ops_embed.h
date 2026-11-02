#pragma once
// embedding_mm engine -- Tile Token Embedding, Patch Position Embedding, and Multimodal Scatter operators.

#include <cstdint>
#include <luisa/core/basic_types.h>
#include <luisa/tile.h>
#include "embed/ops_tile/ops_tile_common.h"

namespace luisa::embed_mm::ops_tile {

// Embedding lookup from BF16 table: input_ids [total_tokens] (uint32), table [vocab_size, H] (bf16) -> out [total_tokens, H] (fp32)
[[nodiscard]] compute::tile::Kernel make_tile_embed_lookup_bf16_kernel(
    uint32_t total_tokens, uint32_t vocab_size, uint32_t H, float scale) noexcept;

// Embedding lookup from FP32 table: input_ids [total_tokens] (uint32), table [vocab_size, H] (fp32) -> out [total_tokens, H] (fp32)
[[nodiscard]] compute::tile::Kernel make_tile_embed_lookup_fp32_kernel(
    uint32_t total_tokens, uint32_t vocab_size, uint32_t H, float scale) noexcept;

// Pixel rescaling: out[i] = 2.0f * (in[i] - 0.5f)
[[nodiscard]] compute::tile::Kernel make_tile_scale_pixels_kernel(uint32_t total) noexcept;

// Add 2D patch position embeddings in-place on hidden [num_patches, H]
[[nodiscard]] compute::tile::Kernel make_tile_add_patch_pos_embed_kernel(
    uint32_t num_patches, uint32_t H, uint32_t pos_table_size) noexcept;

// Generate patch validity mask [num_patches] (1.0f if pos_x < pos_table_size else 0.0f)
[[nodiscard]] compute::tile::Kernel make_tile_mask_from_pos_kernel(
    uint32_t num_patches, uint32_t pos_table_size = 10240u) noexcept;

// Contiguous soft-token scatter into h0 [total_tokens, H] at start_token_idx
[[nodiscard]] compute::tile::Kernel make_tile_scatter_soft_tokens_kernel(
    uint32_t total_tokens, uint32_t start_token_idx, uint32_t num_soft_tokens, uint32_t H) noexcept;

// Indexed soft-token scatter into h0 [total_tokens, H] at token_indices[i]
[[nodiscard]] compute::tile::Kernel make_tile_scatter_indexed_tokens_kernel(
    uint32_t total_tokens, uint32_t num_soft_tokens, uint32_t H) noexcept;

// Cached dispatch helpers
[[nodiscard]] TileCmd dispatch_tile_embed_lookup(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<uint32_t> input_ids,
    compute::BufferView<luisa::bfloat16> table,
    compute::BufferView<float> out,
    uint32_t total_tokens,
    uint32_t vocab_size,
    uint32_t H,
    float scale);

[[nodiscard]] TileCmd dispatch_tile_scale_pixels(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> in_pixels,
    compute::BufferView<float> out_pixels,
    uint32_t total);

[[nodiscard]] TileCmd dispatch_tile_add_patch_pos_embed(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> hidden,
    compute::BufferView<float> table,
    compute::BufferView<uint32_t> pos_ids,
    uint32_t num_patches,
    uint32_t H,
    uint32_t pos_table_size);

[[nodiscard]] TileCmd dispatch_tile_mask_from_pos(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<uint32_t> pos_ids,
    compute::BufferView<float> out_mask,
    uint32_t num_patches,
    uint32_t pos_table_size = 10240u);

[[nodiscard]] TileCmd dispatch_tile_scatter_soft_tokens(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> h0,
    compute::BufferView<float> soft_tokens,
    uint32_t total_tokens,
    uint32_t start_token_idx,
    uint32_t num_soft_tokens,
    uint32_t H);

[[nodiscard]] TileCmd dispatch_tile_scatter_indexed_tokens(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> h0,
    compute::BufferView<float> soft_tokens,
    compute::BufferView<uint32_t> token_indices,
    uint32_t total_tokens,
    uint32_t num_soft_tokens,
    uint32_t H);

}// namespace luisa::embed_mm::ops_tile
