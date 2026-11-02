#pragma once
// embedding_mm engine -- Tile Attention operators (Fused FlashAttention / Online-Softmax GQA & Audio Chunked Attention).

#include <cstdint>
#include <luisa/core/basic_types.h>
#include <luisa/tile.h>
#include "embed/ops_tile/ops_tile_common.h"

namespace luisa::embed_mm::ops_tile {

// Fused Online-Softmax GQA / MHA with BF16 Tensor Core MMA:
// Q: [B, S, num_q_heads, head_dim]
// K: [B, S, num_kv_heads, head_dim]
// V: [B, S, num_kv_heads, head_dim]
// mask: [B, S]
// O: [B, S, num_q_heads, head_dim]
[[nodiscard]] compute::tile::Kernel make_tile_gqa_attention_kernel(
    uint32_t B,
    uint32_t S,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    float scale,
    uint32_t is_sliding,
    uint32_t sliding_window,
    float neg_large = -1e30f) noexcept;

// Audio Q/K scaling with softplus(per_dim_scale)
[[nodiscard]] compute::tile::Kernel make_tile_audio_scale_qk_kernel(
    uint32_t T,
    uint32_t num_heads,
    uint32_t head_dim,
    float q_scale,
    float k_scale) noexcept;

// Audio Conformer Chunked Local Self-Attention with relative position bias and tanh softcap
[[nodiscard]] compute::tile::Kernel make_tile_audio_chunked_attention_kernel(
    uint32_t T,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t chunk_size,
    float softcap) noexcept;

// Cached dispatch helpers
[[nodiscard]] TileCmd dispatch_tile_gqa_attention(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> q,
    compute::BufferView<float> k,
    compute::BufferView<float> v,
    compute::BufferView<float> mask,
    compute::BufferView<float> out,
    uint32_t B,
    uint32_t S,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    float scale,
    uint32_t is_sliding,
    uint32_t sliding_window,
    float neg_large = -1e30f);

[[nodiscard]] TileCmd dispatch_tile_audio_scale_qk(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> q,
    compute::BufferView<float> k,
    compute::BufferView<float> per_dim_scale,
    uint32_t T,
    uint32_t num_heads,
    uint32_t head_dim,
    float q_scale,
    float k_scale);

[[nodiscard]] TileCmd dispatch_tile_audio_chunked_attention(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> q,
    compute::BufferView<float> k,
    compute::BufferView<float> v,
    compute::BufferView<float> rel_k,
    compute::BufferView<float> out_attn,
    uint32_t T,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t chunk_size,
    float softcap);

}// namespace luisa::embed_mm::ops_tile
