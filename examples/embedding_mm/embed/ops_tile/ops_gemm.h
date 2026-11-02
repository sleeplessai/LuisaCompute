#pragma once
// embedding_mm engine -- Tile GEMM (BF16 Tensor Core MMA, FP32 accumulation).
//
// A: [M, K] FP32 row-major activations (cast to BF16 per tile in registers).
// B: [K, N] BF16 row-major weights (pre-transposed offline, same layout as SIMT).
// C: [M, N] FP32 output. Optional bias: [N] FP32.

#include "embed/ops_tile/ops_tile_common.h"

namespace luisa::embed_mm::ops_tile {

struct GemmTileConfig {
    int64_t block_m = 64;
    int64_t block_n = 64;
    int64_t block_k = 32;
};

[[nodiscard]] GemmTileConfig select_gemm_tile_config(uint32_t M, uint32_t N, uint32_t K) noexcept;

[[nodiscard]] compute::tile::Kernel make_tile_gemm_kernel(uint32_t M, uint32_t N, uint32_t K, bool with_bias) noexcept;

[[nodiscard]] TileCmd dispatch_tile_gemm(
    compute::Device &device, TileShaderCache &cache,
    compute::BufferView<float> A, compute::BufferView<luisa::bfloat16> B, compute::BufferView<float> C,
    uint32_t M, uint32_t N, uint32_t K);

[[nodiscard]] TileCmd dispatch_tile_gemm_bias(
    compute::Device &device, TileShaderCache &cache,
    compute::BufferView<float> A, compute::BufferView<luisa::bfloat16> B,
    compute::BufferView<float> bias, compute::BufferView<float> C,
    uint32_t M, uint32_t N, uint32_t K);

}// namespace luisa::embed_mm::ops_tile
