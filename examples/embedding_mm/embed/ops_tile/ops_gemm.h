#pragma once
// embedding_mm engine -- Tile GEMM (BF16 Tensor Core MMA, FP32 accumulation).
//
// A: [M, K] FP32 row-major activations (cast to BF16 per tile in registers).
// B: [K, N] BF16 row-major weights (pre-transposed offline, same layout as SIMT).
// C: [M, N] FP32 output. Optional bias: [N] FP32.

#include "embed/ops_tile/ops_tile_common.h"
#include <luisa/runtime/command_list.h>

namespace luisa::embed_mm::ops_tile {

struct GemmTileConfig {
    int64_t block_m = 64;
    int64_t block_n = 64;
    int64_t block_k = 32;
    uint32_t k_chunks = 1;
};

[[nodiscard]] GemmTileConfig select_gemm_tile_config(uint32_t M, uint32_t N, uint32_t K, bool has_workspace = false) noexcept;

[[nodiscard]] compute::tile::Kernel make_tile_gemm_kernel(uint32_t M, uint32_t N, uint32_t K, bool with_bias) noexcept;
[[nodiscard]] compute::tile::Kernel make_tile_gemm_splitk_kernel(uint32_t M, uint32_t N, uint32_t K, uint32_t k_chunks) noexcept;
[[nodiscard]] compute::tile::Kernel make_tile_gemm_reduce_kernel(uint32_t M, uint32_t N, uint32_t k_chunks, bool with_bias) noexcept;

[[nodiscard]] compute::CommandList dispatch_tile_gemm(
    compute::Device &device, TileShaderCache &cache,
    compute::BufferView<float> A, compute::BufferView<luisa::bfloat16> B, compute::BufferView<float> C,
    uint32_t M, uint32_t N, uint32_t K,
    compute::BufferView<float> workspace = {});

[[nodiscard]] compute::CommandList dispatch_tile_gemm_bias(
    compute::Device &device, TileShaderCache &cache,
    compute::BufferView<float> A, compute::BufferView<luisa::bfloat16> B,
    compute::BufferView<float> bias, compute::BufferView<float> C,
    uint32_t M, uint32_t N, uint32_t K,
    compute::BufferView<float> workspace = {});

}// namespace luisa::embed_mm::ops_tile
