#pragma once
// embedding_mm engine -- Tile normalization: RMSNorm, RMSNormNoWeight, L2Normalize.
// One program per row; non power-of-two widths use a bit_ceil tile whose tail is
// zero-masked by tensor bounds (zeros do not perturb the sums).

#include "embed/ops_tile/ops_tile_common.h"

namespace luisa::embed_mm::ops_tile {

// x == y (same region) automatically selects the in-place variant.
[[nodiscard]] TileCmd dispatch_tile_rmsnorm(
    compute::Device &device, TileShaderCache &cache,
    compute::BufferView<float> x, compute::BufferView<float> y, compute::BufferView<float> gamma,
    uint32_t rows, uint32_t H, float eps);

[[nodiscard]] TileCmd dispatch_tile_rmsnorm_no_weight(
    compute::Device &device, TileShaderCache &cache,
    compute::BufferView<float> x, compute::BufferView<float> y,
    uint32_t rows, uint32_t H, float eps);

[[nodiscard]] TileCmd dispatch_tile_l2_normalize(
    compute::Device &device, TileShaderCache &cache,
    compute::BufferView<float> x, compute::BufferView<float> y,
    uint32_t rows, uint32_t D);

}// namespace luisa::embed_mm::ops_tile
