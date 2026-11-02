#include "embed/ops_tile/ops_gemm.h"

namespace luisa::embed_mm::ops_tile {

GemmTileConfig select_gemm_tile_config(uint32_t M, uint32_t N, uint32_t K) noexcept {
    GemmTileConfig cfg{};
    if (M % 64u == 0u) {
        cfg.block_m = 64;
    } else if (M % 32u == 0u) {
        cfg.block_m = 32;
    } else if (M % 16u == 0u) {
        cfg.block_m = 16;
    } else {
        cfg.block_m = (M <= 16u) ? 16 : (M <= 32u ? 32 : 64);
    }
    cfg.block_n = (N % 64u == 0u) ? 64 : 32;
    cfg.block_k = (K % 32u == 0u) ? 32 : 16;
    return cfg;
}

compute::tile::Kernel make_tile_gemm_kernel(uint32_t M, uint32_t N, uint32_t K, bool with_bias) noexcept {
    using namespace compute::tile;
    const auto cfg = select_gemm_tile_config(M, N, K);
    const int64_t m_ext = M, n_ext = N, k_ext = K;
    const int64_t bm = cfg.block_m, bn = cfg.block_n, bk = cfg.block_k;
    const int64_t k_steps = luisa::ceil_div(k_ext, bk);

    auto body = [=](TensorView<const float, 2> A, TensorView<const bfloat16, 2> B,
                    TensorView<const float, 1> bias, TensorView<float, 2> C) {
        auto gm = axis("gm", luisa::ceil_div(m_ext, bm));
        auto gn = axis("gn", luisa::ceil_div(n_ext, bn));
        auto m = axis("m", bm), n = axis("n", bn), k = axis("k", bk);
        for (auto &nest : parallel(shape(gm, gn))) {
            auto row = nest.index(gm) * bm;
            auto col = nest.index(gn) * bn;
            auto acc = zeros<float>(shape(m, n));
            for (auto &step : nest.pipeline(shape(k_steps), {.window = 1u, .interval = 1u})) {
                step.stage("load");
                auto a = A.tile(coord(row, step.index() * bk), shape(m, k)).load();
                auto b = B.tile(coord(step.index() * bk, col), shape(k, n)).load();
                step.stage("compute");
                acc = mma(cast<bfloat16>(a), b, acc);
            }
            if (with_bias) {
                acc = acc + bias.tile(coord(col), shape(n)).load();
            }
            C(coord(row, col), shape(m, n)).store(acc);
        }
    };
    if (with_bias) {
        return tile_kernel("embed_mm_tile_gemm_bias", body)
            .capture(tensor_shape(m_ext, k_ext), tensor_shape(k_ext, n_ext), tensor_shape(n_ext), tensor_shape(m_ext, n_ext));
    }
    auto nobias = [=](TensorView<const float, 2> A, TensorView<const bfloat16, 2> B, TensorView<float, 2> C) {
        auto gm = axis("gm", luisa::ceil_div(m_ext, bm));
        auto gn = axis("gn", luisa::ceil_div(n_ext, bn));
        auto m = axis("m", bm), n = axis("n", bn), k = axis("k", bk);
        for (auto &nest : parallel(shape(gm, gn))) {
            auto row = nest.index(gm) * bm;
            auto col = nest.index(gn) * bn;
            auto acc = zeros<float>(shape(m, n));
            for (auto &step : nest.pipeline(shape(k_steps), {.window = 1u, .interval = 1u})) {
                step.stage("load");
                auto a = A.tile(coord(row, step.index() * bk), shape(m, k)).load();
                auto b = B.tile(coord(step.index() * bk, col), shape(k, n)).load();
                step.stage("compute");
                acc = mma(cast<bfloat16>(a), b, acc);
            }
            C(coord(row, col), shape(m, n)).store(acc);
        }
    };
    return tile_kernel("embed_mm_tile_gemm", nobias)
        .capture(tensor_shape(m_ext, k_ext), tensor_shape(k_ext, n_ext), tensor_shape(m_ext, n_ext));
}

TileCmd dispatch_tile_gemm(compute::Device &device, TileShaderCache &cache,
                           compute::BufferView<float> A, compute::BufferView<luisa::bfloat16> B,
                           compute::BufferView<float> C, uint32_t M, uint32_t N, uint32_t K) {
    TileShaderKey key{make_op_tag("gemm"), M, N, K};
    const auto &shader = cache.get_or_compile(device, key, [&] { return make_tile_gemm_kernel(M, N, K, false); });
    return shader(A, B, C).dispatch();
}

TileCmd dispatch_tile_gemm_bias(compute::Device &device, TileShaderCache &cache,
                                compute::BufferView<float> A, compute::BufferView<luisa::bfloat16> B,
                                compute::BufferView<float> bias, compute::BufferView<float> C,
                                uint32_t M, uint32_t N, uint32_t K) {
    TileShaderKey key{make_op_tag("gemm_bias"), M, N, K};
    const auto &shader = cache.get_or_compile(device, key, [&] { return make_tile_gemm_kernel(M, N, K, true); });
    return shader(A, B, bias, C).dispatch();
}

}// namespace luisa::embed_mm::ops_tile
