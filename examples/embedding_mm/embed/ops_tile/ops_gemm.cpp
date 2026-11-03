#include "embed/ops_tile/ops_gemm.h"

namespace luisa::embed_mm::ops_tile {

GemmTileConfig select_gemm_tile_config(uint32_t M, uint32_t N, uint32_t K, bool has_workspace) noexcept {
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
    cfg.k_chunks = 1u;

    if (!has_workspace) {
        return cfg;
    }

    uint32_t g_m = luisa::ceil_div(M, static_cast<uint32_t>(cfg.block_m));
    uint32_t g_n = luisa::ceil_div(N, static_cast<uint32_t>(cfg.block_n));
    uint32_t g_spatial = g_m * g_n;

    // GA102 has 82 SMs. Target >= 82 total blocks for full occupancy.
    // Ensure K is evenly divisible by (k_chunks * block_k).
    if (g_spatial < 82u) {
        auto bk = static_cast<uint32_t>(cfg.block_k);
        if (g_spatial * 8u <= 256u && (K % (bk * 8u) == 0u)) {
            cfg.k_chunks = 8u;
        } else if (g_spatial * 4u <= 256u && (K % (bk * 4u) == 0u)) {
            cfg.k_chunks = 4u;
        } else if (g_spatial * 2u <= 256u && (K % (bk * 2u) == 0u)) {
            cfg.k_chunks = 2u;
        }
    }
    return cfg;
}

compute::tile::Kernel make_tile_gemm_kernel(uint32_t M, uint32_t N, uint32_t K, bool with_bias) noexcept {
    using namespace compute::tile;
    const auto cfg = select_gemm_tile_config(M, N, K, false);
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

compute::tile::Kernel make_tile_gemm_splitk_kernel(uint32_t M, uint32_t N, uint32_t K, uint32_t k_chunks) noexcept {
    using namespace compute::tile;
    const auto cfg = select_gemm_tile_config(M, N, K, true);
    const int64_t m_ext = M, n_ext = N, k_ext = K;
    const int64_t bm = cfg.block_m, bn = cfg.block_n, bk = cfg.block_k;
    const int64_t k_slice = k_ext / k_chunks;
    const int64_t k_steps = luisa::ceil_div(k_slice, bk);
    const int64_t m_stride = luisa::ceil_div(m_ext, bm) * bm;

    auto body = [=](TensorView<const float, 2> A, TensorView<const bfloat16, 2> B, TensorView<float, 2> W) {
        auto gm = axis("gm", luisa::ceil_div(m_ext, bm));
        auto gn = axis("gn", luisa::ceil_div(n_ext, bn));
        auto gk = axis("gk", k_chunks);
        auto m = axis("m", bm), n = axis("n", bn), k = axis("k", bk);

        for (auto &nest : parallel(shape(gm, gn, gk))) {
            auto row = nest.index(gm) * bm;
            auto col = nest.index(gn) * bn;
            auto chunk_idx = nest.index(gk);
            auto k_offset = chunk_idx * k_slice;

            auto acc = zeros<float>(shape(m, n));
            for (auto &step : nest.pipeline(shape(k_steps), {.window = 1u, .interval = 1u})) {
                step.stage("load");
                auto a = A.tile(coord(row, k_offset + step.index() * bk), shape(m, k)).load();
                auto b = B.tile(coord(k_offset + step.index() * bk, col), shape(k, n)).load();
                step.stage("compute");
                acc = mma(cast<bfloat16>(a), b, acc);
            }
            W(coord(chunk_idx * m_stride + row, col), shape(m, n)).store(acc);
        }
    };
    return tile_kernel("embed_mm_tile_gemm_splitk", body)
        .capture(tensor_shape(m_ext, k_ext), tensor_shape(k_ext, n_ext), tensor_shape(k_chunks * m_stride, n_ext));
}

compute::tile::Kernel make_tile_gemm_reduce_kernel(uint32_t M, uint32_t N, uint32_t k_chunks, bool with_bias) noexcept {
    using namespace compute::tile;
    const auto cfg = select_gemm_tile_config(M, N, 768u, true);
    const int64_t m_ext = M, n_ext = N;
    const int64_t bm = cfg.block_m, bn = cfg.block_n;
    const int64_t m_stride = luisa::ceil_div(m_ext, bm) * bm;

    if (with_bias) {
        auto body = [=](TensorView<const float, 2> W, TensorView<const float, 1> bias, TensorView<float, 2> C) {
            auto gm = axis("gm", luisa::ceil_div(m_ext, bm));
            auto gn = axis("gn", luisa::ceil_div(n_ext, bn));
            auto m = axis("m", bm), n = axis("n", bn);

            for (auto &nest : parallel(shape(gm, gn))) {
                auto row = nest.index(gm) * bm;
                auto col = nest.index(gn) * bn;
                auto acc = zeros<float>(shape(m, n));

                for (int64_t s = 0; s < static_cast<int64_t>(k_chunks); ++s) {
                    auto partial = W.tile(coord(s * m_stride + row, col), shape(m, n)).load();
                    acc = acc + partial;
                }
                acc = acc + bias.tile(coord(col), shape(n)).load();
                C(coord(row, col), shape(m, n)).store(acc);
            }
        };
        return tile_kernel("embed_mm_tile_gemm_reduce_bias", body)
            .capture(tensor_shape(k_chunks * m_stride, n_ext), tensor_shape(n_ext), tensor_shape(m_ext, n_ext));
    }

    auto nobias = [=](TensorView<const float, 2> W, TensorView<float, 2> C) {
        auto gm = axis("gm", luisa::ceil_div(m_ext, bm));
        auto gn = axis("gn", luisa::ceil_div(n_ext, bn));
        auto m = axis("m", bm), n = axis("n", bn);

        for (auto &nest : parallel(shape(gm, gn))) {
            auto row = nest.index(gm) * bm;
            auto col = nest.index(gn) * bn;
            auto acc = zeros<float>(shape(m, n));

            for (int64_t s = 0; s < static_cast<int64_t>(k_chunks); ++s) {
                auto partial = W.tile(coord(s * m_stride + row, col), shape(m, n)).load();
                acc = acc + partial;
            }
            C(coord(row, col), shape(m, n)).store(acc);
        }
    };
    return tile_kernel("embed_mm_tile_gemm_reduce", nobias)
        .capture(tensor_shape(k_chunks * m_stride, n_ext), tensor_shape(m_ext, n_ext));
}

compute::CommandList dispatch_tile_gemm(
    compute::Device &device, TileShaderCache &cache,
    compute::BufferView<float> A, compute::BufferView<luisa::bfloat16> B, compute::BufferView<float> C,
    uint32_t M, uint32_t N, uint32_t K,
    compute::BufferView<float> workspace) {

    auto cfg = select_gemm_tile_config(M, N, K, static_cast<bool>(workspace));
    const int64_t m_stride = luisa::ceil_div(static_cast<int64_t>(M), cfg.block_m) * cfg.block_m;
    const size_t req_workspace = static_cast<size_t>(cfg.k_chunks) * m_stride * N;
    if (workspace.size() < req_workspace) {
        cfg.k_chunks = 1u;
    }

    compute::CommandList cmd_list;
    if (cfg.k_chunks > 1u) {
        TileShaderKey splitk_key{make_op_tag("gemm_splitk"), M, N, K, cfg.k_chunks};
        const auto &splitk_shader = cache.get_or_compile(device, splitk_key, [&] {
            return make_tile_gemm_splitk_kernel(M, N, K, cfg.k_chunks);
        });
        TileShaderKey reduce_key{make_op_tag("gemm_reduce"), M, N, cfg.k_chunks, 0ull};
        const auto &reduce_shader = cache.get_or_compile(device, reduce_key, [&] {
            return make_tile_gemm_reduce_kernel(M, N, cfg.k_chunks, false);
        });
        auto w_view = workspace.subview(0, req_workspace);
        cmd_list.append(splitk_shader(A, B, w_view).dispatch());
        cmd_list.append(reduce_shader(w_view, C).dispatch());
    } else {
        TileShaderKey key{make_op_tag("gemm"), M, N, K};
        const auto &shader = cache.get_or_compile(device, key, [&] {
            return make_tile_gemm_kernel(M, N, K, false);
        });
        cmd_list.append(shader(A, B, C).dispatch());
    }
    return cmd_list;
}

compute::CommandList dispatch_tile_gemm_bias(
    compute::Device &device, TileShaderCache &cache,
    compute::BufferView<float> A, compute::BufferView<luisa::bfloat16> B,
    compute::BufferView<float> bias, compute::BufferView<float> C,
    uint32_t M, uint32_t N, uint32_t K,
    compute::BufferView<float> workspace) {

    auto cfg = select_gemm_tile_config(M, N, K, static_cast<bool>(workspace));
    const int64_t m_stride = luisa::ceil_div(static_cast<int64_t>(M), cfg.block_m) * cfg.block_m;
    const size_t req_workspace = static_cast<size_t>(cfg.k_chunks) * m_stride * N;
    if (workspace.size() < req_workspace) {
        cfg.k_chunks = 1u;
    }

    compute::CommandList cmd_list;
    if (cfg.k_chunks > 1u) {
        TileShaderKey splitk_key{make_op_tag("gemm_splitk"), M, N, K, cfg.k_chunks};
        const auto &splitk_shader = cache.get_or_compile(device, splitk_key, [&] {
            return make_tile_gemm_splitk_kernel(M, N, K, cfg.k_chunks);
        });
        TileShaderKey reduce_key{make_op_tag("gemm_reduce_bias"), M, N, cfg.k_chunks, 1ull};
        const auto &reduce_shader = cache.get_or_compile(device, reduce_key, [&] {
            return make_tile_gemm_reduce_kernel(M, N, cfg.k_chunks, true);
        });
        auto w_view = workspace.subview(0, req_workspace);
        cmd_list.append(splitk_shader(A, B, w_view).dispatch());
        cmd_list.append(reduce_shader(w_view, bias, C).dispatch());
    } else {
        TileShaderKey key{make_op_tag("gemm_bias"), M, N, K};
        const auto &shader = cache.get_or_compile(device, key, [&] {
            return make_tile_gemm_kernel(M, N, K, true);
        });
        cmd_list.append(shader(A, B, bias, C).dispatch());
    }
    return cmd_list;
}

}// namespace luisa::embed_mm::ops_tile
