#include "embed/ops_tile/ops_pool.h"

#include <bit>

namespace luisa::embed_mm::ops_tile {

compute::tile::Kernel make_tile_vision_pool_kernel(
    uint32_t num_patches,
    uint32_t grid_w_soft,
    uint32_t grid_h_soft,
    uint32_t patch_stride_w,
    uint32_t H,
    float scale_factor) noexcept {
    using namespace compute::tile;
    const int64_t p_ext = static_cast<int64_t>(num_patches);
    const int64_t gw = static_cast<int64_t>(grid_w_soft);
    const int64_t gh = static_cast<int64_t>(grid_h_soft);
    const int64_t total_soft = gw * gh;
    const int64_t stride_w = static_cast<int64_t>(patch_stride_w);
    const int64_t h_ext = static_cast<int64_t>(H);
    const int64_t bh = 64;
    const int64_t gh_tiles = luisa::ceil_div(h_ext, bh);

    auto def = tile_kernel("embed_mm_tile_vision_pool", [=](
        TensorView<const float, 2> in_hidden,
        TensorView<float, 2> out_pooled) {
        auto m = axis("m", 1);
        auto n = axis("n", bh);
        auto s_grid = axis("s_grid", total_soft);
        auto h_grid = axis("h_grid", gh_tiles);
        for (auto &nest : parallel(shape(s_grid, h_grid))) {
            auto s = nest.index(s_grid);
            auto col = nest.index(h_grid) * bh;
            auto x_prime = s % gw;
            auto y_prime = s / gw;
            auto sum = zeros<float>(shape(m, n));
            for (int64_t dy = 0; dy < 3; ++dy) {
                for (int64_t dx = 0; dx < 3; ++dx) {
                    auto px = x_prime * 3 + dx;
                    auto py = y_prime * 3 + dy;
                    auto patch_idx = py * stride_w + px;
                    sum = sum + in_hidden.tile(coord(patch_idx, col), shape(m, n)).load();
                }
            }
            out_pooled(coord(s, col), shape(m, n)).store(sum * scale_factor);
        }
    });
    return def.capture(tensor_shape(p_ext, h_ext), tensor_shape(total_soft, h_ext));
}

compute::tile::Kernel make_tile_mean_pool_kernel(uint32_t B, uint32_t S, uint32_t H) noexcept {
    using namespace compute::tile;
    const int64_t b_ext = static_cast<int64_t>(B);
    const int64_t s_ext = static_cast<int64_t>(S);
    const int64_t h_ext = static_cast<int64_t>(H);
    const int64_t bh = 64;
    const int64_t gh_tiles = luisa::ceil_div(h_ext, bh);

    auto def = tile_kernel("embed_mm_tile_mean_pool", [=](
        TensorView<const float, 3> hidden,
        TensorView<const float, 2> mask,
        TensorView<float, 2> pooled) {
        auto b_axis = axis("b_axis", 1);
        auto s_axis = axis("s_axis", 1);
        auto n = axis("n", bh);

        auto b_grid = axis("b_grid", b_ext);
        auto h_grid = axis("h_grid", gh_tiles);

        for (auto &nest : parallel(shape(b_grid, h_grid))) {
            auto b = nest.index(b_grid);
            auto col = nest.index(h_grid) * bh;
            auto sum_vec = zeros<float>(shape(b_axis, s_axis, n));
            auto count_vec = zeros<float>(shape(b_axis, s_axis));

            for (auto &step : nest.serial(shape(s_ext))) {
                auto s = step.index();
                auto m_tile = mask.tile(coord(b, s), shape(b_axis, s_axis)).load();
                auto h_tile = hidden.tile(coord(b, s, col), shape(b_axis, s_axis, n)).load();
                auto m_bcast = map<float>(shape(b_axis, s_axis, n), [&](const Nest &nst) {
                    return m_tile.at(coord(nst.index(b_axis), nst.index(s_axis)));
                });
                sum_vec = sum_vec + h_tile * m_bcast;
                count_vec = count_vec + m_tile;
            }
            auto safe_count = max(count_vec, 1.0f);
            auto out_2d = map<float>(shape(b_axis, n), [&](const Nest &nst) {
                return sum_vec.at(coord(nst.index(b_axis), 0, nst.index(n))) / safe_count.at(coord(nst.index(b_axis), 0));
            });
            pooled(coord(b, col), shape(b_axis, n)).store(out_2d);
        }
    });
    return def.capture(
        tensor_shape(b_ext, s_ext, h_ext),
        tensor_shape(b_ext, s_ext),
        tensor_shape(b_ext, h_ext));
}

compute::tile::Kernel make_tile_mrl_normalize_kernel(
    uint32_t B, uint32_t D_proj, uint32_t target_dim) noexcept {
    using namespace compute::tile;
    const int64_t b_ext = static_cast<int64_t>(B);
    const int64_t d_proj_ext = static_cast<int64_t>(D_proj);
    const int64_t target_ext = static_cast<int64_t>(target_dim);
    const int64_t pad_target = next_pow2(target_dim);

    auto def = tile_kernel("embed_mm_tile_mrl_normalize", [=](
        TensorView<const float, 2> in_proj,
        TensorView<float, 2> out_embed) {
        auto m = axis("m", 1);
        auto n = axis("n", pad_target);
        for (auto &nest : parallel(shape(b_ext))) {
            auto b = nest.index();
            auto x = in_proj.tile(coord(b, 0), shape(m, n)).load();
            if (target_ext == pad_target) {
                auto sum_sq = reduce(x * x, n, add);
                auto norm = max(sqrt(sum_sq), 1e-12f);
                out_embed(coord(b, 0), shape(m, n)).store(x / norm);
            } else {
                auto valid = iota(n) < target_ext;
                auto x_valid = ite(valid, x, 0.0f);
                auto sum_sq = reduce(x_valid * x_valid, n, add);
                auto norm = max(sqrt(sum_sq), 1e-12f);
                out_embed(coord(b, 0), shape(m, n)).store(ite(valid, x / norm, 0.0f));
            }
        }
    });
    return def.capture(tensor_shape(b_ext, d_proj_ext), tensor_shape(b_ext, target_ext));
}

TileCmd dispatch_tile_vision_pool(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> in_hidden,
    compute::BufferView<float> out_pooled,
    uint32_t num_patches,
    uint32_t grid_w_soft,
    uint32_t grid_h_soft,
    uint32_t patch_stride_w,
    uint32_t H,
    float scale_factor) {
    const uint64_t sf_bits = std::bit_cast<uint32_t>(scale_factor);
    TileShaderKey key{
        make_op_tag("vision_pool"),
        num_patches,
        (static_cast<uint64_t>(grid_w_soft) << 32u) | grid_h_soft,
        patch_stride_w,
        H,
        sf_bits};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_vision_pool_kernel(num_patches, grid_w_soft, grid_h_soft, patch_stride_w, H, scale_factor);
    });
    return shader(in_hidden, out_pooled).dispatch();
}

TileCmd dispatch_tile_mean_pool(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> hidden,
    compute::BufferView<float> mask,
    compute::BufferView<float> pooled,
    uint32_t B, uint32_t S, uint32_t H) {
    TileShaderKey key{make_op_tag("mean_pool"), B, S, H, 0, 0};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_mean_pool_kernel(B, S, H);
    });
    return shader(hidden, mask, pooled).dispatch();
}

TileCmd dispatch_tile_mrl_normalize(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> in_proj,
    compute::BufferView<float> out_embed,
    uint32_t B, uint32_t D_proj, uint32_t target_dim) {
    TileShaderKey key{make_op_tag("mrl_normalize"), B, D_proj, target_dim, 0, 0};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_mrl_normalize_kernel(B, D_proj, target_dim);
    });
    return shader(in_proj, out_embed).dispatch();
}

}// namespace luisa::embed_mm::ops_tile
