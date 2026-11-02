#include "embed/ops_tile/ops_embed.h"

#include <bit>

namespace luisa::embed_mm::ops_tile {

compute::tile::Kernel make_tile_embed_lookup_bf16_kernel(
    uint32_t total_tokens, uint32_t vocab_size, uint32_t H, float scale) noexcept {
    using namespace compute::tile;
    const int64_t t_ext = static_cast<int64_t>(total_tokens);
    const int64_t v_ext = static_cast<int64_t>(vocab_size);
    const int64_t h_ext = static_cast<int64_t>(H);
    const int64_t bh = 128;
    const int64_t h_tiles = luisa::ceil_div(h_ext, bh);

    auto def = tile_kernel("embed_mm_tile_embed_lookup_bf16", [=](
        TensorView<const uint32_t, 1> input_ids,
        TensorView<const luisa::bfloat16, 2> table,
        TensorView<float, 2> out) {
        auto one_id = axis("one_id", 1);
        auto m = axis("m", 1);
        auto n = axis("n", bh);
        auto t_grid = axis("t_grid", t_ext);
        auto h_grid = axis("h_grid", h_tiles);
        for (auto &nest : parallel(shape(t_grid, h_grid))) {
            auto tok = nest.index(t_grid);
            auto col = nest.index(h_grid) * bh;
            auto id_tile = input_ids.tile(coord(tok), shape(one_id)).load();
            auto id = cast<int64_t>(id_tile.at(coord(0)));
            auto w_bf16 = table.tile(coord(id, col), shape(m, n)).load();
            auto w_fp32 = cast<float>(w_bf16) * scale;
            out(coord(tok, col), shape(m, n)).store(w_fp32);
        }
    });
    return def.capture(tensor_shape(t_ext), tensor_shape(v_ext, h_ext), tensor_shape(t_ext, h_ext));
}

compute::tile::Kernel make_tile_embed_lookup_fp32_kernel(
    uint32_t total_tokens, uint32_t vocab_size, uint32_t H, float scale) noexcept {
    using namespace compute::tile;
    const int64_t t_ext = static_cast<int64_t>(total_tokens);
    const int64_t v_ext = static_cast<int64_t>(vocab_size);
    const int64_t h_ext = static_cast<int64_t>(H);
    const int64_t bh = 128;
    const int64_t h_tiles = luisa::ceil_div(h_ext, bh);

    auto def = tile_kernel("embed_mm_tile_embed_lookup_fp32", [=](
        TensorView<const uint32_t, 1> input_ids,
        TensorView<const float, 2> table,
        TensorView<float, 2> out) {
        auto one_id = axis("one_id", 1);
        auto m = axis("m", 1);
        auto n = axis("n", bh);
        auto t_grid = axis("t_grid", t_ext);
        auto h_grid = axis("h_grid", h_tiles);
        for (auto &nest : parallel(shape(t_grid, h_grid))) {
            auto tok = nest.index(t_grid);
            auto col = nest.index(h_grid) * bh;
            auto id_tile = input_ids.tile(coord(tok), shape(one_id)).load();
            auto id = cast<int64_t>(id_tile.at(coord(0)));
            auto w_fp32 = table.tile(coord(id, col), shape(m, n)).load() * scale;
            out(coord(tok, col), shape(m, n)).store(w_fp32);
        }
    });
    return def.capture(tensor_shape(t_ext), tensor_shape(v_ext, h_ext), tensor_shape(t_ext, h_ext));
}

compute::tile::Kernel make_tile_scale_pixels_kernel(uint32_t total) noexcept {
    using namespace compute::tile;
    const int64_t n_ext = static_cast<int64_t>(total);
    constexpr int64_t kVec = 256;
    const int64_t blocks = luisa::ceil_div(n_ext, kVec);

    auto def = tile_kernel("embed_mm_tile_scale_pixels", [=](
        TensorView<const float, 1> in_pixels,
        TensorView<float, 1> out_pixels) {
        auto v = axis("v", kVec);
        for (auto &nest : parallel(shape(blocks))) {
            auto base = nest.index() * kVec;
            auto p = in_pixels.tile(coord(base), shape(v)).load();
            out_pixels(coord(base), shape(v)).store(2.0f * (p - 0.5f));
        }
    });
    return def.capture(tensor_shape(n_ext), tensor_shape(n_ext));
}

compute::tile::Kernel make_tile_add_patch_pos_embed_kernel(
    uint32_t num_patches, uint32_t H, uint32_t pos_table_size) noexcept {
    using namespace compute::tile;
    const int64_t p_ext = static_cast<int64_t>(num_patches);
    const int64_t h_ext = static_cast<int64_t>(H);
    const int64_t tab_ext = static_cast<int64_t>(pos_table_size);
    const uint32_t tab_u32 = pos_table_size;
    const int64_t bh = 128;
    const int64_t h_tiles = luisa::ceil_div(h_ext, bh);

    auto def = tile_kernel("embed_mm_tile_add_patch_pos_embed", [=](
        TensorView<float, 2> hidden,
        TensorView<const float, 3> table,
        TensorView<const uint32_t, 2> pos_ids) {
        auto one_p = axis("one_p", 1);
        auto one_c = axis("one_c", 1);
        auto n = axis("n", bh);

        auto p_grid = axis("p_grid", p_ext);
        auto h_grid = axis("h_grid", h_tiles);

        for (auto &nest : parallel(shape(p_grid, h_grid))) {
            auto p = nest.index(p_grid);
            auto col = nest.index(h_grid) * bh;
            auto px_tile = pos_ids.tile(coord(p, 0), shape(one_p, one_c)).load();
            auto py_tile = pos_ids.tile(coord(p, 1), shape(one_p, one_c)).load();
            auto valid_1x1 = (px_tile < tab_u32) && (py_tile < tab_u32);
            auto safe_px = cast<int64_t>(ite(valid_1x1, px_tile, 0u).at(coord(0, 0)));
            auto safe_py = cast<int64_t>(ite(valid_1x1, py_tile, 0u).at(coord(0, 0)));

            auto h_val = hidden.tile(coord(p, col), shape(one_p, n)).load();
            auto x_3d = table.tile(coord(0, safe_px, col), shape(one_c, one_p, n)).load();
            auto y_3d = table.tile(coord(1, safe_py, col), shape(one_c, one_p, n)).load();
            auto x_2d = map<float>(shape(one_p, n), [&](const Nest &nst) { return x_3d.at(coord(0, nst.index(one_p), nst.index(n))); });
            auto y_2d = map<float>(shape(one_p, n), [&](const Nest &nst) { return y_3d.at(coord(0, nst.index(one_p), nst.index(n))); });
            auto valid_bcast = map<bool>(shape(one_p, n), [&](const Nest &nst) { return valid_1x1.at(coord(nst.index(one_p), 0)); });
            auto res = ite(valid_bcast, h_val + x_2d + y_2d, 0.0f);
            hidden(coord(p, col), shape(one_p, n)).store(res);
        }
    });
    return def.capture(
        tensor_shape(p_ext, h_ext),
        tensor_shape(2, tab_ext, h_ext),
        tensor_shape(p_ext, 2));
}

compute::tile::Kernel make_tile_mask_from_pos_kernel(
    uint32_t num_patches, uint32_t pos_table_size) noexcept {
    using namespace compute::tile;
    const int64_t p_ext = static_cast<int64_t>(num_patches);
    const uint32_t tab_u32 = pos_table_size;
    constexpr int64_t kBlock = 64;
    const int64_t blocks = luisa::ceil_div(p_ext, kBlock);

    auto def = tile_kernel("embed_mm_tile_mask_from_pos", [=](
        TensorView<const uint32_t, 2> pos_ids,
        TensorView<float, 2> out_mask) {
        auto m = axis("m", kBlock);
        auto one_c = axis("one_c", 1);
        for (auto &nest : parallel(shape(blocks))) {
            auto base = nest.index() * kBlock;
            auto px = pos_ids.tile(coord(base, 0), shape(m, one_c)).load();
            auto mask = ite(px < tab_u32, 1.0f, 0.0f);
            out_mask(coord(base, 0), shape(m, one_c)).store(mask);
        }
    });
    return def.capture(tensor_shape(p_ext, 2), tensor_shape(p_ext, 1));
}

compute::tile::Kernel make_tile_scatter_soft_tokens_kernel(
    uint32_t total_tokens, uint32_t start_token_idx, uint32_t num_soft_tokens, uint32_t H) noexcept {
    using namespace compute::tile;
    const int64_t tot_ext = static_cast<int64_t>(total_tokens);
    const int64_t start_idx = static_cast<int64_t>(start_token_idx);
    const int64_t soft_ext = static_cast<int64_t>(num_soft_tokens);
    const int64_t h_ext = static_cast<int64_t>(H);
    const int64_t bh = 128;
    const int64_t h_tiles = luisa::ceil_div(h_ext, bh);

    auto def = tile_kernel("embed_mm_tile_scatter_soft_tokens", [=](
        TensorView<float, 2> h0,
        TensorView<const float, 2> soft_tokens) {
        auto m = axis("m", 1);
        auto n = axis("n", bh);
        auto s_grid = axis("s_grid", soft_ext);
        auto h_grid = axis("h_grid", h_tiles);
        for (auto &nest : parallel(shape(s_grid, h_grid))) {
            auto i = nest.index(s_grid);
            auto col = nest.index(h_grid) * bh;
            auto s_val = soft_tokens.tile(coord(i, col), shape(m, n)).load();
            h0(coord(start_idx + i, col), shape(m, n)).store(s_val);
        }
    });
    return def.capture(tensor_shape(tot_ext, h_ext), tensor_shape(soft_ext, h_ext));
}

compute::tile::Kernel make_tile_scatter_indexed_tokens_kernel(
    uint32_t total_tokens, uint32_t num_soft_tokens, uint32_t H) noexcept {
    using namespace compute::tile;
    const int64_t tot_ext = static_cast<int64_t>(total_tokens);
    const int64_t soft_ext = static_cast<int64_t>(num_soft_tokens);
    const int64_t h_ext = static_cast<int64_t>(H);
    const int64_t bh = 128;
    const int64_t h_tiles = luisa::ceil_div(h_ext, bh);

    auto def = tile_kernel("embed_mm_tile_scatter_indexed_tokens", [=](
        TensorView<float, 2> h0,
        TensorView<const float, 2> soft_tokens,
        TensorView<const uint32_t, 1> token_indices) {
        auto one_id = axis("one_id", 1);
        auto m = axis("m", 1);
        auto n = axis("n", bh);
        auto s_grid = axis("s_grid", soft_ext);
        auto h_grid = axis("h_grid", h_tiles);
        for (auto &nest : parallel(shape(s_grid, h_grid))) {
            auto i = nest.index(s_grid);
            auto col = nest.index(h_grid) * bh;
            auto dst_tok = cast<int64_t>(token_indices.tile(coord(i), shape(one_id)).load().at(coord(0)));
            auto s_val = soft_tokens.tile(coord(i, col), shape(m, n)).load();
            h0(coord(dst_tok, col), shape(m, n)).store(s_val);
        }
    });
    return def.capture(
        tensor_shape(tot_ext, h_ext),
        tensor_shape(soft_ext, h_ext),
        tensor_shape(soft_ext));
}

TileCmd dispatch_tile_embed_lookup(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<uint32_t> input_ids,
    compute::BufferView<luisa::bfloat16> table,
    compute::BufferView<float> out,
    uint32_t total_tokens,
    uint32_t vocab_size,
    uint32_t H,
    float scale) {
    const uint64_t scale_bits = std::bit_cast<uint32_t>(scale);
    TileShaderKey key{make_op_tag("embed_lookup_bf16"), total_tokens, vocab_size, H, scale_bits, 0};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_embed_lookup_bf16_kernel(total_tokens, vocab_size, H, scale);
    });
    return shader(input_ids, table, out).dispatch();
}

TileCmd dispatch_tile_scale_pixels(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> in_pixels,
    compute::BufferView<float> out_pixels,
    uint32_t total) {
    TileShaderKey key{make_op_tag("scale_pixels"), total, 0, 0, 0, 0};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_scale_pixels_kernel(total);
    });
    return shader(in_pixels, out_pixels).dispatch();
}

TileCmd dispatch_tile_add_patch_pos_embed(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> hidden,
    compute::BufferView<float> table,
    compute::BufferView<uint32_t> pos_ids,
    uint32_t num_patches,
    uint32_t H,
    uint32_t pos_table_size) {
    TileShaderKey key{make_op_tag("add_patch_pos"), num_patches, H, pos_table_size, 0, 0};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_add_patch_pos_embed_kernel(num_patches, H, pos_table_size);
    });
    return shader(hidden, table, pos_ids).dispatch();
}

TileCmd dispatch_tile_mask_from_pos(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<uint32_t> pos_ids,
    compute::BufferView<float> out_mask,
    uint32_t num_patches,
    uint32_t pos_table_size) {
    TileShaderKey key{make_op_tag("mask_from_pos"), num_patches, pos_table_size, 0, 0, 0};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_mask_from_pos_kernel(num_patches, pos_table_size);
    });
    return shader(pos_ids, out_mask).dispatch();
}

TileCmd dispatch_tile_scatter_soft_tokens(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> h0,
    compute::BufferView<float> soft_tokens,
    uint32_t total_tokens,
    uint32_t start_token_idx,
    uint32_t num_soft_tokens,
    uint32_t H) {
    TileShaderKey key{make_op_tag("scatter_soft_tokens"), total_tokens, start_token_idx, num_soft_tokens, H, 0};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_scatter_soft_tokens_kernel(total_tokens, start_token_idx, num_soft_tokens, H);
    });
    return shader(h0, soft_tokens).dispatch();
}

TileCmd dispatch_tile_scatter_indexed_tokens(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> h0,
    compute::BufferView<float> soft_tokens,
    compute::BufferView<uint32_t> token_indices,
    uint32_t total_tokens,
    uint32_t num_soft_tokens,
    uint32_t H) {
    TileShaderKey key{make_op_tag("scatter_indexed_tokens"), total_tokens, num_soft_tokens, H, 0, 0};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_scatter_indexed_tokens_kernel(total_tokens, num_soft_tokens, H);
    });
    return shader(h0, soft_tokens, token_indices).dispatch();
}

}// namespace luisa::embed_mm::ops_tile
