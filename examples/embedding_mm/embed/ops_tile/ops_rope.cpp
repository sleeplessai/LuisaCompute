#include "embed/ops_tile/ops_rope.h"

#include <cmath>

namespace luisa::embed_mm::ops_tile {

namespace {

[[nodiscard]] bool same_region(compute::BufferView<float> a, compute::BufferView<float> b) noexcept {
    return a.handle() == b.handle() && a.offset_bytes() == b.offset_bytes();
}

}// namespace

void build_rope_cos_sin_tables(
    uint32_t max_seq_len,
    uint32_t head_dim,
    float rope_theta,
    luisa::vector<float> &out_cos,
    luisa::vector<float> &out_sin) {
    const uint32_t half_dim = head_dim / 2u;
    out_cos.resize(static_cast<size_t>(max_seq_len) * half_dim);
    out_sin.resize(static_cast<size_t>(max_seq_len) * half_dim);

    for (uint32_t c = 0u; c < half_dim; ++c) {
        const float freq_exp = (-2.0f * static_cast<float>(c)) / static_cast<float>(head_dim);
        const float inv_freq = std::pow(rope_theta, freq_exp);
        for (uint32_t pos = 0u; pos < max_seq_len; ++pos) {
            const float theta = static_cast<float>(pos) * inv_freq;
            out_cos[static_cast<size_t>(pos) * half_dim + c] = std::cos(theta);
            out_sin[static_cast<size_t>(pos) * half_dim + c] = std::sin(theta);
        }
    }
}

void build_axial_rope_cos_sin_tables(
    uint32_t pos_table_size,
    uint32_t quarter_dim,
    float rope_theta,
    luisa::vector<float> &out_cos,
    luisa::vector<float> &out_sin) {
    out_cos.resize(static_cast<size_t>(pos_table_size) * quarter_dim);
    out_sin.resize(static_cast<size_t>(pos_table_size) * quarter_dim);

    for (uint32_t c = 0u; c < quarter_dim; ++c) {
        const float exp_val = (2.0f * static_cast<float>(c)) / static_cast<float>(quarter_dim * 2u);
        const float inv_freq = 1.0f / std::pow(rope_theta, exp_val);
        for (uint32_t pos = 0u; pos < pos_table_size; ++pos) {
            const float theta = static_cast<float>(pos) * inv_freq;
            out_cos[static_cast<size_t>(pos) * quarter_dim + c] = std::cos(theta);
            out_sin[static_cast<size_t>(pos) * quarter_dim + c] = std::sin(theta);
        }
    }
}

void build_audio_rel_pos_table(
    uint32_t pos_len,
    uint32_t hidden_size,
    luisa::vector<float> &out_pos_embed) {
    const uint32_t half_dim = hidden_size / 2u;
    out_pos_embed.resize(static_cast<size_t>(pos_len) * hidden_size);
    const float log_inc = std::log(10000.0f) / static_cast<float>(half_dim - 1u);

    for (uint32_t p = 0u; p < pos_len; ++p) {
        const float pos_id = static_cast<float>(pos_len - 1u - p);
        for (uint32_t d = 0u; d < half_dim; ++d) {
            const float inv_ts = std::exp(-static_cast<float>(d) * log_inc);
            const float scaled = pos_id * inv_ts;
            out_pos_embed[static_cast<size_t>(p) * hidden_size + d] = std::sin(scaled);
            out_pos_embed[static_cast<size_t>(p) * hidden_size + half_dim + d] = std::cos(scaled);
        }
    }
}

compute::tile::Kernel make_tile_rope_kernel(
    uint32_t B, uint32_t S, uint32_t num_heads, uint32_t head_dim, uint32_t table_seq_len) noexcept {
    using namespace compute::tile;
    const int64_t total_tokens = static_cast<int64_t>(B) * static_cast<int64_t>(S);
    const int64_t s_ext = static_cast<int64_t>(S);
    const int64_t h_ext = static_cast<int64_t>(num_heads);
    const int64_t d_ext = static_cast<int64_t>(head_dim);
    const int64_t half_d = d_ext / 2;
    const int64_t tab_s = static_cast<int64_t>(table_seq_len);

    auto def = tile_kernel("embed_mm_tile_rope", [=](
        TensorView<const float, 3> qk_in,
        TensorView<const float, 2> cos_table,
        TensorView<const float, 2> sin_table,
        TensorView<float, 3> qk_out) {
        auto one_t = axis("one_t", 1);
        auto one_h = axis("one_h", 1);
        auto n = axis("n", half_d);

        auto tok_grid = axis("tok_grid", total_tokens);
        auto h_grid = axis("h_grid", h_ext);

        for (auto &nest : parallel(shape(tok_grid, h_grid))) {
            auto tok = nest.index(tok_grid);
            auto h = nest.index(h_grid);
            auto pos = tok % s_ext;

            auto x1 = qk_in.tile(coord(tok, h, 0), shape(one_t, one_h, n)).load();
            auto x2 = qk_in.tile(coord(tok, h, half_d), shape(one_t, one_h, n)).load();
            auto c_2d = cos_table.tile(coord(pos, 0), shape(one_t, n)).load();
            auto s_2d = sin_table.tile(coord(pos, 0), shape(one_t, n)).load();
            auto c_3d = map<float>(shape(one_t, one_h, n), [&](const Nest &nst) { return c_2d.at(coord(nst.index(one_t), nst.index(n))); });
            auto s_3d = map<float>(shape(one_t, one_h, n), [&](const Nest &nst) { return s_2d.at(coord(nst.index(one_t), nst.index(n))); });

            qk_out(coord(tok, h, 0), shape(one_t, one_h, n)).store(x1 * c_3d - x2 * s_3d);
            qk_out(coord(tok, h, half_d), shape(one_t, one_h, n)).store(x2 * c_3d + x1 * s_3d);
        }
    });
    return def.capture(
        tensor_shape(total_tokens, h_ext, d_ext),
        tensor_shape(tab_s, half_d),
        tensor_shape(tab_s, half_d),
        tensor_shape(total_tokens, h_ext, d_ext));
}

compute::tile::Kernel make_tile_rope_inplace_kernel(
    uint32_t B, uint32_t S, uint32_t num_heads, uint32_t head_dim, uint32_t table_seq_len) noexcept {
    using namespace compute::tile;
    const int64_t total_tokens = static_cast<int64_t>(B) * static_cast<int64_t>(S);
    const int64_t s_ext = static_cast<int64_t>(S);
    const int64_t h_ext = static_cast<int64_t>(num_heads);
    const int64_t d_ext = static_cast<int64_t>(head_dim);
    const int64_t half_d = d_ext / 2;
    const int64_t tab_s = static_cast<int64_t>(table_seq_len);

    auto def = tile_kernel("embed_mm_tile_rope_inplace", [=](
        TensorView<float, 3> qk_inout,
        TensorView<const float, 2> cos_table,
        TensorView<const float, 2> sin_table) {
        auto one_t = axis("one_t", 1);
        auto one_h = axis("one_h", 1);
        auto n = axis("n", half_d);

        auto tok_grid = axis("tok_grid", total_tokens);
        auto h_grid = axis("h_grid", h_ext);

        for (auto &nest : parallel(shape(tok_grid, h_grid))) {
            auto tok = nest.index(tok_grid);
            auto h = nest.index(h_grid);
            auto pos = tok % s_ext;

            auto x1 = qk_inout.tile(coord(tok, h, 0), shape(one_t, one_h, n)).load();
            auto x2 = qk_inout.tile(coord(tok, h, half_d), shape(one_t, one_h, n)).load();
            auto c_2d = cos_table.tile(coord(pos, 0), shape(one_t, n)).load();
            auto s_2d = sin_table.tile(coord(pos, 0), shape(one_t, n)).load();
            auto c_3d = map<float>(shape(one_t, one_h, n), [&](const Nest &nst) { return c_2d.at(coord(nst.index(one_t), nst.index(n))); });
            auto s_3d = map<float>(shape(one_t, one_h, n), [&](const Nest &nst) { return s_2d.at(coord(nst.index(one_t), nst.index(n))); });

            qk_inout(coord(tok, h, 0), shape(one_t, one_h, n)).store(x1 * c_3d - x2 * s_3d);
            qk_inout(coord(tok, h, half_d), shape(one_t, one_h, n)).store(x2 * c_3d + x1 * s_3d);
        }
    });
    return def.capture(
        tensor_shape(total_tokens, h_ext, d_ext),
        tensor_shape(tab_s, half_d),
        tensor_shape(tab_s, half_d));
}

compute::tile::Kernel make_tile_axial_rope_kernel(
    uint32_t num_patches, uint32_t num_heads, uint32_t head_dim, uint32_t pos_table_size) noexcept {
    using namespace compute::tile;
    const int64_t p_ext = static_cast<int64_t>(num_patches);
    const int64_t h_ext = static_cast<int64_t>(num_heads);
    const int64_t d_ext = static_cast<int64_t>(head_dim);
    const int64_t q_dim = d_ext / 4;// 16 for head_dim = 64
    const int64_t tab_ext = static_cast<int64_t>(pos_table_size);
    const uint32_t tab_u32 = pos_table_size;

    auto def = tile_kernel("embed_mm_tile_axial_rope", [=](
        TensorView<const float, 3> qk_in,
        TensorView<const uint32_t, 2> pos_ids,
        TensorView<const float, 2> cos_table,
        TensorView<const float, 2> sin_table,
        TensorView<float, 3> qk_out) {
        auto one_p = axis("one_p", 1);
        auto one_h = axis("one_h", 1);
        auto n = axis("n", q_dim);

        auto p_grid = axis("p_grid", p_ext);
        auto h_grid = axis("h_grid", h_ext);

        for (auto &nest : parallel(shape(p_grid, h_grid))) {
            auto p = nest.index(p_grid);
            auto h = nest.index(h_grid);

            auto px_tile = pos_ids.tile(coord(p, 0), shape(one_p, one_h)).load();
            auto py_tile = pos_ids.tile(coord(p, 1), shape(one_p, one_h)).load();
            auto valid_1x1 = (px_tile < tab_u32) && (py_tile < tab_u32);
            auto safe_px = cast<int64_t>(ite(valid_1x1, px_tile, 0u).at(coord(0, 0)));
            auto safe_py = cast<int64_t>(ite(valid_1x1, py_tile, 0u).at(coord(0, 0)));

            auto valid_3d = map<bool>(shape(one_p, one_h, n), [&](const Nest &nst) {
                return valid_1x1.at(coord(nst.index(one_p), nst.index(one_h)));
            });

            auto cx_2d = cos_table.tile(coord(safe_px, 0), shape(one_p, n)).load();
            auto sx_2d = sin_table.tile(coord(safe_px, 0), shape(one_p, n)).load();
            auto cy_2d = cos_table.tile(coord(safe_py, 0), shape(one_p, n)).load();
            auto sy_2d = sin_table.tile(coord(safe_py, 0), shape(one_p, n)).load();

            auto cx = map<float>(shape(one_p, one_h, n), [&](const Nest &nst) { return cx_2d.at(coord(nst.index(one_p), nst.index(n))); });
            auto sx = map<float>(shape(one_p, one_h, n), [&](const Nest &nst) { return sx_2d.at(coord(nst.index(one_p), nst.index(n))); });
            auto cy = map<float>(shape(one_p, one_h, n), [&](const Nest &nst) { return cy_2d.at(coord(nst.index(one_p), nst.index(n))); });
            auto sy = map<float>(shape(one_p, one_h, n), [&](const Nest &nst) { return sy_2d.at(coord(nst.index(one_p), nst.index(n))); });

            auto v0 = qk_in.tile(coord(p, h, 0), shape(one_p, one_h, n)).load();
            auto v1 = qk_in.tile(coord(p, h, q_dim), shape(one_p, one_h, n)).load();
            auto w0 = qk_in.tile(coord(p, h, q_dim * 2), shape(one_p, one_h, n)).load();
            auto w1 = qk_in.tile(coord(p, h, q_dim * 3), shape(one_p, one_h, n)).load();

            qk_out(coord(p, h, 0), shape(one_p, one_h, n)).store(ite(valid_3d, v0 * cx - v1 * sx, 0.0f));
            qk_out(coord(p, h, q_dim), shape(one_p, one_h, n)).store(ite(valid_3d, v1 * cx + v0 * sx, 0.0f));
            qk_out(coord(p, h, q_dim * 2), shape(one_p, one_h, n)).store(ite(valid_3d, w0 * cy - w1 * sy, 0.0f));
            qk_out(coord(p, h, q_dim * 3), shape(one_p, one_h, n)).store(ite(valid_3d, w1 * cy + w0 * sy, 0.0f));
        }
    });
    return def.capture(
        tensor_shape(p_ext, h_ext, d_ext),
        tensor_shape(p_ext, 2),
        tensor_shape(tab_ext, q_dim),
        tensor_shape(tab_ext, q_dim),
        tensor_shape(p_ext, h_ext, d_ext));
}

compute::tile::Kernel make_tile_axial_rope_inplace_kernel(
    uint32_t num_patches, uint32_t num_heads, uint32_t head_dim, uint32_t pos_table_size) noexcept {
    using namespace compute::tile;
    const int64_t p_ext = static_cast<int64_t>(num_patches);
    const int64_t h_ext = static_cast<int64_t>(num_heads);
    const int64_t d_ext = static_cast<int64_t>(head_dim);
    const int64_t q_dim = d_ext / 4;
    const int64_t tab_ext = static_cast<int64_t>(pos_table_size);
    const uint32_t tab_u32 = pos_table_size;

    auto def = tile_kernel("embed_mm_tile_axial_rope_inplace", [=](
        TensorView<float, 3> qk_inout,
        TensorView<const uint32_t, 2> pos_ids,
        TensorView<const float, 2> cos_table,
        TensorView<const float, 2> sin_table) {
        auto one_p = axis("one_p", 1);
        auto one_h = axis("one_h", 1);
        auto n = axis("n", q_dim);

        auto p_grid = axis("p_grid", p_ext);
        auto h_grid = axis("h_grid", h_ext);

        for (auto &nest : parallel(shape(p_grid, h_grid))) {
            auto p = nest.index(p_grid);
            auto h = nest.index(h_grid);

            auto px_tile = pos_ids.tile(coord(p, 0), shape(one_p, one_h)).load();
            auto py_tile = pos_ids.tile(coord(p, 1), shape(one_p, one_h)).load();
            auto valid_1x1 = (px_tile < tab_u32) && (py_tile < tab_u32);
            auto safe_px = cast<int64_t>(ite(valid_1x1, px_tile, 0u).at(coord(0, 0)));
            auto safe_py = cast<int64_t>(ite(valid_1x1, py_tile, 0u).at(coord(0, 0)));

            auto valid_3d = map<bool>(shape(one_p, one_h, n), [&](const Nest &nst) {
                return valid_1x1.at(coord(nst.index(one_p), nst.index(one_h)));
            });

            auto cx_2d = cos_table.tile(coord(safe_px, 0), shape(one_p, n)).load();
            auto sx_2d = sin_table.tile(coord(safe_px, 0), shape(one_p, n)).load();
            auto cy_2d = cos_table.tile(coord(safe_py, 0), shape(one_p, n)).load();
            auto sy_2d = sin_table.tile(coord(safe_py, 0), shape(one_p, n)).load();

            auto cx = map<float>(shape(one_p, one_h, n), [&](const Nest &nst) { return cx_2d.at(coord(nst.index(one_p), nst.index(n))); });
            auto sx = map<float>(shape(one_p, one_h, n), [&](const Nest &nst) { return sx_2d.at(coord(nst.index(one_p), nst.index(n))); });
            auto cy = map<float>(shape(one_p, one_h, n), [&](const Nest &nst) { return cy_2d.at(coord(nst.index(one_p), nst.index(n))); });
            auto sy = map<float>(shape(one_p, one_h, n), [&](const Nest &nst) { return sy_2d.at(coord(nst.index(one_p), nst.index(n))); });

            auto v0 = qk_inout.tile(coord(p, h, 0), shape(one_p, one_h, n)).load();
            auto v1 = qk_inout.tile(coord(p, h, q_dim), shape(one_p, one_h, n)).load();
            auto w0 = qk_inout.tile(coord(p, h, q_dim * 2), shape(one_p, one_h, n)).load();
            auto w1 = qk_inout.tile(coord(p, h, q_dim * 3), shape(one_p, one_h, n)).load();

            qk_inout(coord(p, h, 0), shape(one_p, one_h, n)).store(ite(valid_3d, v0 * cx - v1 * sx, 0.0f));
            qk_inout(coord(p, h, q_dim), shape(one_p, one_h, n)).store(ite(valid_3d, v1 * cx + v0 * sx, 0.0f));
            qk_inout(coord(p, h, q_dim * 2), shape(one_p, one_h, n)).store(ite(valid_3d, w0 * cy - w1 * sy, 0.0f));
            qk_inout(coord(p, h, q_dim * 3), shape(one_p, one_h, n)).store(ite(valid_3d, w1 * cy + w0 * sy, 0.0f));
        }
    });
    return def.capture(
        tensor_shape(p_ext, h_ext, d_ext),
        tensor_shape(p_ext, 2),
        tensor_shape(tab_ext, q_dim),
        tensor_shape(tab_ext, q_dim));
}

TileCmd dispatch_tile_rope(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> qk_in,
    compute::BufferView<float> qk_out,
    compute::BufferView<float> cos_table,
    compute::BufferView<float> sin_table,
    uint32_t B, uint32_t S, uint32_t num_heads, uint32_t head_dim, uint32_t table_seq_len, uint64_t table_tag) {
    if (same_region(qk_in, qk_out)) {
        TileShaderKey key{make_op_tag("rope_inplace"), B, S, num_heads, head_dim, (static_cast<uint64_t>(table_seq_len) << 32u) | table_tag};
        const auto &shader = cache.get_or_compile(device, key, [&]() {
            return make_tile_rope_inplace_kernel(B, S, num_heads, head_dim, table_seq_len);
        });
        return shader(qk_out, cos_table, sin_table).dispatch();
    }
    TileShaderKey key{make_op_tag("rope"), B, S, num_heads, head_dim, (static_cast<uint64_t>(table_seq_len) << 32u) | table_tag};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_rope_kernel(B, S, num_heads, head_dim, table_seq_len);
    });
    return shader(qk_in, cos_table, sin_table, qk_out).dispatch();
}

TileCmd dispatch_tile_axial_rope(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> qk_in,
    compute::BufferView<float> qk_out,
    compute::BufferView<uint32_t> pos_ids,
    compute::BufferView<float> cos_table,
    compute::BufferView<float> sin_table,
    uint32_t num_patches, uint32_t num_heads, uint32_t head_dim, uint32_t pos_table_size) {
    if (same_region(qk_in, qk_out)) {
        TileShaderKey key{make_op_tag("axial_rope_inplace"), num_patches, num_heads, head_dim, pos_table_size, 0};
        const auto &shader = cache.get_or_compile(device, key, [&]() {
            return make_tile_axial_rope_inplace_kernel(num_patches, num_heads, head_dim, pos_table_size);
        });
        return shader(qk_out, pos_ids, cos_table, sin_table).dispatch();
    }
    TileShaderKey key{make_op_tag("axial_rope"), num_patches, num_heads, head_dim, pos_table_size, 0};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_axial_rope_kernel(num_patches, num_heads, head_dim, pos_table_size);
    });
    return shader(qk_in, pos_ids, cos_table, sin_table, qk_out).dispatch();
}

}// namespace luisa::embed_mm::ops_tile
