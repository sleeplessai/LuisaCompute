#include "embed/ops_tile/ops_attention.h"

#include <bit>

namespace luisa::embed_mm::ops_tile {

compute::tile::Kernel make_tile_gqa_attention_kernel(
    uint32_t B,
    uint32_t S,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    float scale,
    uint32_t is_sliding,
    uint32_t sliding_window,
    float neg_large) noexcept {
    using namespace compute::tile;
    const int64_t b_ext = static_cast<int64_t>(B);
    const int64_t s_ext = static_cast<int64_t>(S);
    const int64_t hq_ext = static_cast<int64_t>(num_q_heads);
    const int64_t hkv_ext = static_cast<int64_t>(num_kv_heads);
    const int64_t group_size = hq_ext / hkv_ext;
    const int64_t d_ext = static_cast<int64_t>(head_dim);
    const int64_t win_ext = static_cast<int64_t>(sliding_window);
    const bool sliding = (is_sliding != 0u);

    const int64_t bq = (d_ext >= 128) ? 1 : (d_ext >= 64 ? 4 : 8);
    const int64_t bkv = 16;
    const int64_t q_blocks = luisa::ceil_div(s_ext, bq);
    const int64_t kv_blocks = luisa::ceil_div(s_ext, bkv);

    auto def = tile_kernel("embed_mm_tile_gqa_attention", [=](
        TensorView<const float, 4> Q,
        TensorView<const float, 4> K,
        TensorView<const float, 4> V,
        TensorView<const float, 2> mask,
        TensorView<float, 4> O) {
        auto b_ax = axis("b_ax", 1);
        auto q_seq = axis("q_seq", bq);
        auto h_ax = axis("h_ax", 1);
        auto kv_seq = axis("kv_seq", bkv);
        auto d_ax = axis("d_ax", d_ext);

        auto b_grid = axis("b_grid", b_ext);
        auto hq_grid = axis("hq_grid", hq_ext);
        auto q_grid = axis("q_grid", q_blocks);

        for (auto &nest : parallel(shape(b_grid, hq_grid, q_grid))) {
            auto b0 = nest.index(b_grid);
            auto h0 = nest.index(hq_grid);
            auto q0 = nest.index(q_grid) * bq;
            auto kh = h0 / group_size;

            auto q_fp32 = Q.tile(coord(b0, q0, h0, 0), shape(b_ax, q_seq, h_ax, d_ax)).load();
            auto q_bf16 = cast<luisa::bfloat16>(q_fp32);

            auto row_max = full<float>(shape(b_ax, q_seq, h_ax), neg_large);
            auto row_sum = zeros<float>(shape(b_ax, q_seq, h_ax));
            auto acc = zeros<float>(shape(b_ax, q_seq, h_ax, d_ax));

            for (auto &step : nest.pipeline(shape(kv_blocks), {.window = 1u, .interval = 1u})) {
                auto k0 = step.index() * bkv;
                step.stage("load");
                auto k_fp32 = K.tile(coord(b0, k0, kh, 0), shape(b_ax, kv_seq, h_ax, d_ax)).load();
                auto v_fp32 = V.tile(coord(b0, k0, kh, 0), shape(b_ax, kv_seq, h_ax, d_ax)).load();
                auto m_2d = mask.tile(coord(b0, k0), shape(b_ax, kv_seq)).load();

                step.stage("score");
                auto dot = mma(
                    q_bf16,
                    cast<luisa::bfloat16>(k_fp32),
                    zeros<float>(shape(b_ax, q_seq, h_ax, kv_seq)),
                    {.allow_reassociation = true});
                auto score = dot * scale;

                auto q_pos = map<int64_t>(shape(b_ax, q_seq, h_ax, kv_seq), [&](const Nest &nst) { return nst.index(q_seq) + q0; });
                auto k_pos = map<int64_t>(shape(b_ax, q_seq, h_ax, kv_seq), [&](const Nest &nst) { return nst.index(kv_seq) + k0; });
                auto m_4d = map<float>(shape(b_ax, q_seq, h_ax, kv_seq), [&](const Nest &nst) {
                    return m_2d.at(coord(nst.index(b_ax), nst.index(kv_seq)));
                });

                auto valid = (q_pos < s_ext) && (k_pos < s_ext) && (m_4d != 0.0f);
                if (sliding) {
                    valid = valid && ((q_pos - k_pos) <= win_ext) && ((k_pos - q_pos) <= win_ext);
                }

                auto masked = ite(valid, score, neg_large);
                auto next_max = max(row_max, reduce(masked, kv_seq, maximum));
                auto alpha = exp(row_max - next_max);
                auto next_max_4d = broadcast_to(next_max, shape(b_ax, q_seq, h_ax, kv_seq));
                auto prob = ite(valid, exp(masked - next_max_4d), 0.0f);

                step.stage("update");
                row_sum = row_sum * alpha + reduce(prob, kv_seq, add);
                auto alpha_d = broadcast_to(alpha, shape(b_ax, q_seq, h_ax, d_ax));
                acc = mma(
                    cast<luisa::bfloat16>(prob),
                    cast<luisa::bfloat16>(v_fp32),
                    acc * alpha_d,
                    {.allow_reassociation = true});
                row_max = next_max;
            }
            auto norm_d = broadcast_to(max(row_sum, 1e-12f), shape(b_ax, q_seq, h_ax, d_ax));
            O(coord(b0, q0, h0, 0), shape(b_ax, q_seq, h_ax, d_ax)).store(acc / norm_d);
        }
    });
    return def.capture(
        tensor_shape(b_ext, s_ext, hq_ext, d_ext),
        tensor_shape(b_ext, s_ext, hkv_ext, d_ext),
        tensor_shape(b_ext, s_ext, hkv_ext, d_ext),
        tensor_shape(b_ext, s_ext),
        tensor_shape(b_ext, s_ext, hq_ext, d_ext));
}

compute::tile::Kernel make_tile_audio_scale_qk_kernel(
    uint32_t T,
    uint32_t num_heads,
    uint32_t head_dim,
    float q_scale,
    float k_scale) noexcept {
    using namespace compute::tile;
    const int64_t t_ext = static_cast<int64_t>(T);
    const int64_t h_ext = static_cast<int64_t>(num_heads);
    const int64_t d_ext = static_cast<int64_t>(head_dim);

    auto def = tile_kernel("embed_mm_tile_audio_scale_qk", [=](
        TensorView<float, 3> q,
        TensorView<float, 3> k,
        TensorView<const float, 1> per_dim_scale) {
        auto one_t = axis("one_t", 1);
        auto one_h = axis("one_h", 1);
        auto d_ax = axis("d_ax", d_ext);

        auto t_grid = axis("t_grid", t_ext);
        auto h_grid = axis("h_grid", h_ext);

        for (auto &nest : parallel(shape(t_grid, h_grid))) {
            auto t = nest.index(t_grid);
            auto h = nest.index(h_grid);
            auto pds_1d = per_dim_scale.tile(coord(0), shape(d_ax)).load();
            auto sp_1d = ite(pds_1d > 20.0f, pds_1d, log(1.0f + exp(pds_1d)));
            auto sp_3d = map<float>(shape(one_t, one_h, d_ax), [&](const Nest &n) {
                return sp_1d.at(coord(n.index(d_ax)));
            });
            auto q_val = q.tile(coord(t, h, 0), shape(one_t, one_h, d_ax)).load();
            auto k_val = k.tile(coord(t, h, 0), shape(one_t, one_h, d_ax)).load();
            q(coord(t, h, 0), shape(one_t, one_h, d_ax)).store(q_val * (sp_3d * q_scale));
            k(coord(t, h, 0), shape(one_t, one_h, d_ax)).store(k_val * k_scale);
        }
    });
    return def.capture(
        tensor_shape(t_ext, h_ext, d_ext),
        tensor_shape(t_ext, h_ext, d_ext),
        tensor_shape(d_ext));
}

compute::tile::Kernel make_tile_audio_chunked_attention_kernel(
    uint32_t T,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t chunk_size,
    float softcap) noexcept {
    using namespace compute::tile;
    const int64_t t_ext = static_cast<int64_t>(T);
    const int64_t h_ext = static_cast<int64_t>(num_heads);
    const int64_t d_ext = static_cast<int64_t>(head_dim);
    const int64_t c_size = static_cast<int64_t>(chunk_size);
    const float inv_softcap = 1.0f / softcap;

    auto def = tile_kernel("embed_mm_tile_audio_chunked_attn", [=](
        TensorView<const float, 3> q,
        TensorView<const float, 3> k,
        TensorView<const float, 3> v,
        TensorView<const float, 3> rel_k,
        TensorView<float, 3> out_attn) {
        auto one_t = axis("one_t", 1);
        auto one_h = axis("one_h", 1);
        auto d_ax = axis("d_ax", d_ext);

        auto t_grid = axis("t_grid", t_ext);
        auto h_grid = axis("h_grid", h_ext);

        for (auto &nest : parallel(shape(t_grid, h_grid))) {
            auto t_q = nest.index(t_grid);
            auto h = nest.index(h_grid);
            auto b = t_q / c_size;
            auto i = t_q % c_size;

            auto q_vec = q.tile(coord(t_q, h, 0), shape(one_t, one_h, d_ax)).load();
            auto max_logit = full<float>(shape(one_t, one_h), -1e9f);
            auto sum_exp = zeros<float>(shape(one_t, one_h));
            auto acc = zeros<float>(shape(one_t, one_h, d_ax));

            for (auto &step : nest.serial(shape(24))) {
                auto j = step.index();
                auto t_k = b * c_size - 12 + j;
                auto p = j - i;

                auto k_vec = k.tile(coord(t_k, h, 0), shape(one_t, one_h, d_ax)).load();
                auto rk_vec = rel_k.tile(coord(p, h, 0), shape(one_t, one_h, d_ax)).load();
                auto v_vec = v.tile(coord(t_k, h, 0), shape(one_t, one_h, d_ax)).load();

                auto ac = reduce(q_vec * k_vec, d_ax, add);
                auto bd = reduce(q_vec * rk_vec, d_ax, add);
                auto raw_logit = tanh((ac + bd) * inv_softcap) * softcap;

                auto zero_2d = zeros<int64_t>(shape(one_t, one_h));
                auto tk_2d = zero_2d + t_k;
                auto j_2d = zero_2d + j;
                auto i_2d = zero_2d + i;
                auto is_valid = (tk_2d >= 0) && (tk_2d < t_ext) && (j_2d <= (12 + i_2d)) && (j_2d > i_2d);

                auto logit = ite(is_valid, raw_logit, -1e9f);
                auto next_max = max(max_logit, logit);
                auto alpha = exp(max_logit - next_max);
                auto w = ite(is_valid, exp(logit - next_max), 0.0f);

                sum_exp = sum_exp * alpha + w;
                acc = acc * alpha + v_vec * w;
                max_logit = next_max;
            }
            out_attn(coord(t_q, h, 0), shape(one_t, one_h, d_ax)).store(acc / max(sum_exp, 1e-12f));
        }
    });
    return def.capture(
        tensor_shape(t_ext, h_ext, d_ext),
        tensor_shape(t_ext, h_ext, d_ext),
        tensor_shape(t_ext, h_ext, d_ext),
        tensor_shape(13, h_ext, d_ext),
        tensor_shape(t_ext, h_ext, d_ext));
}

TileCmd dispatch_tile_gqa_attention(
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
    float neg_large) {
    const uint64_t scale_bits = std::bit_cast<uint32_t>(scale);
    const uint64_t heads_packed = (static_cast<uint64_t>(num_q_heads) << 32u) | num_kv_heads;
    const uint64_t slide_packed = (static_cast<uint64_t>(is_sliding) << 32u) | sliding_window;
    TileShaderKey key{make_op_tag("gqa_attn"), B, S, heads_packed, head_dim, (scale_bits << 32u) ^ slide_packed};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_gqa_attention_kernel(
            B, S, num_q_heads, num_kv_heads, head_dim, scale, is_sliding, sliding_window, neg_large);
    });
    return shader(q, k, v, mask, out).dispatch();
}

TileCmd dispatch_tile_audio_scale_qk(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> q,
    compute::BufferView<float> k,
    compute::BufferView<float> per_dim_scale,
    uint32_t T,
    uint32_t num_heads,
    uint32_t head_dim,
    float q_scale,
    float k_scale) {
    const uint64_t qs_bits = std::bit_cast<uint32_t>(q_scale);
    const uint64_t ks_bits = std::bit_cast<uint32_t>(k_scale);
    TileShaderKey key{make_op_tag("audio_scale_qk"), T, num_heads, head_dim, qs_bits, ks_bits};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_audio_scale_qk_kernel(T, num_heads, head_dim, q_scale, k_scale);
    });
    return shader(q, k, per_dim_scale).dispatch();
}

TileCmd dispatch_tile_audio_chunked_attention(
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
    float softcap) {
    const uint64_t sc_bits = std::bit_cast<uint32_t>(softcap);
    TileShaderKey key{make_op_tag("audio_chunked_attn"), T, num_heads, head_dim, chunk_size, sc_bits};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_audio_chunked_attention_kernel(T, num_heads, head_dim, chunk_size, softcap);
    });
    return shader(q, k, v, rel_k, out_attn).dispatch();
}

}// namespace luisa::embed_mm::ops_tile
