#include "embed/ops_tile/ops_conv.h"

#include <bit>

namespace luisa::embed_mm::ops_tile {

compute::tile::Kernel make_tile_audio_subsample_conv0_kernel(
    uint32_t H_in, uint32_t W_in, uint32_t H_out, uint32_t W_out, float eps) noexcept {
    using namespace compute::tile;
    const int64_t hin = static_cast<int64_t>(H_in);
    const int64_t win = static_cast<int64_t>(W_in);
    const int64_t hout = static_cast<int64_t>(H_out);
    const int64_t wout = static_cast<int64_t>(W_out);

    auto def = tile_kernel("embed_mm_tile_audio_conv0", [=](
        TensorView<const float, 2> in_features,
        TensorView<const float, 3> weight,
        TensorView<const float, 1> norm_weight,
        TensorView<float, 3> out_h0) {
        auto c_ax = axis("c_ax", 128);
        auto one_h = axis("one_h", 1);
        auto one_w = axis("one_w", 1);

        auto h_grid = axis("h_grid", hout);
        auto w_grid = axis("w_grid", wout);

        for (auto &nest : parallel(shape(h_grid, w_grid))) {
            auto h = nest.index(h_grid);
            auto w = nest.index(w_grid);
            auto in_h_base = h * 2 - 1;
            auto in_w_base = w * 2 - 1;

            auto acc = zeros<float>(shape(c_ax, one_h, one_w));
            for (int64_t kh = 0; kh < 3; ++kh) {
                for (int64_t kw = 0; kw < 3; ++kw) {
                    auto ih = in_h_base + kh;
                    auto iw = in_w_base + kw;
                    auto x_tile = in_features.tile(coord(ih, iw), shape(one_h, one_w)).load();
                    auto w_tile = weight.tile(coord(0, kh, kw), shape(c_ax, one_h, one_w)).load();
                    auto x_val = x_tile.at(coord(0, 0));
                    acc = acc + w_tile * x_val;
                }
            }

            auto v_1d = map<float>(shape(c_ax), [&](const Nest &nst) {
                return acc.at(coord(nst.index(c_ax), 0, 0));
            });
            auto mean = broadcast_to(reduce(v_1d, c_ax, add) * (1.0f / 128.0f), shape(c_ax));
            auto diff = v_1d - mean;
            auto var = broadcast_to(reduce(diff * diff, c_ax, add) * (1.0f / 128.0f), shape(c_ax));
            auto nw = norm_weight.tile(coord(0), shape(c_ax)).load();
            auto normed = (diff / sqrt(var + eps)) * nw;
            auto activated = max(normed, 0.0f);
            auto out_3d = map<float>(shape(c_ax, one_h, one_w), [&](const Nest &nst) {
                return activated.at(coord(nst.index(c_ax)));
            });
            out_h0(coord(0, h, w), shape(c_ax, one_h, one_w)).store(out_3d);
        }
    });
    return def.capture(
        tensor_shape(hin, win),
        tensor_shape(128, 3, 3),
        tensor_shape(128),
        tensor_shape(128, hout, wout));
}

compute::tile::Kernel make_tile_audio_subsample_conv1_kernel(
    uint32_t H_in, uint32_t W_in, uint32_t H_out, uint32_t W_out, float eps) noexcept {
    using namespace compute::tile;
    const int64_t hin = static_cast<int64_t>(H_in);
    const int64_t win = static_cast<int64_t>(W_in);
    const int64_t hout = static_cast<int64_t>(H_out);
    const int64_t wout = static_cast<int64_t>(W_out);

    auto def = tile_kernel("embed_mm_tile_audio_conv1", [=](
        TensorView<const float, 3> in_h0,
        TensorView<const float, 4> weight,
        TensorView<const float, 1> norm_weight,
        TensorView<float, 3> out_flat) {
        auto oc_ax = axis("oc_ax", 32);
        auto ic_ax = axis("ic_ax", 32);
        auto one_h = axis("one_h", 1);
        auto one_w = axis("one_w", 1);

        auto h_grid = axis("h_grid", hout);
        auto w_grid = axis("w_grid", wout);

        for (auto &nest : parallel(shape(h_grid, w_grid))) {
            auto h = nest.index(h_grid);
            auto w = nest.index(w_grid);
            auto in_h_base = h * 2 - 1;
            auto in_w_base = w * 2 - 1;

            auto acc = zeros<float>(shape(oc_ax));
            for (auto &ic_step : nest.serial(shape(4))) {
                auto ic_base = ic_step.index() * 32;
                for (int64_t kh = 0; kh < 3; ++kh) {
                    for (int64_t kw = 0; kw < 3; ++kw) {
                        auto ih = in_h_base + kh;
                        auto iw = in_w_base + kw;
                        auto x_3d = in_h0.tile(coord(ic_base, ih, iw), shape(ic_ax, one_h, one_w)).load();
                        auto w_4d = weight.tile(coord(0, ic_base, kh, kw), shape(oc_ax, ic_ax, one_h, one_w)).load();
                        auto prod_2d = map<float>(shape(oc_ax, ic_ax), [&](const Nest &nst) {
                            return w_4d.at(coord(nst.index(oc_ax), nst.index(ic_ax), 0, 0)) * x_3d.at(coord(nst.index(ic_ax), 0, 0));
                        });
                        acc = acc + reduce(prod_2d, ic_ax, add);
                    }
                }
            }

            auto mean = broadcast_to(reduce(acc, oc_ax, add) * (1.0f / 32.0f), shape(oc_ax));
            auto diff = acc - mean;
            auto var = broadcast_to(reduce(diff * diff, oc_ax, add) * (1.0f / 32.0f), shape(oc_ax));
            auto nw = norm_weight.tile(coord(0), shape(oc_ax)).load();
            auto normed = (diff / sqrt(var + eps)) * nw;
            auto activated = max(normed, 0.0f);
            auto out_3d = map<float>(shape(one_h, one_w, oc_ax), [&](const Nest &nst) {
                return activated.at(coord(nst.index(oc_ax)));
            });
            out_flat(coord(h, w, 0), shape(one_h, one_w, oc_ax)).store(out_3d);
        }
    });
    return def.capture(
        tensor_shape(128, hin, win),
        tensor_shape(32, 128, 3, 3),
        tensor_shape(32),
        tensor_shape(hout, wout, 32));
}

compute::tile::Kernel make_tile_audio_depthwise_conv1d_kernel(
    uint32_t T, uint32_t channels) noexcept {
    using namespace compute::tile;
    const int64_t t_ext = static_cast<int64_t>(T);
    const int64_t c_ext = static_cast<int64_t>(channels);
    const int64_t bc = 128;
    const int64_t c_tiles = luisa::ceil_div(c_ext, bc);

    auto def = tile_kernel("embed_mm_tile_audio_dw_conv1d", [=](
        TensorView<const float, 2> in_x,
        TensorView<const float, 2> weight,
        TensorView<float, 2> out_y) {
        auto one_t = axis("one_t", 1);
        auto c_ax = axis("c_ax", bc);
        auto one_k = axis("one_k", 1);

        auto t_grid = axis("t_grid", t_ext);
        auto c_grid = axis("c_grid", c_tiles);

        for (auto &nest : parallel(shape(t_grid, c_grid))) {
            auto t = nest.index(t_grid);
            auto col = nest.index(c_grid) * bc;
            auto acc = zeros<float>(shape(one_t, c_ax));
            for (int64_t k = 0; k < 5; ++k) {
                auto src_t = t - 4 + k;
                auto x_val = in_x.tile(coord(src_t, col), shape(one_t, c_ax)).load();
                auto w_col = weight.tile(coord(col, k), shape(c_ax, one_k)).load();
                auto w_row = map<float>(shape(one_t, c_ax), [&](const Nest &n) {
                    return w_col.at(coord(n.index(c_ax), 0));
                });
                acc = acc + x_val * w_row;
            }
            out_y(coord(t, col), shape(one_t, c_ax)).store(acc);
        }
    });
    return def.capture(
        tensor_shape(t_ext, c_ext),
        tensor_shape(c_ext, 5),
        tensor_shape(t_ext, c_ext));
}

TileCmd dispatch_tile_audio_subsample_conv0(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> in_features,
    compute::BufferView<float> weight,
    compute::BufferView<float> norm_weight,
    compute::BufferView<float> out_h0,
    uint32_t H_in, uint32_t W_in, uint32_t H_out, uint32_t W_out, float eps) {
    const uint64_t eps_bits = std::bit_cast<uint32_t>(eps);
    TileShaderKey key{make_op_tag("audio_conv0"), H_in, W_in, H_out, W_out, eps_bits};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_audio_subsample_conv0_kernel(H_in, W_in, H_out, W_out, eps);
    });
    return shader(in_features, weight, norm_weight, out_h0).dispatch();
}

TileCmd dispatch_tile_audio_subsample_conv1(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> in_h0,
    compute::BufferView<float> weight,
    compute::BufferView<float> norm_weight,
    compute::BufferView<float> out_flat,
    uint32_t H_in, uint32_t W_in, uint32_t H_out, uint32_t W_out, float eps) {
    const uint64_t eps_bits = std::bit_cast<uint32_t>(eps);
    TileShaderKey key{make_op_tag("audio_conv1"), H_in, W_in, H_out, W_out, eps_bits};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_audio_subsample_conv1_kernel(H_in, W_in, H_out, W_out, eps);
    });
    return shader(in_h0, weight, norm_weight, out_flat).dispatch();
}

TileCmd dispatch_tile_audio_depthwise_conv1d(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> in_x,
    compute::BufferView<float> weight,
    compute::BufferView<float> out_y,
    uint32_t T, uint32_t channels) {
    TileShaderKey key{make_op_tag("audio_dw_conv1d"), T, channels, 0, 0, 0};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_audio_depthwise_conv1d_kernel(T, channels);
    });
    return shader(in_x, weight, out_y).dispatch();
}

}// namespace luisa::embed_mm::ops_tile
