#include "embed/ops_tile/ops_mlp.h"

#include <bit>

namespace luisa::embed_mm::ops_tile {

namespace {

constexpr int64_t kVecTile = 256;
constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kGeluCoeff = 0.044715f;

[[nodiscard]] bool same_region(compute::BufferView<float> a, compute::BufferView<float> b) noexcept {
    return a.handle() == b.handle() && a.offset_bytes() == b.offset_bytes();
}

}// namespace

compute::tile::Kernel make_tile_geglu_kernel(uint32_t N) noexcept {
    using namespace compute::tile;
    const int64_t n_ext = static_cast<int64_t>(N);
    const int64_t blocks = luisa::ceil_div(n_ext, kVecTile);

    auto def = tile_kernel("embed_mm_tile_geglu", [=](
        TensorView<const float, 1> gate,
        TensorView<const float, 1> up,
        TensorView<float, 1> out) {
        auto v = axis("v", kVecTile);
        for (auto &nest : parallel(shape(blocks))) {
            auto base = nest.index() * kVecTile;
            auto x = gate.tile(coord(base), shape(v)).load();
            auto u = up.tile(coord(base), shape(v)).load();
            auto inner = kSqrt2OverPi * (x + kGeluCoeff * x * x * x);
            auto act = 0.5f * x * (1.0f + tanh(inner));
            out(coord(base), shape(v)).store(act * u);
        }
    });
    return def.capture(tensor_shape(n_ext), tensor_shape(n_ext), tensor_shape(n_ext));
}

compute::tile::Kernel make_tile_residual_add_kernel(uint32_t N) noexcept {
    using namespace compute::tile;
    const int64_t n_ext = static_cast<int64_t>(N);
    const int64_t blocks = luisa::ceil_div(n_ext, kVecTile);

    auto def = tile_kernel("embed_mm_tile_residual_add", [=](
        TensorView<const float, 1> x,
        TensorView<const float, 1> residual,
        TensorView<float, 1> out) {
        auto v = axis("v", kVecTile);
        for (auto &nest : parallel(shape(blocks))) {
            auto base = nest.index() * kVecTile;
            auto xv = x.tile(coord(base), shape(v)).load();
            auto rv = residual.tile(coord(base), shape(v)).load();
            out(coord(base), shape(v)).store(xv + rv);
        }
    });
    return def.capture(tensor_shape(n_ext), tensor_shape(n_ext), tensor_shape(n_ext));
}

compute::tile::Kernel make_tile_residual_add_inplace_kernel(uint32_t N) noexcept {
    using namespace compute::tile;
    const int64_t n_ext = static_cast<int64_t>(N);
    const int64_t blocks = luisa::ceil_div(n_ext, kVecTile);

    auto def = tile_kernel("embed_mm_tile_residual_add_inplace", [=](
        TensorView<float, 1> inout,
        TensorView<const float, 1> other) {
        auto v = axis("v", kVecTile);
        for (auto &nest : parallel(shape(blocks))) {
            auto base = nest.index() * kVecTile;
            auto xv = inout.tile(coord(base), shape(v)).load();
            auto rv = other.tile(coord(base), shape(v)).load();
            inout(coord(base), shape(v)).store(xv + rv);
        }
    });
    return def.capture(tensor_shape(n_ext), tensor_shape(n_ext));
}

compute::tile::Kernel make_tile_ple_gate_kernel(
    uint32_t layer_idx, uint32_t total_tokens, uint32_t num_layers, uint32_t H) noexcept {
    using namespace compute::tile;
    const int64_t t_ext = static_cast<int64_t>(total_tokens);
    const int64_t l_ext = static_cast<int64_t>(num_layers);
    const int64_t h_ext = static_cast<int64_t>(H);
    const int64_t h_pow2 = next_pow2(H);
    const int64_t l_idx = static_cast<int64_t>(layer_idx);

    auto def = tile_kernel("embed_mm_tile_ple_gate", [=](
        TensorView<const float, 2> gate,
        TensorView<const float, 2> ple_all,
        TensorView<float, 2> out) {
        auto m = axis("m", 1);
        auto n = axis("n", h_pow2);
        for (auto &nest : parallel(shape(t_ext))) {
            auto tok = nest.index();
            auto g = gate.tile(coord(tok, 0), shape(m, n)).load();
            auto ple = ple_all.tile(coord(tok, l_idx * h_ext), shape(m, n)).load();
            auto inner = kSqrt2OverPi * (g + kGeluCoeff * g * g * g);
            auto act = 0.5f * g * (1.0f + tanh(inner));
            out(coord(tok, 0), shape(m, n)).store(act * ple);
        }
    });
    return def.capture(
        tensor_shape(t_ext, h_ext),
        tensor_shape(t_ext, l_ext * h_ext),
        tensor_shape(t_ext, h_ext));
}

compute::tile::Kernel make_tile_scale_add_kernel(uint32_t N, float scale) noexcept {
    using namespace compute::tile;
    const int64_t n_ext = static_cast<int64_t>(N);
    const int64_t blocks = luisa::ceil_div(n_ext, kVecTile);

    auto def = tile_kernel("embed_mm_tile_scale_add", [=](
        TensorView<const float, 1> x,
        TensorView<const float, 1> residual,
        TensorView<float, 1> out) {
        auto v = axis("v", kVecTile);
        for (auto &nest : parallel(shape(blocks))) {
            auto base = nest.index() * kVecTile;
            auto xv = x.tile(coord(base), shape(v)).load();
            auto rv = residual.tile(coord(base), shape(v)).load();
            out(coord(base), shape(v)).store((xv + rv) * scale);
        }
    });
    return def.capture(tensor_shape(n_ext), tensor_shape(n_ext), tensor_shape(n_ext));
}

compute::tile::Kernel make_tile_scale_add_inplace_kernel(uint32_t N, float scale) noexcept {
    using namespace compute::tile;
    const int64_t n_ext = static_cast<int64_t>(N);
    const int64_t blocks = luisa::ceil_div(n_ext, kVecTile);

    auto def = tile_kernel("embed_mm_tile_scale_add_inplace", [=](
        TensorView<float, 1> inout,
        TensorView<const float, 1> other) {
        auto v = axis("v", kVecTile);
        for (auto &nest : parallel(shape(blocks))) {
            auto base = nest.index() * kVecTile;
            auto xv = inout.tile(coord(base), shape(v)).load();
            auto rv = other.tile(coord(base), shape(v)).load();
            inout(coord(base), shape(v)).store((xv + rv) * scale);
        }
    });
    return def.capture(tensor_shape(n_ext), tensor_shape(n_ext));
}

compute::tile::Kernel make_tile_audio_clamp_kernel(uint32_t N, float min_v, float max_v) noexcept {
    using namespace compute::tile;
    const int64_t n_ext = static_cast<int64_t>(N);
    const int64_t blocks = luisa::ceil_div(n_ext, kVecTile);

    auto def = tile_kernel("embed_mm_tile_audio_clamp", [=](
        TensorView<const float, 1> x,
        TensorView<float, 1> y) {
        auto v = axis("v", kVecTile);
        for (auto &nest : parallel(shape(blocks))) {
            auto base = nest.index() * kVecTile;
            auto xv = x.tile(coord(base), shape(v)).load();
            y(coord(base), shape(v)).store(min(max(xv, min_v), max_v));
        }
    });
    return def.capture(tensor_shape(n_ext), tensor_shape(n_ext));
}

compute::tile::Kernel make_tile_audio_clamp_inplace_kernel(uint32_t N, float min_v, float max_v) noexcept {
    using namespace compute::tile;
    const int64_t n_ext = static_cast<int64_t>(N);
    const int64_t blocks = luisa::ceil_div(n_ext, kVecTile);

    auto def = tile_kernel("embed_mm_tile_audio_clamp_inplace", [=](
        TensorView<float, 1> x) {
        auto v = axis("v", kVecTile);
        for (auto &nest : parallel(shape(blocks))) {
            auto base = nest.index() * kVecTile;
            auto xv = x.tile(coord(base), shape(v)).load();
            x(coord(base), shape(v)).store(min(max(xv, min_v), max_v));
        }
    });
    return def.capture(tensor_shape(n_ext));
}

compute::tile::Kernel make_tile_audio_silu_kernel(uint32_t N) noexcept {
    using namespace compute::tile;
    const int64_t n_ext = static_cast<int64_t>(N);
    const int64_t blocks = luisa::ceil_div(n_ext, kVecTile);

    auto def = tile_kernel("embed_mm_tile_audio_silu", [=](
        TensorView<const float, 1> x,
        TensorView<float, 1> y) {
        auto v = axis("v", kVecTile);
        for (auto &nest : parallel(shape(blocks))) {
            auto base = nest.index() * kVecTile;
            auto xv = x.tile(coord(base), shape(v)).load();
            y(coord(base), shape(v)).store(xv / (1.0f + exp(-xv)));
        }
    });
    return def.capture(tensor_shape(n_ext), tensor_shape(n_ext));
}

compute::tile::Kernel make_tile_audio_silu_inplace_kernel(uint32_t N) noexcept {
    using namespace compute::tile;
    const int64_t n_ext = static_cast<int64_t>(N);
    const int64_t blocks = luisa::ceil_div(n_ext, kVecTile);

    auto def = tile_kernel("embed_mm_tile_audio_silu_inplace", [=](
        TensorView<float, 1> x) {
        auto v = axis("v", kVecTile);
        for (auto &nest : parallel(shape(blocks))) {
            auto base = nest.index() * kVecTile;
            auto xv = x.tile(coord(base), shape(v)).load();
            x(coord(base), shape(v)).store(xv / (1.0f + exp(-xv)));
        }
    });
    return def.capture(tensor_shape(n_ext));
}

compute::tile::Kernel make_tile_audio_glu_kernel(uint32_t T, uint32_t H) noexcept {
    using namespace compute::tile;
    const int64_t t_ext = static_cast<int64_t>(T);
    const int64_t h_ext = static_cast<int64_t>(H);
    const int64_t bh = 128;
    const int64_t h_tiles = luisa::ceil_div(h_ext, bh);

    auto def = tile_kernel("embed_mm_tile_audio_glu", [=](
        TensorView<const float, 2> in_x,
        TensorView<float, 2> out_y) {
        auto m = axis("m", 1);
        auto n = axis("n", bh);
        auto t_grid = axis("t_grid", t_ext);
        auto h_grid = axis("h_grid", h_tiles);
        for (auto &nest : parallel(shape(t_grid, h_grid))) {
            auto t = nest.index(t_grid);
            auto col = nest.index(h_grid) * bh;
            auto x1 = in_x.tile(coord(t, col), shape(m, n)).load();
            auto x2 = in_x.tile(coord(t, h_ext + col), shape(m, n)).load();
            auto y = x1 / (1.0f + exp(-x2));
            out_y(coord(t, col), shape(m, n)).store(y);
        }
    });
    return def.capture(tensor_shape(t_ext, h_ext * 2), tensor_shape(t_ext, h_ext));
}

compute::tile::Kernel make_tile_audio_add_scale_kernel(uint32_t N, float scale) noexcept {
    using namespace compute::tile;
    const int64_t n_ext = static_cast<int64_t>(N);
    const int64_t blocks = luisa::ceil_div(n_ext, kVecTile);

    auto def = tile_kernel("embed_mm_tile_audio_add_scale", [=](
        TensorView<const float, 1> x,
        TensorView<const float, 1> residual,
        TensorView<float, 1> y) {
        auto v = axis("v", kVecTile);
        for (auto &nest : parallel(shape(blocks))) {
            auto base = nest.index() * kVecTile;
            auto xv = x.tile(coord(base), shape(v)).load();
            auto rv = residual.tile(coord(base), shape(v)).load();
            y(coord(base), shape(v)).store(xv * scale + rv);
        }
    });
    return def.capture(tensor_shape(n_ext), tensor_shape(n_ext), tensor_shape(n_ext));
}

compute::tile::Kernel make_tile_audio_add_scale_inplace_kernel(uint32_t N, float scale) noexcept {
    using namespace compute::tile;
    const int64_t n_ext = static_cast<int64_t>(N);
    const int64_t blocks = luisa::ceil_div(n_ext, kVecTile);

    auto def = tile_kernel("embed_mm_tile_audio_add_scale_inplace", [=](
        TensorView<const float, 1> x,
        TensorView<float, 1> residual_inout) {
        auto v = axis("v", kVecTile);
        for (auto &nest : parallel(shape(blocks))) {
            auto base = nest.index() * kVecTile;
            auto xv = x.tile(coord(base), shape(v)).load();
            auto rv = residual_inout.tile(coord(base), shape(v)).load();
            residual_inout(coord(base), shape(v)).store(xv * scale + rv);
        }
    });
    return def.capture(tensor_shape(n_ext), tensor_shape(n_ext));
}

TileCmd dispatch_tile_geglu(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> gate,
    compute::BufferView<float> up,
    compute::BufferView<float> out,
    uint32_t N) {
    TileShaderKey key{make_op_tag("geglu"), N, 0, 0, 0, 0};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_geglu_kernel(N);
    });
    return shader(gate, up, out).dispatch();
}

TileCmd dispatch_tile_residual_add(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> x,
    compute::BufferView<float> residual,
    compute::BufferView<float> out,
    uint32_t N) {
    if (same_region(x, out)) {
        TileShaderKey key{make_op_tag("residual_add_inplace"), N, 0, 0, 0, 0};
        const auto &shader = cache.get_or_compile(device, key, [&]() {
            return make_tile_residual_add_inplace_kernel(N);
        });
        return shader(out, residual).dispatch();
    }
    if (same_region(residual, out)) {
        TileShaderKey key{make_op_tag("residual_add_inplace"), N, 0, 0, 0, 0};
        const auto &shader = cache.get_or_compile(device, key, [&]() {
            return make_tile_residual_add_inplace_kernel(N);
        });
        return shader(out, x).dispatch();
    }
    TileShaderKey key{make_op_tag("residual_add"), N, 0, 0, 0, 0};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_residual_add_kernel(N);
    });
    return shader(x, residual, out).dispatch();
}

TileCmd dispatch_tile_ple_gate(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> gate,
    compute::BufferView<float> ple_all,
    compute::BufferView<float> out,
    uint32_t layer_idx,
    uint32_t total_tokens,
    uint32_t num_layers,
    uint32_t H) {
    TileShaderKey key{make_op_tag("ple_gate"), layer_idx, total_tokens, num_layers, H, 0};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_ple_gate_kernel(layer_idx, total_tokens, num_layers, H);
    });
    return shader(gate, ple_all, out).dispatch();
}

TileCmd dispatch_tile_scale_add(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> x,
    compute::BufferView<float> residual,
    compute::BufferView<float> out,
    float scale,
    uint32_t N) {
    const uint64_t scale_bits = std::bit_cast<uint32_t>(scale);
    if (same_region(x, out)) {
        TileShaderKey key{make_op_tag("scale_add_inplace"), N, scale_bits, 0, 0, 0};
        const auto &shader = cache.get_or_compile(device, key, [&]() {
            return make_tile_scale_add_inplace_kernel(N, scale);
        });
        return shader(out, residual).dispatch();
    }
    if (same_region(residual, out)) {
        TileShaderKey key{make_op_tag("scale_add_inplace"), N, scale_bits, 0, 0, 0};
        const auto &shader = cache.get_or_compile(device, key, [&]() {
            return make_tile_scale_add_inplace_kernel(N, scale);
        });
        return shader(out, x).dispatch();
    }
    TileShaderKey key{make_op_tag("scale_add"), N, scale_bits, 0, 0, 0};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_scale_add_kernel(N, scale);
    });
    return shader(x, residual, out).dispatch();
}

TileCmd dispatch_tile_audio_clamp(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> x,
    compute::BufferView<float> y,
    float min_v,
    float max_v,
    uint32_t N) {
    const uint64_t min_bits = std::bit_cast<uint32_t>(min_v);
    const uint64_t max_bits = std::bit_cast<uint32_t>(max_v);
    if (same_region(x, y)) {
        TileShaderKey key{make_op_tag("audio_clamp_inplace"), N, min_bits, max_bits, 0, 0};
        const auto &shader = cache.get_or_compile(device, key, [&]() {
            return make_tile_audio_clamp_inplace_kernel(N, min_v, max_v);
        });
        return shader(x).dispatch();
    }
    TileShaderKey key{make_op_tag("audio_clamp"), N, min_bits, max_bits, 0, 0};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_audio_clamp_kernel(N, min_v, max_v);
    });
    return shader(x, y).dispatch();
}

TileCmd dispatch_tile_audio_silu(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> x,
    compute::BufferView<float> y,
    uint32_t N) {
    if (same_region(x, y)) {
        TileShaderKey key{make_op_tag("audio_silu_inplace"), N, 0, 0, 0, 0};
        const auto &shader = cache.get_or_compile(device, key, [&]() {
            return make_tile_audio_silu_inplace_kernel(N);
        });
        return shader(x).dispatch();
    }
    TileShaderKey key{make_op_tag("audio_silu"), N, 0, 0, 0, 0};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_audio_silu_kernel(N);
    });
    return shader(x, y).dispatch();
}

TileCmd dispatch_tile_audio_glu(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> in_x,
    compute::BufferView<float> out_y,
    uint32_t T,
    uint32_t H) {
    TileShaderKey key{make_op_tag("audio_glu"), T, H, 0, 0, 0};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_audio_glu_kernel(T, H);
    });
    return shader(in_x, out_y).dispatch();
}

TileCmd dispatch_tile_audio_add_scale(
    compute::Device &device,
    TileShaderCache &cache,
    compute::BufferView<float> x,
    compute::BufferView<float> residual,
    compute::BufferView<float> y,
    float scale,
    uint32_t N) {
    const uint64_t scale_bits = std::bit_cast<uint32_t>(scale);
    if (same_region(residual, y)) {
        TileShaderKey key{make_op_tag("audio_add_scale_inplace"), N, scale_bits, 0, 0, 0};
        const auto &shader = cache.get_or_compile(device, key, [&]() {
            return make_tile_audio_add_scale_inplace_kernel(N, scale);
        });
        return shader(x, y).dispatch();
    }
    TileShaderKey key{make_op_tag("audio_add_scale"), N, scale_bits, 0, 0, 0};
    const auto &shader = cache.get_or_compile(device, key, [&]() {
        return make_tile_audio_add_scale_kernel(N, scale);
    });
    return shader(x, residual, y).dispatch();
}

}// namespace luisa::embed_mm::ops_tile
