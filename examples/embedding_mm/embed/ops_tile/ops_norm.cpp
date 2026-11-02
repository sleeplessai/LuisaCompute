#include "embed/ops_tile/ops_norm.h"

namespace luisa::embed_mm::ops_tile {

namespace {

enum class NormKind : uint32_t { RMS = 0u, RMS_NO_WEIGHT = 1u, L2 = 2u };

[[nodiscard]] compute::tile::Kernel make_norm_kernel(NormKind kind, bool inplace,
                                                     uint32_t rows, uint32_t H, float eps) {
    using namespace compute::tile;
    const int64_t r_ext = rows, h_ext = H;
    const int64_t pad_h = next_pow2(H);
    const float inv_h = 1.0f / static_cast<float>(H);

    auto normalize = [=](const Tile<float> &x, const Axis &m, const Axis &n) {
        if (h_ext == pad_h) {
            auto ss = reduce(x * x, n, add);
            auto denom = (kind == NormKind::L2) ? max(sqrt(ss), 1e-12f) : sqrt(ss * inv_h + eps);
            return x / denom;
        } else {
            auto valid = iota(n) < h_ext;
            auto x_valid = ite(valid, x, 0.0f);
            auto ss = reduce(x_valid * x_valid, n, add);
            auto denom = (kind == NormKind::L2) ? max(sqrt(ss), 1e-12f) : sqrt(ss * inv_h + eps);
            return ite(valid, x / denom, 0.0f);
        }
    };

    if (kind == NormKind::RMS) {
        if (inplace) {
            return tile_kernel("embed_mm_tile_rmsnorm_inplace", [=](TensorView<float, 2> X, TensorView<const float, 2> G) {
                       auto ra = axis("rows", r_ext);
                       auto m = axis("m", 1), n = axis("n", pad_h);
                       for (auto &nest : parallel(shape(ra))) {
                           auto row = nest.index(ra);
                           auto x = X.tile(coord(row, 0), shape(m, n)).load();
                           auto g = G.tile(coord(0, 0), shape(m, n)).load();
                           X(coord(row, 0), shape(m, n)).store(normalize(x, m, n) * g);
                       }
                   })
                .capture(tensor_shape(r_ext, h_ext), tensor_shape(1, h_ext));
        }
        return tile_kernel("embed_mm_tile_rmsnorm", [=](TensorView<const float, 2> X, TensorView<const float, 2> G, TensorView<float, 2> Y) {
                   auto ra = axis("rows", r_ext);
                   auto m = axis("m", 1), n = axis("n", pad_h);
                   for (auto &nest : parallel(shape(ra))) {
                       auto row = nest.index(ra);
                       auto x = X.tile(coord(row, 0), shape(m, n)).load();
                       auto g = G.tile(coord(0, 0), shape(m, n)).load();
                       Y(coord(row, 0), shape(m, n)).store(normalize(x, m, n) * g);
                   }
               })
            .capture(tensor_shape(r_ext, h_ext), tensor_shape(1, h_ext), tensor_shape(r_ext, h_ext));
    }
    if (inplace) {
        return tile_kernel(kind == NormKind::L2 ? "embed_mm_tile_l2norm_inplace" : "embed_mm_tile_rmsnorm_nw_inplace",
                           [=](TensorView<float, 2> X) {
                               auto ra = axis("rows", r_ext);
                               auto m = axis("m", 1), n = axis("n", pad_h);
                               for (auto &nest : parallel(shape(ra))) {
                                   auto row = nest.index(ra);
                                   auto x = X.tile(coord(row, 0), shape(m, n)).load();
                                   X(coord(row, 0), shape(m, n)).store(normalize(x, m, n));
                               }
                           })
            .capture(tensor_shape(r_ext, h_ext));
    }
    return tile_kernel(kind == NormKind::L2 ? "embed_mm_tile_l2norm" : "embed_mm_tile_rmsnorm_nw",
                       [=](TensorView<const float, 2> X, TensorView<float, 2> Y) {
                           auto ra = axis("rows", r_ext);
                           auto m = axis("m", 1), n = axis("n", pad_h);
                           for (auto &nest : parallel(shape(ra))) {
                               auto row = nest.index(ra);
                               auto x = X.tile(coord(row, 0), shape(m, n)).load();
                               Y(coord(row, 0), shape(m, n)).store(normalize(x, m, n));
                           }
                       })
        .capture(tensor_shape(r_ext, h_ext), tensor_shape(r_ext, h_ext));
}

}// namespace

TileCmd dispatch_tile_rmsnorm(compute::Device &device, TileShaderCache &cache,
                              compute::BufferView<float> x, compute::BufferView<float> y,
                              compute::BufferView<float> gamma, uint32_t rows, uint32_t H, float eps) {
    auto inplace = same_region(x, y);
    TileShaderKey key{make_op_tag("rmsnorm"), rows, H, float_bits(eps), inplace ? 1u : 0u};
    const auto &shader = cache.get_or_compile(device, key, [&] { return make_norm_kernel(NormKind::RMS, inplace, rows, H, eps); });
    return inplace ? shader(x, gamma).dispatch() : shader(x, gamma, y).dispatch();
}

TileCmd dispatch_tile_rmsnorm_no_weight(compute::Device &device, TileShaderCache &cache,
                                        compute::BufferView<float> x, compute::BufferView<float> y,
                                        uint32_t rows, uint32_t H, float eps) {
    auto inplace = same_region(x, y);
    TileShaderKey key{make_op_tag("rmsnorm_nw"), rows, H, float_bits(eps), inplace ? 1u : 0u};
    const auto &shader = cache.get_or_compile(device, key, [&] { return make_norm_kernel(NormKind::RMS_NO_WEIGHT, inplace, rows, H, eps); });
    return inplace ? shader(x).dispatch() : shader(x, y).dispatch();
}

TileCmd dispatch_tile_l2_normalize(compute::Device &device, TileShaderCache &cache,
                                   compute::BufferView<float> x, compute::BufferView<float> y,
                                   uint32_t rows, uint32_t D) {
    auto inplace = same_region(x, y);
    TileShaderKey key{make_op_tag("l2norm"), rows, D, 0u, inplace ? 1u : 0u};
    const auto &shader = cache.get_or_compile(device, key, [&] { return make_norm_kernel(NormKind::L2, inplace, rows, D, 0.0f); });
    return inplace ? shader(x).dispatch() : shader(x, y).dispatch();
}

}// namespace luisa::embed_mm::ops_tile
