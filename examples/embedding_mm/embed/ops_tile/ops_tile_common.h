#pragma once
// embedding_mm engine -- Common Tile compilation and shader caching utilities.
//
// Conventions shared by every ops_tile operator:
// * Tile kernels capture static tensor extents, so each distinct shape is
//   compiled once and memoised in a per-model TileShaderCache.
// * Every Tile-local extent (tile shape) is a power of two (CUDA native Tile IR
//   requirement); tensor extents and loop domains may be arbitrary and partial
//   tiles are zero-masked on load / bounds-masked on store.
// * Writable bindings must never alias another binding (TIRx disjoint_writes):
//   in-place operators use a dedicated single read-write TensorView variant.

#include <bit>
#include <cstdint>

#include <luisa/core/basic_types.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/tile.h>
#include <luisa/tile/runtime.h>

namespace luisa::embed_mm::ops_tile {

using TileCmd = luisa::unique_ptr<compute::Command>;

struct TileShaderKey {
    uint64_t op_tag = 0;
    uint64_t d0 = 0;
    uint64_t d1 = 0;
    uint64_t d2 = 0;
    uint64_t d3 = 0;
    uint64_t extra = 0;

    [[nodiscard]] bool operator==(const TileShaderKey &o) const noexcept {
        return op_tag == o.op_tag && d0 == o.d0 && d1 == o.d1 && d2 == o.d2 && d3 == o.d3 && extra == o.extra;
    }
};

struct TileShaderKeyHash {
    using is_avalanching = void;
    [[nodiscard]] size_t operator()(const TileShaderKey &k) const noexcept {
        uint64_t h = k.op_tag;
        auto mix = [&](uint64_t v) noexcept { h ^= v + 0x9e3779b97f4a7c15ull + (h << 6u) + (h >> 2u); };
        mix(k.d0);
        mix(k.d1);
        mix(k.d2);
        mix(k.d3);
        mix(k.extra);
        return static_cast<size_t>(h);
    }
};

[[nodiscard]] constexpr uint64_t make_op_tag(const char *str) noexcept {
    uint64_t h = 14695981039346656037ull;
    while (*str) {
        h ^= static_cast<uint64_t>(static_cast<unsigned char>(*str++));
        h *= 1099511628211ull;
    }
    return h;
}

[[nodiscard]] inline uint64_t float_bits(float f) noexcept { return std::bit_cast<uint32_t>(f); }

[[nodiscard]] inline int64_t next_pow2(uint32_t x) noexcept {
    return static_cast<int64_t>(std::bit_ceil(x == 0u ? 1u : x));
}

template<typename T>
[[nodiscard]] inline bool same_region(compute::BufferView<T> a, compute::BufferView<T> b) noexcept {
    return a.handle() == b.handle() && a.offset_bytes() == b.offset_bytes();
}

// Compile a Tile kernel with automatic NATIVE (CUDA Tile IR / Tensor Core) -> TIRX fallback.
// NATIVE is enabled by default (LUISA_CUDA_TILE_IR=1 is exported unless the user set it);
// set LUISA_EMBED_MM_TILE_LOWERING=tirx to force TIRX.
[[nodiscard]] compute::tile::Shader compile_tile_kernel(
    compute::Device &device,
    const compute::tile::Kernel &kernel) noexcept;

class TileShaderCache {
public:
    TileShaderCache() noexcept = default;
    TileShaderCache(const TileShaderCache &) = delete;
    TileShaderCache &operator=(const TileShaderCache &) = delete;

    template<typename Builder>
    [[nodiscard]] const compute::tile::Shader &get_or_compile(
        compute::Device &device, const TileShaderKey &key, Builder &&builder) {
        if (auto it = _shaders.find(key); it != _shaders.end()) { return *it->second; }
        auto kernel = builder();
        auto shader = luisa::make_unique<compute::tile::Shader>(compile_tile_kernel(device, kernel));
        auto [ins, _] = _shaders.emplace(key, std::move(shader));
        return *ins->second;
    }

    void clear() noexcept { _shaders.clear(); }
    [[nodiscard]] size_t size() const noexcept { return _shaders.size(); }

private:
    luisa::unordered_map<TileShaderKey, luisa::unique_ptr<compute::tile::Shader>, TileShaderKeyHash> _shaders;
};

}// namespace luisa::embed_mm::ops_tile
