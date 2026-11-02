#include "embed/ops_tile/ops_tile_common.h"

#include <cstdlib>

#include <luisa/core/logging.h>
#include <luisa/core/platform.h>

namespace luisa::embed_mm::ops_tile {

namespace {

void ensure_cuda_tile_ir_env() noexcept {
    static const bool initialized = [] {
#if defined(_WIN32)
        _putenv_s("LUISA_CUDA_TILE_IR", "1");
        _putenv_s("LUISA_TILE_XIR_MAX_LOCAL_BYTES", "1048576");
        // Ensure CUDA_ROOT reflects the configured CUDA_PATH for tileiras
        if (auto cuda_path = luisa::get_environment_variable("CUDA_PATH"); cuda_path && !cuda_path->empty()) {
            if (!luisa::get_environment_variable("CUDA_ROOT")) {
                _putenv_s("CUDA_ROOT", cuda_path->c_str());
            }
        }
#else
        setenv("LUISA_CUDA_TILE_IR", "1", 0);
        setenv("LUISA_TILE_XIR_MAX_LOCAL_BYTES", "1048576", 0);
        if (auto cuda_path = luisa::get_environment_variable("CUDA_PATH"); cuda_path && !cuda_path->empty()) {
            if (!luisa::get_environment_variable("CUDA_ROOT")) {
                setenv("CUDA_ROOT", cuda_path->c_str(), 0);
            }
        }
#endif
        return true;
    }();
    static_cast<void>(initialized);
}

[[nodiscard]] bool force_tirx() noexcept {
    auto v = luisa::get_environment_variable("LUISA_EMBED_MM_TILE_LOWERING");
    return v && luisa::string_view{*v} == "tirx";
}

}// namespace

compute::tile::Shader compile_tile_kernel(compute::Device &device, const compute::tile::Kernel &kernel) noexcept {
    LUISA_ASSERT(kernel.valid(), "Invalid Tile kernel passed to compile_tile_kernel.");
    ensure_cuda_tile_ir_env();
    auto kernel_name = kernel.function().name();
    if (!force_tirx()) {
        auto shader = compute::tile::compile(device, kernel, {.lowering = compute::tile::Lowering::NATIVE});
        if (shader) {
            LUISA_VERBOSE("[embed_mm/tile] {} -> NATIVE ({})", kernel_name, shader.metadata().realization);
            return shader;
        }
        LUISA_WARNING("[embed_mm/tile] {}: NATIVE unavailable ({}); retrying with TIRX.",
                      kernel_name, shader.metadata().error);
    }
    auto shader = compute::tile::compile(device, kernel, {.lowering = compute::tile::Lowering::TIRX});
    LUISA_ASSERT(static_cast<bool>(shader), "[embed_mm/tile] {}: compilation failed: {}",
                 kernel_name, shader.metadata().error);
    return shader;
}

}// namespace luisa::embed_mm::ops_tile
