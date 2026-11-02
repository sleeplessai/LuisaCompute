#pragma once
// embedding_mm engine -- Common weight uploading and helpers for Tile models.

#include <cstring>
#include <luisa/core/basic_types.h>
#include <luisa/core/bfloat16.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/device.h>

#include "embedding_mm/embed/package.h"

namespace luisa::embed_mm {

[[nodiscard]] compute::Buffer<luisa::bfloat16> upload_tensor_bf16(
    compute::Device &device,
    const Package &package,
    luisa::string_view name,
    compute::CommandList &upload_cmds,
    luisa::vector<luisa::vector<luisa::bfloat16>> &bf16_storage);

[[nodiscard]] compute::Buffer<float> upload_tensor_fp32(
    compute::Device &device,
    const Package &package,
    luisa::string_view name,
    compute::CommandList &upload_cmds,
    luisa::vector<luisa::vector<float>> &fp32_storage);

} // namespace luisa::embed_mm
