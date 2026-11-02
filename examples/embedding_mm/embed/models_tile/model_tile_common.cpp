#include "embedding_mm/embed/models_tile/model_tile_common.h"

namespace luisa::embed_mm {

compute::Buffer<luisa::bfloat16> upload_tensor_bf16(
    compute::Device &device,
    const Package &package,
    luisa::string_view name,
    compute::CommandList &upload_cmds,
    luisa::vector<luisa::vector<luisa::bfloat16>> &bf16_storage) {
    const auto *t = package.find(name);
    if (!t) {
        LUISA_ERROR("upload_tensor_bf16: missing required weight tensor '{}'", name);
    }
    auto elem_count = static_cast<size_t>(t->element_count());
    auto buf = device.create_buffer<luisa::bfloat16>(elem_count);

    if (t->half_data && package.precision() == PackagePrecision::bf16) {
        upload_cmds << buf.copy_from(reinterpret_cast<const luisa::bfloat16 *>(t->half_data));
    } else if (t->data) {
        auto &storage = bf16_storage.emplace_back(elem_count);
        for (size_t i = 0; i < elem_count; i++) {
            storage[i] = luisa::bfloat16{t->data[i]};
        }
        upload_cmds << buf.copy_from(storage.data());
    } else if (t->half_data) {
        // FP16 -> Float -> BF16
        auto &storage = bf16_storage.emplace_back(elem_count);
        for (size_t i = 0; i < elem_count; i++) {
            uint16_t h = t->half_data[i];
            uint32_t sign = (h & 0x8000u) << 16;
            uint32_t exp = (h & 0x7C00u) >> 10;
            uint32_t mant = (h & 0x03FFu) << 13;
            uint32_t f_val = 0;
            if (exp == 0x1Fu) {
                f_val = sign | 0x7F800000u | mant;
            } else if (exp == 0) {
                if (mant != 0) {
                    exp = 113;
                    while ((mant & (1u << 23)) == 0) {
                        mant <<= 1;
                        exp--;
                    }
                    mant &= 0x007FFFFFu;
                    f_val = sign | (exp << 23) | mant;
                } else {
                    f_val = sign;
                }
            } else {
                f_val = sign | ((exp + 112) << 23) | mant;
            }
            float f;
            std::memcpy(&f, &f_val, sizeof(float));
            storage[i] = luisa::bfloat16{f};
        }
        upload_cmds << buf.copy_from(storage.data());
    }
    return buf;
}

compute::Buffer<float> upload_tensor_fp32(
    compute::Device &device,
    const Package &package,
    luisa::string_view name,
    compute::CommandList &upload_cmds,
    luisa::vector<luisa::vector<float>> &fp32_storage) {
    const auto *t = package.find(name);
    if (!t) {
        LUISA_ERROR("upload_tensor_fp32: missing required weight tensor '{}'", name);
    }
    auto elem_count = static_cast<size_t>(t->element_count());
    auto buf = device.create_buffer<float>(elem_count);

    if (t->data) {
        upload_cmds << buf.copy_from(t->data);
    } else if (t->half_data) {
        auto &storage = fp32_storage.emplace_back(elem_count);
        for (size_t i = 0; i < elem_count; i++) {
            uint16_t h = t->half_data[i];
            if (package.precision() == PackagePrecision::bf16) {
                uint32_t w = static_cast<uint32_t>(h) << 16;
                std::memcpy(&storage[i], &w, sizeof(float));
            } else {
                uint32_t sign = (h & 0x8000u) << 16;
                uint32_t exp = (h & 0x7C00u) >> 10;
                uint32_t mant = (h & 0x03FFu) << 13;
                uint32_t f_val = 0;
                if (exp == 0x1Fu) {
                    f_val = sign | 0x7F800000u | mant;
                } else if (exp == 0) {
                    if (mant != 0) {
                        exp = 113;
                        while ((mant & (1u << 23)) == 0) {
                            mant <<= 1;
                            exp--;
                        }
                        mant &= 0x007FFFFFu;
                        f_val = sign | (exp << 23) | mant;
                    } else {
                        f_val = sign;
                    }
                } else {
                    f_val = sign | ((exp + 112) << 23) | mant;
                }
                std::memcpy(&storage[i], &f_val, sizeof(float));
            }
        }
        upload_cmds << buf.copy_from(storage.data());
    }
    return buf;
}

} // namespace luisa::embed_mm
