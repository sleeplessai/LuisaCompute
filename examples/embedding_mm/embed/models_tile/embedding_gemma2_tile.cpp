// embedding_mm engine -- Top-level EmbeddingGemma-2 multimodal model orchestrator (Tile backend).
#include "embedding_mm/embed/models_tile/embedding_gemma2_tile.h"

#include <luisa/core/logging.h>
#include <luisa/runtime/stream.h>

namespace luisa::embed_mm {

void EmbeddingGemma2Tile::initialize(const Package &package, bool enable_vision, bool enable_audio) {
    auto upload_stream = _device.create_stream(compute::StreamTag::COMPUTE);
    compute::CommandList upload_cmds;
    luisa::vector<luisa::vector<float>> fp32_storage;
    luisa::vector<luisa::vector<luisa::bfloat16>> bf16_storage;

    // 1. Load Language Model weights
    _language_model.load_weights(package, upload_cmds, fp32_storage, bf16_storage);

    // 2. Selective Encoder Loading: Vision Tower
    if (enable_vision && package.has_vision()) {
        LUISA_INFO("EmbeddingGemma2Tile: Selective Encoder Loading -> initializing Vision Tower Tile...");
        _vision_tower.emplace(_device);
        _vision_tower->load_weights(package, upload_cmds, fp32_storage, bf16_storage);
    } else {
        LUISA_INFO("EmbeddingGemma2Tile: Selective Encoder Loading -> Vision Tower skipped.");
    }

    // 3. Selective Encoder Loading: Audio Tower
    if (enable_audio && package.has_audio()) {
        LUISA_INFO("EmbeddingGemma2Tile: Selective Encoder Loading -> initializing Audio Tower Tile...");
        _audio_tower.emplace(_device);
        _audio_tower->load_weights(package, upload_cmds, fp32_storage, bf16_storage);
    } else {
        LUISA_INFO("EmbeddingGemma2Tile: Selective Encoder Loading -> Audio Tower skipped.");
    }

    upload_stream << upload_cmds.commit() << compute::synchronize();

    // 4. Compile shaders (pre-warming)
    _language_model.compile_shaders();
    if (_vision_tower) {
        _vision_tower->compile_shaders();
    }
    if (_audio_tower) {
        _audio_tower->compile_shaders();
    }

    LUISA_INFO("EmbeddingGemma2Tile: initialized and ready (vision={}, audio={})",
               has_vision() ? "enabled" : "disabled",
               has_audio() ? "enabled" : "disabled");
}

void EmbeddingGemma2Tile::record_vision_encode(
    compute::CommandList &cmd_list,
    compute::BufferView<float> pixel_values,
    compute::BufferView<int32_t> position_ids,
    compute::BufferView<float> out_soft_tokens,
    uint32_t B,
    uint32_t num_patches_w,
    uint32_t num_patches_h,
    uint32_t max_patches,
    uint32_t num_soft_tokens,
    VisionDebugActivations *debug_acts) {
    if (!_vision_tower) {
        LUISA_ERROR("EmbeddingGemma2Tile::record_vision_encode: Vision Tower not enabled.");
    }
    _vision_tower->record_encode(
        cmd_list, pixel_values, position_ids, out_soft_tokens,
        B, num_patches_w, num_patches_h, max_patches, num_soft_tokens, debug_acts);
}

void EmbeddingGemma2Tile::record_audio_encode(
    compute::CommandList &cmd_list,
    compute::BufferView<float> input_features,
    compute::BufferView<float> out_soft_tokens,
    uint32_t num_frames,
    uint32_t num_mel_bins,
    uint32_t num_soft_tokens,
    AudioDebugActivations *debug_acts) {
    if (!_audio_tower) {
        LUISA_ERROR("EmbeddingGemma2Tile::record_audio_encode: Audio Tower not enabled.");
    }
    _audio_tower->record_encode(
        cmd_list, input_features, out_soft_tokens,
        num_frames, num_mel_bins, num_soft_tokens, debug_acts);
}

void EmbeddingGemma2Tile::record_encode(
    compute::CommandList &cmd_list,
    compute::BufferView<uint32_t> input_ids,
    compute::BufferView<float> attention_mask,
    compute::BufferView<float> out_embeddings,
    uint32_t B,
    uint32_t S,
    uint32_t target_dim,
    const MultimodalInputs &mm_inputs) {
    _language_model.record_encode(
        cmd_list, input_ids, attention_mask, out_embeddings,
        B, S, target_dim, mm_inputs);
}

} // namespace luisa::embed_mm
