// embedding_mm engine -- Gemma4VisionModel + embed_vision connector (Tile backend).
#include "embedding_mm/embed/models_tile/vision_tower_tile.h"

#include <cmath>
#include <luisa/core/logging.h>

namespace luisa::embed_mm {

void VisionTowerTile::load_weights(
    const Package &package,
    compute::CommandList &upload_cmds,
    luisa::vector<luisa::vector<float>> &fp32_storage,
    luisa::vector<luisa::vector<luisa::bfloat16>> &bf16_storage) {

    LUISA_INFO("VisionTowerTile: uploading weight tensors (BF16 Tensor Core MMA)...");

    _patch_proj = upload_tensor_bf16(_device, package, "vision.patch_embed.proj", upload_cmds, bf16_storage);
    _pos_table = upload_tensor_fp32(_device, package, "vision.patch_embed.pos_table", upload_cmds, fp32_storage);

    // Precompute axial RoPE cos/sin tables [10240, 16]
    constexpr uint32_t pos_table_size = 10240u;
    constexpr uint32_t quarter_dim = 16u;
    constexpr float rope_theta = 100.0f;
    auto &cos_host = fp32_storage.emplace_back();
    auto &sin_host = fp32_storage.emplace_back();
    ops_tile::build_axial_rope_cos_sin_tables(pos_table_size, quarter_dim, rope_theta, cos_host, sin_host);
    _cos_table = _device.create_buffer<float>(cos_host.size());
    _sin_table = _device.create_buffer<float>(sin_host.size());
    upload_cmds << _cos_table.copy_from(cos_host.data())
                << _sin_table.copy_from(sin_host.data());

    _layers.clear();
    _layers.reserve(16u);

    for (uint32_t l = 0u; l < 16u; ++l) {
        VisionLayerWeightsTile layer;
        layer.input_norm = upload_tensor_fp32(_device, package, luisa::format("vision.layer{}.input_norm", l), upload_cmds, fp32_storage);
        layer.q_proj = upload_tensor_bf16(_device, package, luisa::format("vision.layer{}.q_proj", l), upload_cmds, bf16_storage);
        layer.k_proj = upload_tensor_bf16(_device, package, luisa::format("vision.layer{}.k_proj", l), upload_cmds, bf16_storage);
        layer.v_proj = upload_tensor_bf16(_device, package, luisa::format("vision.layer{}.v_proj", l), upload_cmds, bf16_storage);
        layer.q_norm = upload_tensor_fp32(_device, package, luisa::format("vision.layer{}.q_norm", l), upload_cmds, fp32_storage);
        layer.k_norm = upload_tensor_fp32(_device, package, luisa::format("vision.layer{}.k_norm", l), upload_cmds, fp32_storage);
        layer.out_proj = upload_tensor_bf16(_device, package, luisa::format("vision.layer{}.out_proj", l), upload_cmds, bf16_storage);
        layer.post_attn_norm = upload_tensor_fp32(_device, package, luisa::format("vision.layer{}.post_attn_norm", l), upload_cmds, fp32_storage);
        layer.pre_ffn_norm = upload_tensor_fp32(_device, package, luisa::format("vision.layer{}.pre_ffn_norm", l), upload_cmds, fp32_storage);
        layer.gate_proj = upload_tensor_bf16(_device, package, luisa::format("vision.layer{}.gate_proj", l), upload_cmds, bf16_storage);
        layer.up_proj = upload_tensor_bf16(_device, package, luisa::format("vision.layer{}.up_proj", l), upload_cmds, bf16_storage);
        layer.down_proj = upload_tensor_bf16(_device, package, luisa::format("vision.layer{}.down_proj", l), upload_cmds, bf16_storage);
        layer.post_ffn_norm = upload_tensor_fp32(_device, package, luisa::format("vision.layer{}.post_ffn_norm", l), upload_cmds, fp32_storage);

        _layers.push_back(std::move(layer));
    }

    _connector_proj = upload_tensor_bf16(_device, package, "vision.connector.proj", upload_cmds, bf16_storage);
}

void VisionTowerTile::ensure_workspace(uint32_t max_patches, uint32_t num_soft_tokens) {
    size_t alloc_patches = luisa::ceil_div(max_patches, 64u) * 64u;
    size_t alloc_soft_tokens = luisa::ceil_div(num_soft_tokens, 64u) * 64u;

    if (alloc_patches <= _max_patches && alloc_soft_tokens <= _max_soft_tokens) {
        return;
    }

    _max_patches = std::max(_max_patches, alloc_patches);
    _max_soft_tokens = std::max(_max_soft_tokens, alloc_soft_tokens);

    constexpr size_t H = 768;
    constexpr size_t I = 3072;

    _scaled_pixels = _device.create_buffer<float>(_max_patches * H);
    _h0 = _device.create_buffer<float>(_max_patches * H);
    _mask = _device.create_buffer<float>(_max_patches);
    _h_norm = _device.create_buffer<float>(_max_patches * H);
    _q_buf = _device.create_buffer<float>(_max_patches * H);
    _k_buf = _device.create_buffer<float>(_max_patches * H);
    _v_buf = _device.create_buffer<float>(_max_patches * H);
    _attn_ctx = _device.create_buffer<float>(_max_patches * H);
    _attn_out = _device.create_buffer<float>(_max_patches * H);
    _attn_norm = _device.create_buffer<float>(_max_patches * H);
    _ffn_norm = _device.create_buffer<float>(_max_patches * H);
    _gate_buf = _device.create_buffer<float>(_max_patches * I);
    _up_buf = _device.create_buffer<float>(_max_patches * I);
    _mlp_act = _device.create_buffer<float>(_max_patches * I);
    _mlp_out = _device.create_buffer<float>(_max_patches * H);
    _mlp_norm = _device.create_buffer<float>(_max_patches * H);
    _pooled_tokens = _device.create_buffer<float>(_max_soft_tokens * H);
    _pooled_norm = _device.create_buffer<float>(_max_soft_tokens * H);
}

void VisionTowerTile::record_encode(
    compute::CommandList &cmd_list,
    compute::BufferView<float> pixel_values,
    compute::BufferView<int32_t> position_ids,
    compute::BufferView<float> out_soft_tokens,
    uint32_t B,
    uint32_t num_patches_w,
    uint32_t num_patches_h,
    uint32_t max_patches,
    uint32_t num_soft_tokens,
    Executor::VisionDebugActivations *debug_acts) {

    ensure_workspace(max_patches, num_soft_tokens);

    constexpr uint32_t H = 768u;
    constexpr uint32_t I = 3072u;
    constexpr uint32_t num_heads = 12u;
    constexpr uint32_t head_dim = 64u;
    constexpr float scale_pool = 3.079201435678004f;// sqrt(768) / 9.0

    // 1. Rescale pixels [0, 1] -> [-1, 1]
    cmd_list << ops_tile::dispatch_tile_scale_pixels(
        _device, _cache, pixel_values, _scaled_pixels, max_patches * H);

    // 2. Patch embedding GEMM: [max_patches, 768] * [768, 768] -> [max_patches, 768]
    cmd_list << ops_tile::dispatch_tile_gemm(
        _device, _cache, _scaled_pixels, _patch_proj, _h0, max_patches, H, H);

    // 3. Add 2D position embeddings
    auto pos_ids_u32 = position_ids.as<uint32_t>();
    cmd_list << ops_tile::dispatch_tile_add_patch_pos_embed(
        _device, _cache, _h0, _pos_table, pos_ids_u32, max_patches, H, 10240u);

    if (debug_acts && debug_acts->patch_embed) {
        cmd_list << debug_acts->patch_embed.copy_from(_h0.view(0, max_patches * H));
    }

    // 4. Generate attention mask from position_ids
    cmd_list << ops_tile::dispatch_tile_mask_from_pos(
        _device, _cache, pos_ids_u32, _mask, max_patches);

    // 5. 16 Vision Transformer layers
    for (uint32_t l = 0u; l < 16u; ++l) {
        const auto &layer = _layers[l];

        // (a) Input RMSNorm
        cmd_list << ops_tile::dispatch_tile_rmsnorm(
            _device, _cache, _h0, _h_norm, layer.input_norm, max_patches, H, 1e-6f);

        // (b) Q, K, V projections
        cmd_list << ops_tile::dispatch_tile_gemm(_device, _cache, _h_norm, layer.q_proj, _q_buf, max_patches, H, H);
        cmd_list << ops_tile::dispatch_tile_gemm(_device, _cache, _h_norm, layer.k_proj, _k_buf, max_patches, H, H);
        cmd_list << ops_tile::dispatch_tile_gemm(_device, _cache, _h_norm, layer.v_proj, _v_buf, max_patches, H, H);

        // (c) Q, K, V RMSNorm
        cmd_list << ops_tile::dispatch_tile_rmsnorm(_device, _cache, _q_buf, _q_buf, layer.q_norm, max_patches * num_heads, head_dim, 1e-6f);
        cmd_list << ops_tile::dispatch_tile_rmsnorm(_device, _cache, _k_buf, _k_buf, layer.k_norm, max_patches * num_heads, head_dim, 1e-6f);
        cmd_list << ops_tile::dispatch_tile_rmsnorm_no_weight(_device, _cache, _v_buf, _v_buf, max_patches * num_heads, head_dim, 1e-6f);

        // (d) 2D Axial RoPE
        cmd_list << ops_tile::dispatch_tile_axial_rope(
            _device, _cache, _q_buf, _q_buf, pos_ids_u32, _cos_table, _sin_table, max_patches, num_heads, head_dim, 10240u);
        cmd_list << ops_tile::dispatch_tile_axial_rope(
            _device, _cache, _k_buf, _k_buf, pos_ids_u32, _cos_table, _sin_table, max_patches, num_heads, head_dim, 10240u);

        // (e) Fused Attention
        cmd_list << ops_tile::dispatch_tile_gqa_attention(
            _device, _cache, _q_buf, _k_buf, _v_buf, _mask, _attn_ctx,
            1u, max_patches, num_heads, num_heads, head_dim, 1.0f, 0u, 0u);

        // (f) Out projection GEMM
        cmd_list << ops_tile::dispatch_tile_gemm(_device, _cache, _attn_ctx, layer.out_proj, _attn_out, max_patches, H, H);

        // (g) Post-attn RMSNorm + Residual add
        cmd_list << ops_tile::dispatch_tile_rmsnorm(_device, _cache, _attn_out, _attn_norm, layer.post_attn_norm, max_patches, H, 1e-6f);
        cmd_list << ops_tile::dispatch_tile_residual_add(_device, _cache, _h0, _attn_norm, _h0, max_patches * H);

        // (h) Pre-FFN RMSNorm
        cmd_list << ops_tile::dispatch_tile_rmsnorm(_device, _cache, _h0, _ffn_norm, layer.pre_ffn_norm, max_patches, H, 1e-6f);

        // (i) MLP projections
        cmd_list << ops_tile::dispatch_tile_gemm(_device, _cache, _ffn_norm, layer.gate_proj, _gate_buf, max_patches, I, H);
        cmd_list << ops_tile::dispatch_tile_gemm(_device, _cache, _ffn_norm, layer.up_proj, _up_buf, max_patches, I, H);

        // (j) GeGLU
        cmd_list << ops_tile::dispatch_tile_geglu(_device, _cache, _gate_buf, _up_buf, _mlp_act, max_patches * I);

        // (k) Down projection
        cmd_list << ops_tile::dispatch_tile_gemm(_device, _cache, _mlp_act, layer.down_proj, _mlp_out, max_patches, H, I);

        // (l) Post-FFN RMSNorm + Residual add
        cmd_list << ops_tile::dispatch_tile_rmsnorm(_device, _cache, _mlp_out, _mlp_norm, layer.post_ffn_norm, max_patches, H, 1e-6f);
        cmd_list << ops_tile::dispatch_tile_residual_add(_device, _cache, _h0, _mlp_norm, _h0, max_patches * H);

        if (debug_acts && debug_acts->layer0_out && l == 0u) {
            cmd_list << debug_acts->layer0_out.copy_from(_h0.view(0, max_patches * H));
        }
    }

    // 6. 2D 3x3 Spatial Average Pooling
    uint32_t gw = num_patches_w / 3u;
    uint32_t gh = num_patches_h / 3u;
    cmd_list << ops_tile::dispatch_tile_vision_pool(
        _device, _cache, _h0, _pooled_tokens, max_patches, gw, gh, num_patches_w, H, scale_pool);

    if (debug_acts && debug_acts->pooled_out) {
        cmd_list << debug_acts->pooled_out.copy_from(_pooled_tokens.view(0, num_soft_tokens * H));
    }

    // 7. Connector pre-norm (without scale)
    cmd_list << ops_tile::dispatch_tile_rmsnorm_no_weight(
        _device, _cache, _pooled_tokens, _pooled_norm, num_soft_tokens, H, 1e-6f);

    // 8. Connector projection: [num_soft_tokens, 768] * [768, 512] -> [num_soft_tokens, 512]
    cmd_list << ops_tile::dispatch_tile_gemm(
        _device, _cache, _pooled_norm, _connector_proj, out_soft_tokens, num_soft_tokens, 512u, H);
}

} // namespace luisa::embed_mm
