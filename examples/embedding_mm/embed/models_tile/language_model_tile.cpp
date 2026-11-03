// embedding_mm engine -- Gemma4TextModel + PLE + Mean Pool + Projection + MRL Head (Tile backend).
#include "embedding_mm/embed/models_tile/language_model_tile.h"

#include <cmath>
#include <luisa/core/logging.h>
#include <luisa/runtime/stream.h>

namespace luisa::embed_mm {

void LanguageModelTile::load_weights(
    const Package &package,
    compute::CommandList &upload_cmds,
    luisa::vector<luisa::vector<float>> &fp32_storage,
    luisa::vector<luisa::vector<luisa::bfloat16>> &bf16_storage) {

    LUISA_INFO("LanguageModelTile: uploading weight tensors (BF16 Tensor Core MMA)...");

    _word_embeddings = upload_tensor_bf16(_device, package, "emb.word", upload_cmds, bf16_storage);
    _ple_proj = upload_tensor_bf16(_device, package, "ple.proj", upload_cmds, bf16_storage);
    _ple_norm = upload_tensor_fp32(_device, package, "ple.norm", upload_cmds, fp32_storage);
    _final_norm = upload_tensor_fp32(_device, package, "final_norm", upload_cmds, fp32_storage);
    _embedding_proj = upload_tensor_bf16(_device, package, "embedding_proj", upload_cmds, bf16_storage);

    _layers.clear();
    _layers.reserve(24u);

    for (uint32_t i = 0; i < 24u; ++i) {
        TextLayerWeightsTile layer;
        layer.input_norm = upload_tensor_fp32(_device, package, luisa::format("layer{}.input_norm", i), upload_cmds, fp32_storage);
        layer.q_proj = upload_tensor_bf16(_device, package, luisa::format("layer{}.q_proj", i), upload_cmds, bf16_storage);
        layer.k_proj = upload_tensor_bf16(_device, package, luisa::format("layer{}.k_proj", i), upload_cmds, bf16_storage);
        layer.v_proj = upload_tensor_bf16(_device, package, luisa::format("layer{}.v_proj", i), upload_cmds, bf16_storage);
        layer.q_norm = upload_tensor_fp32(_device, package, luisa::format("layer{}.q_norm", i), upload_cmds, fp32_storage);
        layer.k_norm = upload_tensor_fp32(_device, package, luisa::format("layer{}.k_norm", i), upload_cmds, fp32_storage);
        layer.out_proj = upload_tensor_bf16(_device, package, luisa::format("layer{}.out_proj", i), upload_cmds, bf16_storage);
        layer.post_attn_norm = upload_tensor_fp32(_device, package, luisa::format("layer{}.post_attn_norm", i), upload_cmds, fp32_storage);
        layer.pre_ffn_norm = upload_tensor_fp32(_device, package, luisa::format("layer{}.pre_ffn_norm", i), upload_cmds, fp32_storage);
        layer.gate_proj = upload_tensor_bf16(_device, package, luisa::format("layer{}.gate_proj", i), upload_cmds, bf16_storage);
        layer.up_proj = upload_tensor_bf16(_device, package, luisa::format("layer{}.up_proj", i), upload_cmds, bf16_storage);
        layer.down_proj = upload_tensor_bf16(_device, package, luisa::format("layer{}.down_proj", i), upload_cmds, bf16_storage);
        layer.post_ffn_norm = upload_tensor_fp32(_device, package, luisa::format("layer{}.post_ffn_norm", i), upload_cmds, fp32_storage);
        layer.ple_gate = upload_tensor_bf16(_device, package, luisa::format("layer{}.ple_gate", i), upload_cmds, bf16_storage);
        layer.ple_proj = upload_tensor_bf16(_device, package, luisa::format("layer{}.ple_proj", i), upload_cmds, bf16_storage);
        layer.ple_norm = upload_tensor_fp32(_device, package, luisa::format("layer{}.ple_norm", i), upload_cmds, fp32_storage);

        auto *scalar_tensor = package.find(luisa::format("layer{}.layer_scalar", i));
        if (scalar_tensor && scalar_tensor->data) {
            layer.layer_scalar = scalar_tensor->data[0];
        } else if (scalar_tensor && scalar_tensor->half_data) {
            uint16_t h = scalar_tensor->half_data[0];
            if (package.precision() == PackagePrecision::bf16) {
                uint32_t w = static_cast<uint32_t>(h) << 16;
                std::memcpy(&layer.layer_scalar, &w, sizeof(float));
            } else {
                layer.layer_scalar = 1.0f;
            }
        } else {
            layer.layer_scalar = 1.0f;
        }

        _layers.push_back(std::move(layer));
    }
}

void LanguageModelTile::ensure_workspace(uint32_t B, uint32_t S) {
    size_t alloc_tokens = luisa::ceil_div(static_cast<size_t>(B) * S, 64ull) * 64ull;
    bool need_realloc = (alloc_tokens > _max_tokens || B > _max_batch || S > _max_s);
    if (!need_realloc) {
        return;
    }

    _max_tokens = std::max(_max_tokens, alloc_tokens);
    _max_batch = std::max(_max_batch, static_cast<size_t>(B));
    
    if (S > _max_s) {
        _max_s = std::max(_max_s, static_cast<size_t>(S));
        luisa::vector<float> cos_s, sin_s, cos_f, sin_f;
        ops_tile::build_rope_cos_sin_tables(_max_s, 256u, 10000.0f, cos_s, sin_s);
        ops_tile::build_rope_cos_sin_tables(_max_s, 512u, 1000000.0f, cos_f, sin_f);
        _cos_sliding = _device.create_buffer<float>(cos_s.size());
        _sin_sliding = _device.create_buffer<float>(sin_s.size());
        _cos_full = _device.create_buffer<float>(cos_f.size());
        _sin_full = _device.create_buffer<float>(sin_f.size());
        auto upload_stream = _device.create_stream(compute::StreamTag::COMPUTE);
        upload_stream << _cos_sliding.copy_from(cos_s.data())
                      << _sin_sliding.copy_from(sin_s.data())
                      << _cos_full.copy_from(cos_f.data())
                      << _sin_full.copy_from(sin_f.data())
                      << compute::synchronize();
    }

    constexpr size_t H = 512;
    constexpr size_t I = 2048;
    constexpr size_t num_layers = 24;
    constexpr size_t max_q_dim = 4 * 512;
    constexpr size_t max_kv_dim = 2 * 512;

    _h0 = _device.create_buffer<float>(_max_tokens * H);
    _h_norm = _device.create_buffer<float>(_max_tokens * H);
    _ple_all = _device.create_buffer<float>(_max_tokens * num_layers * H);
    _q_buf = _device.create_buffer<float>(_max_tokens * max_q_dim);
    _k_buf = _device.create_buffer<float>(_max_tokens * max_kv_dim);
    _v_buf = _device.create_buffer<float>(_max_tokens * max_kv_dim);
    _attn_ctx = _device.create_buffer<float>(_max_tokens * max_q_dim);
    _attn_out = _device.create_buffer<float>(_max_tokens * H);
    _attn_norm = _device.create_buffer<float>(_max_tokens * H);
    _ffn_norm = _device.create_buffer<float>(_max_tokens * H);
    _gate_buf = _device.create_buffer<float>(_max_tokens * I);
    _up_buf = _device.create_buffer<float>(_max_tokens * I);
    _mlp_act = _device.create_buffer<float>(_max_tokens * I);
    _mlp_out = _device.create_buffer<float>(_max_tokens * H);
    _mlp_norm = _device.create_buffer<float>(_max_tokens * H);
    _ple_gate_buf = _device.create_buffer<float>(_max_tokens * H);
    _ple_gated = _device.create_buffer<float>(_max_tokens * H);
    _ple_out = _device.create_buffer<float>(_max_tokens * H);
    _ple_norm_buf = _device.create_buffer<float>(_max_tokens * H);
    _pooled = _device.create_buffer<float>(_max_batch * H);
    _proj_out = _device.create_buffer<float>(_max_batch * 768);

    constexpr size_t max_splitk_workspace = 8ull * 64ull * 2048ull;// 4 MB, strictly resident in 6 MB L2
    if (_splitk_workspace.size() < max_splitk_workspace) {
        _splitk_workspace = _device.create_buffer<float>(max_splitk_workspace);
    }
}

void LanguageModelTile::record_encode(
    compute::CommandList &cmd_list,
    compute::BufferView<uint32_t> input_ids,
    compute::BufferView<float> attention_mask,
    compute::BufferView<float> out_embeddings,
    uint32_t B,
    uint32_t S,
    uint32_t target_dim,
    const MultimodalInputs &mm_inputs) {

    ensure_workspace(B, S);

    uint32_t total_tokens = B * S;
    constexpr uint32_t H = 512u;
    constexpr uint32_t I = 2048u;
    constexpr uint32_t num_layers = 24u;
    const float embed_scale = std::sqrt(static_cast<float>(H));

    // 1. Word embedding lookup (BF16 table -> FP32 activations * scale)
    cmd_list << ops_tile::dispatch_tile_embed_lookup(
        _device, _cache, input_ids, _word_embeddings, _h0, total_tokens, 262144u, H, embed_scale);

    // 2. Multimodal soft token scatter
    if (mm_inputs.vision_embeds) {
        if (mm_inputs.vision_token_indices) {
            cmd_list << ops_tile::dispatch_tile_scatter_indexed_tokens(
                _device, _cache, _h0, mm_inputs.vision_embeds, mm_inputs.vision_token_indices,
                total_tokens, mm_inputs.num_vision_tokens, H);
        } else {
            cmd_list << ops_tile::dispatch_tile_scatter_soft_tokens(
                _device, _cache, _h0, mm_inputs.vision_embeds,
                total_tokens, mm_inputs.vision_start_idx, mm_inputs.num_vision_tokens, H);
        }
    }
    if (mm_inputs.audio_embeds) {
        if (mm_inputs.audio_token_indices) {
            cmd_list << ops_tile::dispatch_tile_scatter_indexed_tokens(
                _device, _cache, _h0, mm_inputs.audio_embeds, mm_inputs.audio_token_indices,
                total_tokens, mm_inputs.num_audio_tokens, H);
        } else {
            cmd_list << ops_tile::dispatch_tile_scatter_soft_tokens(
                _device, _cache, _h0, mm_inputs.audio_embeds,
                total_tokens, mm_inputs.audio_start_idx, mm_inputs.num_audio_tokens, H);
        }
    }

    // 3. Global PLE Projection + RMSNorm
    cmd_list << ops_tile::dispatch_tile_gemm(
        _device, _cache, _h0, _ple_proj, _ple_all, total_tokens, num_layers * H, H, _splitk_workspace.view());
    cmd_list << ops_tile::dispatch_tile_rmsnorm(
        _device, _cache, _ple_all, _ple_all, _ple_norm, total_tokens * num_layers, H, 1e-6f);

    // 4. 24 Transformer Layers
    for (uint32_t i = 0; i < 24u; ++i) {
        const auto &layer = _layers[i];
        bool is_full = (i == 5u || i == 11u || i == 17u || i == 23u);
        uint32_t head_dim = is_full ? 512u : 256u;
        uint32_t num_q_heads = 4u;
        uint32_t num_kv_heads = is_full ? 1u : 2u;
        bool is_sliding = !is_full;
        uint32_t sliding_window = 512u;

        uint32_t q_dim = num_q_heads * head_dim;
        uint32_t kv_dim = num_kv_heads * head_dim;

        // (a) Input RMSNorm
        cmd_list << ops_tile::dispatch_tile_rmsnorm(
            _device, _cache, _h0, _h_norm, layer.input_norm, total_tokens, H, 1e-6f);

        // (b) Q, K, V projections
        cmd_list << ops_tile::dispatch_tile_gemm(_device, _cache, _h_norm, layer.q_proj, _q_buf, total_tokens, q_dim, H, _splitk_workspace.view());
        cmd_list << ops_tile::dispatch_tile_gemm(_device, _cache, _h_norm, layer.k_proj, _k_buf, total_tokens, kv_dim, H, _splitk_workspace.view());
        cmd_list << ops_tile::dispatch_tile_gemm(_device, _cache, _h_norm, layer.v_proj, _v_buf, total_tokens, kv_dim, H, _splitk_workspace.view());

        // (c) Q, K, V RMSNorm
        cmd_list << ops_tile::dispatch_tile_rmsnorm(_device, _cache, _q_buf, _q_buf, layer.q_norm, total_tokens * num_q_heads, head_dim, 1e-6f);
        cmd_list << ops_tile::dispatch_tile_rmsnorm(_device, _cache, _k_buf, _k_buf, layer.k_norm, total_tokens * num_kv_heads, head_dim, 1e-6f);
        cmd_list << ops_tile::dispatch_tile_rmsnorm_no_weight(_device, _cache, _v_buf, _v_buf, total_tokens * num_kv_heads, head_dim, 1e-6f);

        // (d) RoPE
        auto &cos_tab = is_full ? _cos_full : _cos_sliding;
        auto &sin_tab = is_full ? _sin_full : _sin_sliding;
        uint64_t table_tag = is_full ? 1ull : 0ull;
        cmd_list << ops_tile::dispatch_tile_rope(_device, _cache, _q_buf, _q_buf, cos_tab, sin_tab, B, S, num_q_heads, head_dim, S, table_tag);
        cmd_list << ops_tile::dispatch_tile_rope(_device, _cache, _k_buf, _k_buf, cos_tab, sin_tab, B, S, num_kv_heads, head_dim, S, table_tag);

        // (e) Fused GQA Attention
        cmd_list << ops_tile::dispatch_tile_gqa_attention(
            _device, _cache, _q_buf, _k_buf, _v_buf, attention_mask, _attn_ctx,
            B, S, num_q_heads, num_kv_heads, head_dim, 1.0f, is_sliding ? 1u : 0u, sliding_window);

        // (f) Out GEMM
        cmd_list << ops_tile::dispatch_tile_gemm(_device, _cache, _attn_ctx, layer.out_proj, _attn_out, total_tokens, H, q_dim, _splitk_workspace.view());

        // (g) Post-attention RMSNorm + Residual Add
        cmd_list << ops_tile::dispatch_tile_rmsnorm(_device, _cache, _attn_out, _attn_norm, layer.post_attn_norm, total_tokens, H, 1e-6f);
        cmd_list << ops_tile::dispatch_tile_residual_add(_device, _cache, _h0, _attn_norm, _h0, total_tokens * H);

        // (h) Pre-FFN RMSNorm
        cmd_list << ops_tile::dispatch_tile_rmsnorm(_device, _cache, _h0, _ffn_norm, layer.pre_ffn_norm, total_tokens, H, 1e-6f);

        // (i) MLP Gate & Up projections
        cmd_list << ops_tile::dispatch_tile_gemm(_device, _cache, _ffn_norm, layer.gate_proj, _gate_buf, total_tokens, I, H, _splitk_workspace.view());
        cmd_list << ops_tile::dispatch_tile_gemm(_device, _cache, _ffn_norm, layer.up_proj, _up_buf, total_tokens, I, H, _splitk_workspace.view());

        // (j) GeGLU activation
        cmd_list << ops_tile::dispatch_tile_geglu(_device, _cache, _gate_buf, _up_buf, _mlp_act, total_tokens * I);

        // (k) MLP Down projection
        cmd_list << ops_tile::dispatch_tile_gemm(_device, _cache, _mlp_act, layer.down_proj, _mlp_out, total_tokens, H, I, _splitk_workspace.view());

        // (l) Post-FFN RMSNorm + Residual Add
        cmd_list << ops_tile::dispatch_tile_rmsnorm(_device, _cache, _mlp_out, _mlp_norm, layer.post_ffn_norm, total_tokens, H, 1e-6f);
        cmd_list << ops_tile::dispatch_tile_residual_add(_device, _cache, _h0, _mlp_norm, _h0, total_tokens * H);

        // (m) PLE Block
        cmd_list << ops_tile::dispatch_tile_gemm(_device, _cache, _h0, layer.ple_gate, _ple_gate_buf, total_tokens, H, H, _splitk_workspace.view());
        cmd_list << ops_tile::dispatch_tile_ple_gate(_device, _cache, _ple_gate_buf, _ple_all, _ple_gated, i, total_tokens, num_layers, H);
        cmd_list << ops_tile::dispatch_tile_gemm(_device, _cache, _ple_gated, layer.ple_proj, _ple_out, total_tokens, H, H, _splitk_workspace.view());
        cmd_list << ops_tile::dispatch_tile_rmsnorm(_device, _cache, _ple_out, _ple_norm_buf, layer.ple_norm, total_tokens, H, 1e-6f);

        // (n) Residual Add + Layer Scalar
        cmd_list << ops_tile::dispatch_tile_scale_add(_device, _cache, _h0, _ple_norm_buf, _h0, layer.layer_scalar, total_tokens * H);
    }

    // 5. Final RMSNorm
    cmd_list << ops_tile::dispatch_tile_rmsnorm(_device, _cache, _h0, _h_norm, _final_norm, total_tokens, H, 1e-6f);

    // 6. Mean Pooling
    cmd_list << ops_tile::dispatch_tile_mean_pool(_device, _cache, _h_norm, attention_mask, _pooled, B, S, H);

    // 7. Dense Projection (512 -> 768)
    cmd_list << ops_tile::dispatch_tile_gemm(_device, _cache, _pooled, _embedding_proj, _proj_out, B, 768u, H, _splitk_workspace.view());

    // 8. MRL Truncation + L2 Normalization
    cmd_list << ops_tile::dispatch_tile_mrl_normalize(_device, _cache, _proj_out, out_embeddings, B, 768u, target_dim);
}

} // namespace luisa::embed_mm
