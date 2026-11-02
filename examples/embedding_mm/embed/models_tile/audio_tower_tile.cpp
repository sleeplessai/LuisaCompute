// embedding_mm engine -- Gemma4AudioModel + embed_audio connector (Tile backend).
#include "embedding_mm/embed/models_tile/audio_tower_tile.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>

namespace luisa::embed_mm {

void AudioTowerTile::load_weights(
    const Package &package,
    compute::CommandList &upload_cmds,
    luisa::vector<luisa::vector<float>> &fp32_storage,
    luisa::vector<luisa::vector<luisa::bfloat16>> &bf16_storage) {

    LUISA_INFO("AudioTowerTile: uploading weight tensors (BF16 Tensor Core MMA)...");

    _conv0_weight = upload_tensor_fp32(_device, package, "audio.subsample.layer0.conv", upload_cmds, fp32_storage);
    _conv0_norm = upload_tensor_fp32(_device, package, "audio.subsample.layer0.norm", upload_cmds, fp32_storage);
    _conv1_weight = upload_tensor_fp32(_device, package, "audio.subsample.layer1.conv", upload_cmds, fp32_storage);
    _conv1_norm = upload_tensor_fp32(_device, package, "audio.subsample.layer1.norm", upload_cmds, fp32_storage);
    _subsample_proj = upload_tensor_bf16(_device, package, "audio.subsample.input_proj", upload_cmds, bf16_storage);

    // Precompute relative position embedding table [13, 1024]
    constexpr uint32_t pos_len = 13u;
    constexpr uint32_t hidden_size = 1024u;
    constexpr uint32_t half_dim = hidden_size / 2u;
    auto &pos_table = fp32_storage.emplace_back(pos_len * hidden_size);
    float log_inc = std::log(10000.0f) / static_cast<float>(half_dim - 1u);
    for (uint32_t p = 0u; p < pos_len; ++p) {
        float pos_id = static_cast<float>(pos_len - 1u - p);
        for (uint32_t d = 0u; d < half_dim; ++d) {
            float inv_ts = std::exp(-static_cast<float>(d) * log_inc);
            float scaled = pos_id * inv_ts;
            pos_table[p * hidden_size + d] = std::sin(scaled);
            pos_table[p * hidden_size + half_dim + d] = std::cos(scaled);
        }
    }
    _pos_embed = _device.create_buffer<float>(pos_len * hidden_size);
    upload_cmds << _pos_embed.copy_from(pos_table.data());

    _output_proj_weight = upload_tensor_bf16(_device, package, "audio.output_proj.weight", upload_cmds, bf16_storage);
    _output_proj_bias = upload_tensor_fp32(_device, package, "audio.output_proj.bias", upload_cmds, fp32_storage);
    _connector_proj = upload_tensor_bf16(_device, package, "audio.connector.proj", upload_cmds, bf16_storage);

    const auto *t_clips = package.find("audio.clip_bounds");
    const float *clip_src = nullptr;
    if (t_clips && t_clips->data) {
        clip_src = t_clips->data;
    } else if (t_clips && t_clips->half_data) {
        auto &unpacked = fp32_storage.emplace_back(t_clips->element_count());
        for (size_t k = 0; k < unpacked.size(); ++k) {
            uint16_t h = t_clips->half_data[k];
            if (package.precision() == PackagePrecision::bf16) {
                uint32_t w = static_cast<uint32_t>(h) << 16;
                std::memcpy(&unpacked[k], &w, sizeof(float));
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
                std::memcpy(&unpacked[k], &f_val, sizeof(float));
            }
        }
        clip_src = unpacked.data();
    }

    _layers.clear();
    _layers.reserve(12u);
    for (uint32_t i = 0u; i < 12u; ++i) {
        AudioLayerWeightsTile al;
        al.ffn1_pre_norm = upload_tensor_fp32(_device, package, luisa::format("audio.layer{}.ffn1_pre_norm", i), upload_cmds, fp32_storage);
        al.ffn1_linear1 = upload_tensor_bf16(_device, package, luisa::format("audio.layer{}.ffn1_linear1", i), upload_cmds, bf16_storage);
        al.ffn1_linear2 = upload_tensor_bf16(_device, package, luisa::format("audio.layer{}.ffn1_linear2", i), upload_cmds, bf16_storage);
        al.ffn1_post_norm = upload_tensor_fp32(_device, package, luisa::format("audio.layer{}.ffn1_post_norm", i), upload_cmds, fp32_storage);

        al.norm_pre_attn = upload_tensor_fp32(_device, package, luisa::format("audio.layer{}.norm_pre_attn", i), upload_cmds, fp32_storage);
        al.q_proj = upload_tensor_bf16(_device, package, luisa::format("audio.layer{}.q_proj", i), upload_cmds, bf16_storage);
        al.k_proj = upload_tensor_bf16(_device, package, luisa::format("audio.layer{}.k_proj", i), upload_cmds, bf16_storage);
        al.v_proj = upload_tensor_bf16(_device, package, luisa::format("audio.layer{}.v_proj", i), upload_cmds, bf16_storage);
        al.per_dim_scale = upload_tensor_fp32(_device, package, luisa::format("audio.layer{}.per_dim_scale", i), upload_cmds, fp32_storage);
        al.relative_k_proj = upload_tensor_bf16(_device, package, luisa::format("audio.layer{}.relative_k_proj", i), upload_cmds, bf16_storage);
        al.out_proj = upload_tensor_bf16(_device, package, luisa::format("audio.layer{}.out_proj", i), upload_cmds, bf16_storage);
        al.norm_post_attn = upload_tensor_fp32(_device, package, luisa::format("audio.layer{}.norm_post_attn", i), upload_cmds, fp32_storage);

        al.lconv_pre_norm = upload_tensor_fp32(_device, package, luisa::format("audio.layer{}.lconv_pre_norm", i), upload_cmds, fp32_storage);
        al.lconv_linear_start = upload_tensor_bf16(_device, package, luisa::format("audio.layer{}.lconv_linear_start", i), upload_cmds, bf16_storage);
        al.lconv_depthwise = upload_tensor_fp32(_device, package, luisa::format("audio.layer{}.lconv_depthwise", i), upload_cmds, fp32_storage);
        al.lconv_norm = upload_tensor_fp32(_device, package, luisa::format("audio.layer{}.lconv_norm", i), upload_cmds, fp32_storage);
        al.lconv_linear_end = upload_tensor_bf16(_device, package, luisa::format("audio.layer{}.lconv_linear_end", i), upload_cmds, bf16_storage);

        al.ffn2_pre_norm = upload_tensor_fp32(_device, package, luisa::format("audio.layer{}.ffn2_pre_norm", i), upload_cmds, fp32_storage);
        al.ffn2_linear1 = upload_tensor_bf16(_device, package, luisa::format("audio.layer{}.ffn2_linear1", i), upload_cmds, bf16_storage);
        al.ffn2_linear2 = upload_tensor_bf16(_device, package, luisa::format("audio.layer{}.ffn2_linear2", i), upload_cmds, bf16_storage);
        al.ffn2_post_norm = upload_tensor_fp32(_device, package, luisa::format("audio.layer{}.ffn2_post_norm", i), upload_cmds, fp32_storage);

        al.norm_out = upload_tensor_fp32(_device, package, luisa::format("audio.layer{}.norm_out", i), upload_cmds, fp32_storage);

        if (clip_src) {
            std::memcpy(al.clip_bounds, clip_src + i * 40, sizeof(al.clip_bounds));
        } else {
            for (int j = 0; j < 10; ++j) {
                al.clip_bounds[j][0] = -1e9f;
                al.clip_bounds[j][1] = 1e9f;
                al.clip_bounds[j][2] = -1e9f;
                al.clip_bounds[j][3] = 1e9f;
            }
        }
        _layers.push_back(std::move(al));
    }
}

void AudioTowerTile::ensure_workspace(uint32_t num_frames, uint32_t num_soft_tokens) {
    if (num_frames <= _max_frames && num_soft_tokens <= _max_soft_tokens) {
        return;
    }

    _max_frames = std::max(_max_frames, static_cast<size_t>(num_frames));
    _max_soft_tokens = std::max(_max_soft_tokens, static_cast<size_t>(num_soft_tokens));

    size_t H0 = (_max_frames + 1) / 2;
    size_t W0 = 64;
    _conv0_out = _device.create_buffer<float>(128 * H0 * W0);
    _conv1_flat = _device.create_buffer<float>(_max_soft_tokens * 1024);
    _hidden = _device.create_buffer<float>(_max_soft_tokens * 1024);
    _tmp1 = _device.create_buffer<float>(_max_soft_tokens * 1024);
    _tmp2 = _device.create_buffer<float>(_max_soft_tokens * 1024);
    _ffn_mid = _device.create_buffer<float>(_max_soft_tokens * 4096);
    _q = _device.create_buffer<float>(_max_soft_tokens * 1024);
    _k = _device.create_buffer<float>(_max_soft_tokens * 1024);
    _v = _device.create_buffer<float>(_max_soft_tokens * 1024);
    _rel_k = _device.create_buffer<float>(13 * 1024);
    _attn_out = _device.create_buffer<float>(_max_soft_tokens * 1024);
    _glu_in = _device.create_buffer<float>(_max_soft_tokens * 2048);
    _glu_out = _device.create_buffer<float>(_max_soft_tokens * 1024);
    _tower_out = _device.create_buffer<float>(_max_soft_tokens * 1536);
    _conn_in = _device.create_buffer<float>(_max_soft_tokens * 1536);
}

void AudioTowerTile::record_encode(
    compute::CommandList &cmd_list,
    compute::BufferView<float> input_features,
    compute::BufferView<float> out_soft_tokens,
    uint32_t num_frames,
    uint32_t num_mel_bins,
    uint32_t num_soft_tokens,
    Executor::AudioDebugActivations *debug_acts) {

    ensure_workspace(num_frames, num_soft_tokens);

    constexpr uint32_t H = 1024u;
    uint32_t H0 = (num_frames + 1u) / 2u;
    uint32_t W0 = num_mel_bins / 2u;
    uint32_t H1 = num_soft_tokens;
    uint32_t W1 = W0 / 2u;

    // 1. Subsample Conv0: [num_frames, 128] -> [128, H0, W0]
    cmd_list << ops_tile::dispatch_tile_audio_subsample_conv0(
        _device, _cache, input_features, _conv0_weight, _conv0_norm, _conv0_out,
        num_frames, num_mel_bins, H0, W0, 1e-6f);

    // 2. Subsample Conv1: [128, H0, W0] -> [H1, W1, 32] flattened to [H1, 1024]
    cmd_list << ops_tile::dispatch_tile_audio_subsample_conv1(
        _device, _cache, _conv0_out, _conv1_weight, _conv1_norm, _conv1_flat,
        H0, W0, H1, W1, 1e-6f);

    // 3. Subsample Input Proj GEMM: [H1, 1024] @ [1024, 1024] -> [H1, 1024]
    cmd_list << ops_tile::dispatch_tile_gemm(
        _device, _cache, _conv1_flat, _subsample_proj, _hidden, H1, H, H);

    if (debug_acts && debug_acts->subsample_proj) {
        cmd_list << debug_acts->subsample_proj.copy_from(_hidden.view(0, H1 * H));
    }

    // 4. 12 Conformer layers
    for (uint32_t l = 0u; l < 12u; ++l) {
        const auto &layer = _layers[l];

        // --- FeedForward 1 ---
        cmd_list << ops_tile::dispatch_tile_rmsnorm(
            _device, _cache, _hidden, _tmp1, layer.ffn1_pre_norm, H1, H, 1e-6f);
        cmd_list << ops_tile::dispatch_tile_audio_clamp(
            _device, _cache, _tmp1, _tmp1, layer.clip_bounds[0][0], layer.clip_bounds[0][1], H1 * H);
        cmd_list << ops_tile::dispatch_tile_gemm(
            _device, _cache, _tmp1, layer.ffn1_linear1, _ffn_mid, H1, 4096u, H);
        cmd_list << ops_tile::dispatch_tile_audio_clamp(
            _device, _cache, _ffn_mid, _ffn_mid, layer.clip_bounds[0][2], layer.clip_bounds[0][3], H1 * 4096u);
        cmd_list << ops_tile::dispatch_tile_audio_silu(
            _device, _cache, _ffn_mid, _ffn_mid, H1 * 4096u);
        cmd_list << ops_tile::dispatch_tile_audio_clamp(
            _device, _cache, _ffn_mid, _ffn_mid, layer.clip_bounds[1][0], layer.clip_bounds[1][1], H1 * 4096u);
        cmd_list << ops_tile::dispatch_tile_gemm(
            _device, _cache, _ffn_mid, layer.ffn1_linear2, _tmp1, H1, H, 4096u);
        cmd_list << ops_tile::dispatch_tile_audio_clamp(
            _device, _cache, _tmp1, _tmp1, layer.clip_bounds[1][2], layer.clip_bounds[1][3], H1 * H);
        cmd_list << ops_tile::dispatch_tile_rmsnorm(
            _device, _cache, _tmp1, _tmp2, layer.ffn1_post_norm, H1, H, 1e-6f);
        cmd_list << ops_tile::dispatch_tile_audio_add_scale(
            _device, _cache, _tmp2, _hidden, _hidden, 0.5f, H1 * H);

        // --- Self Attention ---
        cmd_list << ops_tile::dispatch_tile_rmsnorm(
            _device, _cache, _hidden, _tmp1, layer.norm_pre_attn, H1, H, 1e-6f);
        cmd_list << ops_tile::dispatch_tile_audio_clamp(
            _device, _cache, _tmp1, _tmp1, layer.clip_bounds[2][0], layer.clip_bounds[2][1], H1 * H);
        cmd_list << ops_tile::dispatch_tile_gemm(
            _device, _cache, _tmp1, layer.q_proj, _q, H1, H, H);
        cmd_list << ops_tile::dispatch_tile_gemm(
            _device, _cache, _tmp1, layer.k_proj, _k, H1, H, H);
        cmd_list << ops_tile::dispatch_tile_gemm(
            _device, _cache, _tmp1, layer.v_proj, _v, H1, H, H);
        cmd_list << ops_tile::dispatch_tile_audio_clamp(
            _device, _cache, _q, _q, layer.clip_bounds[2][2], layer.clip_bounds[2][3], H1 * H);
        cmd_list << ops_tile::dispatch_tile_audio_clamp(
            _device, _cache, _k, _k, layer.clip_bounds[3][2], layer.clip_bounds[3][3], H1 * H);
        cmd_list << ops_tile::dispatch_tile_audio_clamp(
            _device, _cache, _v, _v, layer.clip_bounds[4][2], layer.clip_bounds[4][3], H1 * H);

        float q_scale = (std::pow(128.0f, -0.5f)) / std::log(2.0f);
        float k_scale = std::log(1.0f + std::exp(1.0f)) / std::log(2.0f);
        cmd_list << ops_tile::dispatch_tile_audio_scale_qk(
            _device, _cache, _q, _k, layer.per_dim_scale, H1, 8u, 128u, q_scale, k_scale);
        cmd_list << ops_tile::dispatch_tile_gemm(
            _device, _cache, _pos_embed, layer.relative_k_proj, _rel_k, 13u, H, H);
        cmd_list << ops_tile::dispatch_tile_audio_chunked_attention(
            _device, _cache, _q, _k, _v, _rel_k, _attn_out, H1, 8u, 128u, 12u, 50.0f);

        cmd_list << ops_tile::dispatch_tile_audio_clamp(
            _device, _cache, _attn_out, _attn_out, layer.clip_bounds[5][0], layer.clip_bounds[5][1], H1 * H);
        cmd_list << ops_tile::dispatch_tile_gemm(
            _device, _cache, _attn_out, layer.out_proj, _tmp1, H1, H, H);
        cmd_list << ops_tile::dispatch_tile_audio_clamp(
            _device, _cache, _tmp1, _tmp1, layer.clip_bounds[5][2], layer.clip_bounds[5][3], H1 * H);
        cmd_list << ops_tile::dispatch_tile_rmsnorm(
            _device, _cache, _tmp1, _tmp2, layer.norm_post_attn, H1, H, 1e-6f);
        cmd_list << ops_tile::dispatch_tile_residual_add(
            _device, _cache, _tmp2, _hidden, _hidden, H1 * H);

        // --- LightConv1D ---
        cmd_list << ops_tile::dispatch_tile_rmsnorm(
            _device, _cache, _hidden, _tmp1, layer.lconv_pre_norm, H1, H, 1e-6f);
        cmd_list << ops_tile::dispatch_tile_audio_clamp(
            _device, _cache, _tmp1, _tmp1, layer.clip_bounds[6][0], layer.clip_bounds[6][1], H1 * H);
        cmd_list << ops_tile::dispatch_tile_gemm(
            _device, _cache, _tmp1, layer.lconv_linear_start, _glu_in, H1, 2048u, H);
        cmd_list << ops_tile::dispatch_tile_audio_clamp(
            _device, _cache, _glu_in, _glu_in, layer.clip_bounds[6][2], layer.clip_bounds[6][3], H1 * 2048u);
        cmd_list << ops_tile::dispatch_tile_audio_glu(
            _device, _cache, _glu_in, _glu_out, H1, H);
        cmd_list << ops_tile::dispatch_tile_audio_depthwise_conv1d(
            _device, _cache, _glu_out, layer.lconv_depthwise, _tmp1, H1, H);
        cmd_list << ops_tile::dispatch_tile_rmsnorm(
            _device, _cache, _tmp1, _tmp2, layer.lconv_norm, H1, H, 1e-6f);
        cmd_list << ops_tile::dispatch_tile_audio_silu(
            _device, _cache, _tmp2, _tmp1, H1 * H);
        cmd_list << ops_tile::dispatch_tile_audio_clamp(
            _device, _cache, _tmp1, _tmp1, layer.clip_bounds[7][0], layer.clip_bounds[7][1], H1 * H);
        cmd_list << ops_tile::dispatch_tile_gemm(
            _device, _cache, _tmp1, layer.lconv_linear_end, _tmp2, H1, H, H);
        cmd_list << ops_tile::dispatch_tile_audio_clamp(
            _device, _cache, _tmp2, _tmp2, layer.clip_bounds[7][2], layer.clip_bounds[7][3], H1 * H);
        cmd_list << ops_tile::dispatch_tile_residual_add(
            _device, _cache, _tmp2, _hidden, _hidden, H1 * H);

        // --- FeedForward 2 ---
        cmd_list << ops_tile::dispatch_tile_rmsnorm(
            _device, _cache, _hidden, _tmp1, layer.ffn2_pre_norm, H1, H, 1e-6f);
        cmd_list << ops_tile::dispatch_tile_audio_clamp(
            _device, _cache, _tmp1, _tmp1, layer.clip_bounds[8][0], layer.clip_bounds[8][1], H1 * H);
        cmd_list << ops_tile::dispatch_tile_gemm(
            _device, _cache, _tmp1, layer.ffn2_linear1, _ffn_mid, H1, 4096u, H);
        cmd_list << ops_tile::dispatch_tile_audio_clamp(
            _device, _cache, _ffn_mid, _ffn_mid, layer.clip_bounds[8][2], layer.clip_bounds[8][3], H1 * 4096u);
        cmd_list << ops_tile::dispatch_tile_audio_silu(
            _device, _cache, _ffn_mid, _ffn_mid, H1 * 4096u);
        cmd_list << ops_tile::dispatch_tile_audio_clamp(
            _device, _cache, _ffn_mid, _ffn_mid, layer.clip_bounds[9][0], layer.clip_bounds[9][1], H1 * 4096u);
        cmd_list << ops_tile::dispatch_tile_gemm(
            _device, _cache, _ffn_mid, layer.ffn2_linear2, _tmp1, H1, H, 4096u);
        cmd_list << ops_tile::dispatch_tile_audio_clamp(
            _device, _cache, _tmp1, _tmp1, layer.clip_bounds[9][2], layer.clip_bounds[9][3], H1 * H);
        cmd_list << ops_tile::dispatch_tile_rmsnorm(
            _device, _cache, _tmp1, _tmp2, layer.ffn2_post_norm, H1, H, 1e-6f);
        cmd_list << ops_tile::dispatch_tile_audio_add_scale(
            _device, _cache, _tmp2, _hidden, _hidden, 0.5f, H1 * H);

        // --- Norm Out ---
        cmd_list << ops_tile::dispatch_tile_rmsnorm(
            _device, _cache, _hidden, _hidden, layer.norm_out, H1, H, 1e-6f);

        if (l == 0u && debug_acts && debug_acts->layer0_out) {
            cmd_list << debug_acts->layer0_out.copy_from(_hidden.view(0, H1 * H));
        }
        if (l == 11u && debug_acts && debug_acts->layer11_out) {
            cmd_list << debug_acts->layer11_out.copy_from(_hidden.view(0, H1 * H));
        }
    }

    // 5. Output Projection: GEMM with Bias [H1, 1024] @ [1024, 1536] + bias -> [H1, 1536]
    cmd_list << ops_tile::dispatch_tile_gemm_bias(
        _device, _cache, _hidden, _output_proj_weight, _output_proj_bias, _tower_out, H1, 1536u, H);

    if (debug_acts && debug_acts->audio_tower_out) {
        cmd_list << debug_acts->audio_tower_out.copy_from(_tower_out.view(0, H1 * 1536u));
    }

    // 6. Connector: RMSNorm without weight [H1, 1536]
    cmd_list << ops_tile::dispatch_tile_rmsnorm_no_weight(
        _device, _cache, _tower_out, _conn_in, H1, 1536u, 1e-6f);

    // 7. Connector Projection: GEMM [H1, 1536] @ [1536, 512] -> out_soft_tokens [H1, 512]
    cmd_list << ops_tile::dispatch_tile_gemm(
        _device, _cache, _conn_in, _connector_proj, out_soft_tokens, H1, 512u, 1536u);
}

} // namespace luisa::embed_mm
