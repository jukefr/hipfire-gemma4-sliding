//! Gemma 4 model: hybrid sliding-window + full attention, dense FFN (SwiGLU + gelu_pytorch_tanh).
//!
//! Architectural features vs. Qwen3.5:
//!   • Sliding-window attention on 5 of every 6 layers (window=1024).
//!   • Full attention layers use head_dim=512 (global_head_dim) with
//!     attention_k_eq_v: V is the pre-k_norm output of k_proj (no v_proj).
//!   • Partial proportional RoPE on full layers (first 64 of 512 dims rotate,
//!     rope_theta=1e6; sliding uses default RoPE with theta=10000).
//!   • Sandwich RMSNorm: input + post-attn + pre-FFN + post-FFN per layer,
//!     plus a learned per-layer `layer_scalar [1]` at layer end.
//!   • Attention scale = 1.0 (not 1/√d); Q/K norms absorb scaling.
//!   • Final logit softcap: `tanh(logits/30) * 30` before sampling.
//!   • MLP: SwiGLU with `gelu_pytorch_tanh` activation.
//!   • Tied LM head (embed_tokens.weight aliased).
//!   • Embed scale: sqrt(hidden_size) multiplied onto every embedding row lookup.

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{self, f16_to_f32, weight_gemv, WeightTensor, EmbeddingFormat};
use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};

// ─── Config ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    /// Sliding-window causal attention (window=1024 on 31B).
    Sliding,
    /// Full causal attention (global).
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeType {
    /// Standard RoPE: all head_dim positions rotate.
    Default,
    /// Proportional RoPE (Gemma 4 full layers): only the first
    /// `partial_rotary_factor × head_dim` positions rotate; rest are NoPE.
    Proportional,
}

#[derive(Debug, Clone)]
pub struct Gemma4Config {
    // Common
    pub dim: usize,                        // hidden_size, e.g. 5376 on 31B
    pub n_layers: usize,                   // 60 on 31B
    pub vocab_size: usize,                 // 262144 on Gemma 4
    pub norm_eps: f32,                     // 1e-6
    pub bos_token: u32,                    // 2
    pub eos_token: u32,                    // 1
    pub pad_token: u32,                    // 0

    // Attention heads (same count for sliding + full)
    pub n_heads: usize,                    // 32 on 31B

    // Sliding-window attention
    pub sliding_head_dim: usize,           // 256 on 31B
    pub sliding_n_kv_heads: usize,         // 16 on 31B
    pub sliding_rope_theta: f32,           // 10000.0
    pub sliding_window: usize,             // 1024

    // Full attention (global)
    pub full_head_dim: usize,              // 512 on 31B (= global_head_dim)
    pub full_n_kv_heads: usize,            // 4 on 31B
    pub full_rope_theta: f32,              // 1_000_000.0
    pub full_rope_type: RopeType,          // Proportional on 31B
    pub full_partial_rotary_factor: f32,   // 0.25
    pub attention_k_eq_v: bool,            // true on 31B — V = pre-k_norm output

    // FFN (SwiGLU, gelu_pytorch_tanh)
    pub hidden_dim: usize,                 // intermediate_size = 21504 on 31B

    // MoE (26B-A4B). enable_moe_block=true → every layer carries a parallel
    // MoE branch whose output sums with the standard SwiGLU output before
    // the post_feedforward_layernorm. Zero on dense models (31B).
    pub enable_moe_block: bool,            // true on 26B-A4B
    pub moe_intermediate_size: usize,      // 704 on 26B-A4B (per-expert FFN hidden)
    pub num_experts: usize,                // 128 on 26B-A4B
    pub top_k_experts: usize,              // 8 on 26B-A4B (kernel hardcoded to 8)

    // Output
    pub final_logit_softcapping: f32,      // 30.0 — tanh(x/30)*30
    pub tie_word_embeddings: bool,         // true — lm_head aliases embed_tokens
    pub embed_scale: f32,                  // sqrt(dim), applied at embed lookup

    // Per-layer dispatch (len == n_layers)
    pub layer_types: Vec<LayerType>,

    // Vision integration (present even on text-only 31B since config ships it)
    pub has_vision: bool,
    pub image_token_id: u32,               // 258880
    pub boi_token_id: u32,                 // 255999
    pub eoi_token_id: u32,                 // 258882
    pub audio_token_id: u32,               // 258881 (reserved, unused on dense 31B)
    pub video_token_id: u32,               // 258884 (reserved)
}

pub fn config_from_hfq(hfq: &HfqFile) -> Option<Gemma4Config> {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).ok()?;
    let config = meta.get("config")?;
    let tc = config.get("text_config").unwrap_or(config);

    let dim = tc.get("hidden_size")?.as_u64()? as usize;
    let n_layers = tc.get("num_hidden_layers")?.as_u64()? as usize;
    let vocab_size = tc.get("vocab_size")?.as_u64()? as usize;
    let norm_eps = tc.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32;
    let bos_token = tc.get("bos_token_id").and_then(|v| v.as_u64()).unwrap_or(2) as u32;
    let eos_token = tc.get("eos_token_id").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
    let pad_token = tc.get("pad_token_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

    let n_heads = tc.get("num_attention_heads")?.as_u64()? as usize;

    // Sliding attention params
    let sliding_head_dim = tc.get("head_dim").and_then(|v| v.as_u64()).map(|v| v as usize)
        .unwrap_or(dim / n_heads);
    let sliding_n_kv_heads = tc.get("num_key_value_heads").and_then(|v| v.as_u64())
        .unwrap_or(n_heads as u64) as usize;
    let sliding_window = tc.get("sliding_window").and_then(|v| v.as_u64()).unwrap_or(1024) as usize;

    // Full attention params (may differ from sliding)
    let full_head_dim = tc.get("global_head_dim").and_then(|v| v.as_u64()).map(|v| v as usize)
        .unwrap_or(sliding_head_dim);
    let full_n_kv_heads = tc.get("num_global_key_value_heads").and_then(|v| v.as_u64())
        .unwrap_or(sliding_n_kv_heads as u64) as usize;
    let attention_k_eq_v = tc.get("attention_k_eq_v").and_then(|v| v.as_bool()).unwrap_or(false);

    // rope_parameters is a dict with "sliding_attention" and "full_attention" sub-dicts
    // per the Gemma 4 config schema. Parse both independently.
    let rope_params = tc.get("rope_parameters");
    let sliding_rope = rope_params.and_then(|r| r.get("sliding_attention"));
    let full_rope = rope_params.and_then(|r| r.get("full_attention"));

    let sliding_rope_theta = sliding_rope.and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64()).unwrap_or(10_000.0) as f32;
    let full_rope_theta = full_rope.and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64()).unwrap_or(1_000_000.0) as f32;
    let full_rope_type = match full_rope.and_then(|r| r.get("rope_type")).and_then(|v| v.as_str()) {
        Some("proportional") => RopeType::Proportional,
        _ => RopeType::Default,
    };
    let full_partial_rotary_factor = full_rope.and_then(|r| r.get("partial_rotary_factor"))
        .and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;

    let hidden_dim = tc.get("intermediate_size")?.as_u64()? as usize;

    // MoE config (26B-A4B). Absent / false on dense models (31B).
    let enable_moe_block = tc.get("enable_moe_block").and_then(|v| v.as_bool()).unwrap_or(false);
    let moe_intermediate_size = tc.get("moe_intermediate_size")
        .and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let num_experts = tc.get("num_experts").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let top_k_experts = tc.get("top_k_experts").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

    let final_logit_softcapping = tc.get("final_logit_softcapping")
        .and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
    let tie_word_embeddings = tc.get("tie_word_embeddings").and_then(|v| v.as_bool())
        .or_else(|| config.get("tie_word_embeddings").and_then(|v| v.as_bool()))
        .unwrap_or(true);

    let embed_scale = (dim as f32).sqrt();

    let layer_types: Vec<LayerType> = tc.get("layer_types")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().map(|v| match v.as_str().unwrap_or("sliding_attention") {
            "full_attention" => LayerType::Full,
            _ => LayerType::Sliding,
        }).collect())
        .unwrap_or_else(|| vec![LayerType::Sliding; n_layers]);

    // Multimodal token IDs (top-level in config, not under text_config)
    let has_vision = config.get("vision_config").map(|v| !v.is_null()).unwrap_or(false);
    let image_token_id = config.get("image_token_id").and_then(|v| v.as_u64()).unwrap_or(258880) as u32;
    let boi_token_id = config.get("boi_token_id").and_then(|v| v.as_u64()).unwrap_or(255999) as u32;
    let eoi_token_id = config.get("eoi_token_id").and_then(|v| v.as_u64()).unwrap_or(258882) as u32;
    let audio_token_id = config.get("audio_token_id").and_then(|v| v.as_u64()).unwrap_or(258881) as u32;
    let video_token_id = config.get("video_token_id").and_then(|v| v.as_u64()).unwrap_or(258884) as u32;

    Some(Gemma4Config {
        dim, n_layers, vocab_size, norm_eps,
        bos_token, eos_token, pad_token,
        n_heads,
        sliding_head_dim, sliding_n_kv_heads, sliding_rope_theta, sliding_window,
        full_head_dim, full_n_kv_heads, full_rope_theta, full_rope_type,
        full_partial_rotary_factor, attention_k_eq_v,
        hidden_dim,
        enable_moe_block, moe_intermediate_size, num_experts, top_k_experts,
        final_logit_softcapping, tie_word_embeddings, embed_scale,
        layer_types,
        has_vision,
        image_token_id, boi_token_id, eoi_token_id, audio_token_id, video_token_id,
    })
}

// ─── Weights ────────────────────────────────────────────────────────────

/// Per-layer weights for a SLIDING layer (head_dim=256, 16 KV heads, full RoPE).
pub struct SlidingLayerWeights {
    pub input_layernorm: GpuTensor,           // [dim]
    pub post_attention_layernorm: GpuTensor,  // [dim]
    pub pre_feedforward_layernorm: GpuTensor, // [dim]
    pub post_feedforward_layernorm: GpuTensor,// [dim]
    pub layer_scalar: GpuTensor,              // [1]
    /// Host-side mirror of layer_scalar. Populated at load time so decode can
    /// call `gpu.scale_f32(x, layer_scalar_host)` without a D2H round-trip.
    pub layer_scalar_host: f32,

    // Attention (sliding — head_dim=256)
    pub q_proj: WeightTensor,   // [n_heads * 256, dim]
    pub k_proj: WeightTensor,   // [16 * 256, dim]
    pub v_proj: WeightTensor,   // [16 * 256, dim]
    pub o_proj: WeightTensor,   // [dim, n_heads * 256]
    pub q_norm: GpuTensor,      // [256]
    pub k_norm: GpuTensor,      // [256]

    // MLP (SwiGLU)
    pub gate_proj: WeightTensor, // [hidden_dim, dim]
    pub up_proj: WeightTensor,   // [hidden_dim, dim]
    pub down_proj: WeightTensor, // [dim, hidden_dim]

    // MoE branch — Some on 26B-A4B (every layer is MoE), None on dense models.
    pub moe: Option<MoeLayerExtras>,
}

/// Per-layer weights for a FULL layer (head_dim=512, 4 KV heads, K=V shared).
///
/// Note: no `v_proj` — V is the pre-k_norm output of k_proj, renormed by
/// weight-less `v_norm`. No `v_norm` tensor either (no_scale — the `with_scale=False`
/// RMSNorm applies only the divide, no learned gain). We reuse the existing
/// rmsnorm kernel with a ones-filled `v_norm_ones` buffer (shared across
/// full-attn layers) to preserve the no-scale semantics.
pub struct FullLayerWeights {
    pub input_layernorm: GpuTensor,
    pub post_attention_layernorm: GpuTensor,
    pub pre_feedforward_layernorm: GpuTensor,
    pub post_feedforward_layernorm: GpuTensor,
    pub layer_scalar: GpuTensor,
    /// Host-side mirror of layer_scalar. See SlidingLayerWeights for rationale.
    pub layer_scalar_host: f32,

    // Attention (full — head_dim=512, K=V)
    pub q_proj: WeightTensor,   // [n_heads * 512, dim]
    pub k_proj: WeightTensor,   // [4 * 512, dim]
    // no v_proj — V = pre-k_norm output of k_proj
    pub o_proj: WeightTensor,   // [dim, n_heads * 512]
    pub q_norm: GpuTensor,      // [512]
    pub k_norm: GpuTensor,      // [512]
    // no v_norm weight — v_norm is no-scale (divide only)

    // MLP (SwiGLU, same shape as sliding)
    pub gate_proj: WeightTensor,
    pub up_proj: WeightTensor,
    pub down_proj: WeightTensor,

    // MoE branch — Some on 26B-A4B (every layer is MoE), None on dense models.
    pub moe: Option<MoeLayerExtras>,
}

/// Per-expert FFN weights for a single MoE expert. 128 of these per layer
/// on 26B-A4B; views into the per-layer pool allocation (so `free_gpu`
/// doesn't free these — the pool owns the bytes).
pub struct MoeExpertWeights {
    /// `[2 * moe_intermediate, dim]` — gate + up fused. Rows [0, mi) are
    /// gate; rows [mi, 2*mi) are up. Quantized as MQ4G256 / MG4G256 when
    /// dim is 256-aligned (it is on 26B-A4B: dim=2816).
    pub gate_up_proj: WeightTensor,
    /// `[dim, moe_intermediate]` — projects per-expert FFN hidden back to dim.
    /// On 26B-A4B, mi=704 isn't 256-aligned so this drops to Q8_0 via the
    /// quantizer fallback chain.
    pub down_proj: WeightTensor,
}

/// MoE branch weights for a Gemma 4 MoE layer (26B-A4B). Present on every
/// layer when `config.enable_moe_block` is set. The branch adds a parallel
/// FFN computation alongside the standard SwiGLU; outputs are summed via
/// sandwich norms then a final post_feedforward_layernorm closes the layer.
pub struct MoeLayerExtras {
    /// `[n_experts, dim]` — projects router input to expert logits.
    pub router_proj: WeightTensor,
    /// `[dim]` — multiplicative scale on router input (`router.scale` in HF).
    pub router_scale: GpuTensor,
    /// `[n_experts]` — per-expert post-`down_proj` scale (`router.per_expert_scale`).
    pub per_expert_scale: GpuTensor,
    /// Host mirror of `per_expert_scale` for fast top-K weight composition.
    pub per_expert_scale_host: Vec<f32>,
    /// `[dim]` — RMSNorm applied to attn_out before the MoE branch.
    pub pre_feedforward_layernorm_2: GpuTensor,
    /// `[dim]` — RMSNorm applied to cur_mlp (standard SwiGLU output) BEFORE summing.
    pub post_feedforward_layernorm_1: GpuTensor,
    /// `[dim]` — RMSNorm applied to cur_moe (MoE branch output) BEFORE summing.
    pub post_feedforward_layernorm_2: GpuTensor,
    /// Pool allocation for all gate_up tensors. Per-expert WeightTensors
    /// alias into this; `free_gpu` frees the pool, not each WeightTensor.
    pub experts_gate_up_pool: GpuTensor,
    /// Pool allocation for all down tensors. Same aliasing.
    pub experts_down_pool: GpuTensor,
    /// Per-expert views into the pools above.
    pub experts: Vec<MoeExpertWeights>,
}

pub enum LayerWeights {
    Sliding(SlidingLayerWeights),
    Full(FullLayerWeights),
}

pub struct Gemma4Weights {
    /// Token embedding [vocab_size, dim], Q8F16 to keep the 262144×5376 table manageable.
    /// Aliased as lm_head when tie_word_embeddings is true.
    pub embed_tokens: GpuTensor,
    /// Embed/LM-head format tag for dispatch.
    pub embd_format: EmbeddingFormat,
    /// LM-head projection (shares bytes with embed_tokens when tied).
    pub lm_head: WeightTensor,
    /// Model-final RMSNorm scale [dim].
    pub final_norm: GpuTensor,
    /// Per-layer weights indexed by layer ordinal.
    pub layers: Vec<LayerWeights>,
}

impl Gemma4Weights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.embed_tokens);
        let _ = gpu.free_tensor(self.final_norm);
        // lm_head may alias embed_tokens — skip if so (we rely on the loader
        // to set `lm_head.buf` to an alias and not a separate allocation).
        for l in self.layers {
            match l {
                LayerWeights::Sliding(s) => {
                    for t in [s.input_layernorm, s.post_attention_layernorm,
                              s.pre_feedforward_layernorm, s.post_feedforward_layernorm,
                              s.layer_scalar, s.q_norm, s.k_norm] {
                        let _ = gpu.free_tensor(t);
                    }
                    for wt in [s.q_proj.buf, s.k_proj.buf, s.v_proj.buf, s.o_proj.buf,
                               s.gate_proj.buf, s.up_proj.buf, s.down_proj.buf] {
                        let _ = gpu.free_tensor(wt);
                    }
                    if let Some(moe) = s.moe { Self::free_moe(gpu, moe); }
                }
                LayerWeights::Full(f) => {
                    for t in [f.input_layernorm, f.post_attention_layernorm,
                              f.pre_feedforward_layernorm, f.post_feedforward_layernorm,
                              f.layer_scalar, f.q_norm, f.k_norm] {
                        let _ = gpu.free_tensor(t);
                    }
                    for wt in [f.q_proj.buf, f.k_proj.buf, f.o_proj.buf,
                               f.gate_proj.buf, f.up_proj.buf, f.down_proj.buf] {
                        let _ = gpu.free_tensor(wt);
                    }
                    if let Some(moe) = f.moe { Self::free_moe(gpu, moe); }
                }
            }
        }
    }

    fn free_moe(gpu: &mut Gpu, moe: MoeLayerExtras) {
        let _ = gpu.free_tensor(moe.router_proj.buf);
        let _ = gpu.free_tensor(moe.router_scale);
        let _ = gpu.free_tensor(moe.per_expert_scale);
        let _ = gpu.free_tensor(moe.pre_feedforward_layernorm_2);
        let _ = gpu.free_tensor(moe.post_feedforward_layernorm_1);
        let _ = gpu.free_tensor(moe.post_feedforward_layernorm_2);
        // per-expert WeightTensors alias into the pools — skip freeing them.
        // free the two pool allocations.
        let _ = gpu.free_tensor(moe.experts_gate_up_pool);
        let _ = gpu.free_tensor(moe.experts_down_pool);
    }
}

// ─── Loading helpers ───────────────────────────────────────────────────

/// Decode a shape-[n] F16 or F32 tensor from HFQ into an F32 host Vec.
fn load_f32_vec(hfq: &HfqFile, name: &str, expected_n: usize) -> HipResult<Vec<f32>> {
    let (info, data) = hfq.tensor_data(name).ok_or_else(|| {
        hip_bridge::HipError::new(0, &format!("tensor not found: {name}"))
    })?;
    let n: usize = info.shape.iter().map(|&s| s as usize).product();
    if n != expected_n {
        return Err(hip_bridge::HipError::new(
            0, &format!("shape mismatch for {name}: expected {expected_n}, got {n}"),
        ));
    }
    let f32_data = match info.quant_type {
        1 => data.chunks_exact(2)
                 .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                 .collect(),
        2 => data.chunks_exact(4)
                 .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                 .collect(),
        qt => return Err(hip_bridge::HipError::new(
            0, &format!("expected F16/F32 for {name}, got qt={qt}"),
        )),
    };
    Ok(f32_data)
}

/// Load a Gemma 4 RMSNorm weight — `x * weight` form, NO +1 shift.
///
/// Distinct from qwen35::load_norm_weight which shifts by +1 for HF Gemma
/// 2/3-style `x * (1 + weight)`. Gemma 4 uses plain `x * weight` with weights
/// initialized to 1.0 (see modeling_gemma4.py::Gemma4RMSNorm line 157).
fn load_gemma4_norm(hfq: &HfqFile, gpu: &mut Gpu, name: &str, dim: usize)
    -> HipResult<GpuTensor>
{
    let f32_data = load_f32_vec(hfq, name, dim)?;
    gpu.upload_f32(&f32_data, &[dim])
}

/// Load a 256-element head-dim Q/K RMSNorm weight. Same semantics as
/// `load_gemma4_norm` but scoped to the attention head_dim (256 on sliding,
/// 512 on full).
fn load_gemma4_head_norm(hfq: &HfqFile, gpu: &mut Gpu, name: &str, head_dim: usize)
    -> HipResult<GpuTensor>
{
    load_gemma4_norm(hfq, gpu, name, head_dim)
}

/// Load the per-layer `layer_scalar` — shape-[1] BF16/F16 tensor — returning
/// both a GPU-resident [1]-tensor (for potential batched use) and its host-side
/// f32 value (used by the decode path to call `scale_f32(x, cpu_scalar)`).
fn load_layer_scalar(hfq: &HfqFile, gpu: &mut Gpu, name: &str)
    -> HipResult<(GpuTensor, f32)>
{
    let data = load_f32_vec(hfq, name, 1)?;
    let host_val = data[0];
    let gpu_tensor = gpu.upload_f32(&data, &[1])?;
    Ok((gpu_tensor, host_val))
}

/// Load a quantized projection weight. Mirrors qwen35::load_weight_tensor_raw
/// but uses the Gemma 4 tensor-name convention (`model.language_model.<name>`).
fn load_gemma4_weight(hfq: &HfqFile, gpu: &mut Gpu, name: &str, m: usize, k: usize)
    -> HipResult<WeightTensor>
{
    let (info, data) = hfq.tensor_data(name).ok_or_else(|| {
        hip_bridge::HipError::new(0, &format!("tensor not found: {name}"))
    })?;
    let dtype = match info.quant_type {
        1 => {
            // F16 → upload as f32
            let f32_data: Vec<f32> = data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
            };
            let buf = gpu.upload_raw(bytes, &[m, k])?;
            return Ok(WeightTensor { buf, gpu_dtype: DType::F32, m, k, row_stride: 0, awq_scale: None });
        }
        3  => DType::Q8_0,
        4  => DType::Q4K,
        6  => DType::HFQ4G256,
        7  => DType::HFQ4G128,
        8  => DType::HFQ6G256,
        9  => DType::HFQ2G256,
        10 => DType::HFQ2G128,
        11 => DType::HFQ3G256,
        12 => DType::HFQ3G128,
        13 => DType::MQ4G256,
        14 => DType::MQ8G256,
        15 => DType::MQ6G256,
        17 => DType::MQ3G256,
        18 => DType::MQ2G256,
        // MG4-G256 — Magnum-Gemma 4-bit. Same binary layout as MQ4G256 (136 B/group),
        // differs only in calibration policy at quant time. Alias to MQ4G256 so the
        // existing GEMV path handles it without a kernel change. ID was 19 on
        // origin/gemma4 pre-rebase; reassigned to 30 because master shipped
        // MQ2G256Lloyd at 19.
        30 => DType::MQ4G256,
        qt => return Err(hip_bridge::HipError::new(
            0, &format!("unsupported quant_type {qt} for {name}"),
        )),
    };
    let buf = gpu.upload_raw(data, &[data.len()])?;
    Ok(WeightTensor { buf, gpu_dtype: dtype, m, k, row_stride: 0, awq_scale: None })
}

/// Load the MoE branch weights for a single Gemma 4 MoE layer (26B-A4B).
/// Builds 128 expert WeightTensors aliased into per-layer gate_up / down
/// pool allocations. Pooling avoids the small-allocation HIP fragmentation
/// that OOM'd at layer 25 on the pre-pool path (origin/gemma4 commit log).
fn load_moe_layer_extras(hfq: &HfqFile, gpu: &mut Gpu, p: &str, config: &Gemma4Config)
    -> HipResult<MoeLayerExtras>
{
    let n_exp = config.num_experts;
    let dim = config.dim;
    let mi = config.moe_intermediate_size;

    let router_proj = load_gemma4_weight(hfq, gpu,
        &format!("{p}.router.proj.weight"), n_exp, dim)?;
    // NOTE: `router.scale` and `router.per_expert_scale` ship WITHOUT the
    // `.weight` suffix in HF's 26B-A4B safetensors (so `should_quantize`
    // returns false → stored as F16). Loader uses bare paths.
    let router_scale = load_gemma4_norm(hfq, gpu, &format!("{p}.router.scale"), dim)?;
    let per_expert_scale_host = load_f32_vec(hfq,
        &format!("{p}.router.per_expert_scale"), n_exp)?;
    let per_expert_scale = {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                per_expert_scale_host.as_ptr() as *const u8,
                per_expert_scale_host.len() * 4,
            )
        };
        gpu.upload_raw(bytes, &[n_exp])?
    };
    let pre_feedforward_layernorm_2 = load_gemma4_norm(hfq, gpu,
        &format!("{p}.pre_feedforward_layernorm_2.weight"), dim)?;
    let post_feedforward_layernorm_1 = load_gemma4_norm(hfq, gpu,
        &format!("{p}.post_feedforward_layernorm_1.weight"), dim)?;
    let post_feedforward_layernorm_2 = load_gemma4_norm(hfq, gpu,
        &format!("{p}.post_feedforward_layernorm_2.weight"), dim)?;

    // Pool all `n_experts` weights of one kind into a single GPU allocation.
    // 128 experts × 2 kinds × 30 layers = 7680 separate hipMalloc on the
    // unpooled path fragmented the HIP heap and OOM'd at layer ~25 even
    // when total memory fit. Pool collapses that to 60 allocs.
    let load_pool = |gpu: &mut Gpu, base: &str|
        -> HipResult<(GpuTensor, DType, usize)>
    {
        // First pass: read first expert to learn quant_type + bytes-per-expert.
        let first_name = format!("{p}.experts.0.{base}.weight");
        let (first_info, first_data) = hfq.tensor_data(&first_name).ok_or_else(|| {
            hip_bridge::HipError::new(0, &format!("MoE expert tensor not found: {first_name}"))
        })?;
        let bytes_per_expert = first_data.len();
        let dtype = match first_info.quant_type {
            3 => DType::Q8_0, 4 => DType::Q4K,
            6 => DType::HFQ4G256, 7 => DType::HFQ4G128, 8 => DType::HFQ6G256,
            9 => DType::HFQ2G256, 10 => DType::HFQ2G128,
            11 => DType::HFQ3G256, 12 => DType::HFQ3G128,
            // MQ4G256 (13) and MG4G256 (30) share dispatch.
            13 | 30 => DType::MQ4G256,
            14 => DType::MQ8G256, 15 => DType::MQ6G256,
            17 => DType::MQ3G256, 18 => DType::MQ2G256,
            qt => return Err(hip_bridge::HipError::new(
                0, &format!("unsupported MoE expert quant_type {qt} for {first_name}"),
            )),
        };
        // Concat all experts' bytes into one CPU buffer, upload once.
        let mut concat = Vec::with_capacity(bytes_per_expert * n_exp);
        concat.extend_from_slice(first_data);
        for x in 1..n_exp {
            let name = format!("{p}.experts.{x}.{base}.weight");
            let (info, data) = hfq.tensor_data(&name).ok_or_else(|| {
                hip_bridge::HipError::new(0, &format!("MoE expert tensor not found: {name}"))
            })?;
            if data.len() != bytes_per_expert {
                return Err(hip_bridge::HipError::new(
                    0, &format!("MoE expert {name} byte size mismatch ({} vs {bytes_per_expert})",
                        data.len()),
                ));
            }
            if info.quant_type != first_info.quant_type {
                return Err(hip_bridge::HipError::new(
                    0, &format!("MoE expert {name} quant_type mismatch ({} vs {})",
                        info.quant_type, first_info.quant_type),
                ));
            }
            concat.extend_from_slice(data);
        }
        let pool = gpu.upload_raw(&concat, &[concat.len()])?;
        Ok((pool, dtype, bytes_per_expert))
    };

    let (gate_up_pool, gate_up_dtype, gate_up_bytes) = load_pool(gpu, "gate_up_proj")?;
    let (down_pool, down_dtype, down_bytes) = load_pool(gpu, "down_proj")?;

    let mut experts = Vec::with_capacity(n_exp);
    for x in 0..n_exp {
        let gu_view = gate_up_pool.sub_offset(x * gate_up_bytes, gate_up_bytes);
        let dn_view = down_pool.sub_offset(x * down_bytes, down_bytes);
        experts.push(MoeExpertWeights {
            gate_up_proj: WeightTensor {
                buf: gu_view, gpu_dtype: gate_up_dtype,
                m: 2 * mi, k: dim, row_stride: 0, awq_scale: None,
            },
            down_proj: WeightTensor {
                buf: dn_view, gpu_dtype: down_dtype,
                m: dim, k: mi, row_stride: 0, awq_scale: None,
            },
        });
    }

    Ok(MoeLayerExtras {
        router_proj,
        router_scale,
        per_expert_scale,
        per_expert_scale_host,
        pre_feedforward_layernorm_2,
        post_feedforward_layernorm_1,
        post_feedforward_layernorm_2,
        experts_gate_up_pool: gate_up_pool,
        experts_down_pool: down_pool,
        experts,
    })
}

/// Load Gemma 4 text model weights from an HFQ file.
///
/// Design notes:
///   - `lm_head` aliases the `embed_tokens` GPU bytes (tied weights). We upload
///     the embed data once and create a second WeightTensor whose DeviceBuffer
///     points at the same allocation via `buf.alias()`. `Gemma4Weights::free_gpu`
///     skips freeing the LM head to avoid a double-free.
///   - Vision tensors are skipped here — Phase 7 `gemma4_vision::load_weights`
///     picks those up from the same HFQ file in a separate pass.
///   - The `v_norm_ones_full` ones-filled scratch buffer is populated here so
///     the forward pass never has to manage one-time init state.
pub fn load_weights(hfq: &mut HfqFile, config: &Gemma4Config, gpu: &mut Gpu)
    -> HipResult<Gemma4Weights>
{
    eprintln!("gemma4: loading embed_tokens...");
    let embed_name = "model.language_model.embed_tokens.weight";
    let (embed_info, embed_data) = hfq.tensor_data(embed_name).ok_or_else(|| {
        hip_bridge::HipError::new(0, "embed_tokens not found in HFQ")
    })?;
    let (embed_tokens, embd_format) = match embed_info.quant_type {
        3 => {
            eprintln!("  (Q8_0 / Q8F16, {} MB)", embed_data.len() / 1_000_000);
            (gpu.upload_raw(embed_data, &[embed_data.len()])?, EmbeddingFormat::Q8_0)
        }
        6 => {
            eprintln!("  (HFQ4-G256, {} MB)", embed_data.len() / 1_000_000);
            (gpu.upload_raw(embed_data, &[embed_data.len()])?, EmbeddingFormat::HFQ4G256)
        }
        7 => {
            eprintln!("  (HFQ4-G128, {} MB)", embed_data.len() / 1_000_000);
            (gpu.upload_raw(embed_data, &[embed_data.len()])?, EmbeddingFormat::HFQ4G128)
        }
        1 => {
            eprintln!("  (F16 → F32)");
            let f32_data: Vec<f32> = embed_data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            (gpu.upload_f32(&f32_data, &[config.vocab_size, config.dim])?, EmbeddingFormat::F32)
        }
        qt => return Err(hip_bridge::HipError::new(
            0, &format!("unsupported embed quant_type {qt}"),
        )),
    };

    // Tied LM head: WeightTensor whose buffer aliases the embed allocation.
    // free_gpu skips freeing this — embed_tokens owns the bytes.
    let lm_head = {
        let alias_buf = unsafe { embed_tokens.buf.alias() };
        let dtype = match embd_format {
            EmbeddingFormat::Q8_0    => DType::Q8_0,
            EmbeddingFormat::HFQ4G256 => DType::HFQ4G256,
            EmbeddingFormat::HFQ4G128 => DType::HFQ4G128,
            EmbeddingFormat::F32     => DType::F32,
            EmbeddingFormat::Q4K     => DType::Q4K,
        };
        let alias_tensor = GpuTensor {
            buf: alias_buf,
            shape: embed_tokens.shape.clone(),
            dtype,
        };
        WeightTensor { buf: alias_tensor, gpu_dtype: dtype, m: config.vocab_size, k: config.dim, row_stride: 0, awq_scale: None }
    };

    eprintln!("gemma4: loading final norm...");
    let final_norm = load_gemma4_norm(hfq, gpu, "model.language_model.norm.weight", config.dim)?;

    eprintln!("gemma4: loading {} layers...", config.n_layers);
    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        let p = format!("model.language_model.layers.{i}");
        match config.layer_types[i] {
            LayerType::Sliding => {
                let hd = config.sliding_head_dim;
                let kv_dim = config.sliding_n_kv_heads * hd;
                let q_dim = config.n_heads * hd;
                let (layer_scalar, layer_scalar_host) =
                    load_layer_scalar(hfq, gpu, &format!("{p}.layer_scalar"))?;
                let moe = if config.enable_moe_block {
                    Some(load_moe_layer_extras(hfq, gpu, &p, config)?)
                } else { None };
                layers.push(LayerWeights::Sliding(SlidingLayerWeights {
                    input_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.input_layernorm.weight"), config.dim)?,
                    post_attention_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.post_attention_layernorm.weight"), config.dim)?,
                    pre_feedforward_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.pre_feedforward_layernorm.weight"), config.dim)?,
                    post_feedforward_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.post_feedforward_layernorm.weight"), config.dim)?,
                    layer_scalar,
                    layer_scalar_host,
                    q_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.self_attn.q_proj.weight"), q_dim, config.dim)?,
                    k_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.self_attn.k_proj.weight"), kv_dim, config.dim)?,
                    v_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.self_attn.v_proj.weight"), kv_dim, config.dim)?,
                    o_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.self_attn.o_proj.weight"), config.dim, q_dim)?,
                    q_norm: load_gemma4_head_norm(hfq, gpu,
                        &format!("{p}.self_attn.q_norm.weight"), hd)?,
                    k_norm: load_gemma4_head_norm(hfq, gpu,
                        &format!("{p}.self_attn.k_norm.weight"), hd)?,
                    gate_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.gate_proj.weight"), config.hidden_dim, config.dim)?,
                    up_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.up_proj.weight"), config.hidden_dim, config.dim)?,
                    down_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.down_proj.weight"), config.dim, config.hidden_dim)?,
                    moe,
                }));
            }
            LayerType::Full => {
                let hd = config.full_head_dim;
                let kv_dim = config.full_n_kv_heads * hd;
                let q_dim = config.n_heads * hd;
                let (layer_scalar, layer_scalar_host) =
                    load_layer_scalar(hfq, gpu, &format!("{p}.layer_scalar"))?;
                let moe = if config.enable_moe_block {
                    Some(load_moe_layer_extras(hfq, gpu, &p, config)?)
                } else { None };
                layers.push(LayerWeights::Full(FullLayerWeights {
                    input_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.input_layernorm.weight"), config.dim)?,
                    post_attention_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.post_attention_layernorm.weight"), config.dim)?,
                    pre_feedforward_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.pre_feedforward_layernorm.weight"), config.dim)?,
                    post_feedforward_layernorm: load_gemma4_norm(hfq, gpu,
                        &format!("{p}.post_feedforward_layernorm.weight"), config.dim)?,
                    layer_scalar,
                    layer_scalar_host,
                    q_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.self_attn.q_proj.weight"), q_dim, config.dim)?,
                    k_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.self_attn.k_proj.weight"), kv_dim, config.dim)?,
                    // no v_proj on full layers — V reuses k_proj's pre-norm output.
                    o_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.self_attn.o_proj.weight"), config.dim, q_dim)?,
                    q_norm: load_gemma4_head_norm(hfq, gpu,
                        &format!("{p}.self_attn.q_norm.weight"), hd)?,
                    k_norm: load_gemma4_head_norm(hfq, gpu,
                        &format!("{p}.self_attn.k_norm.weight"), hd)?,
                    // no v_norm weight — v_norm is no-scale (ones buffer passed at decode time).
                    gate_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.gate_proj.weight"), config.hidden_dim, config.dim)?,
                    up_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.up_proj.weight"), config.hidden_dim, config.dim)?,
                    down_proj: load_gemma4_weight(hfq, gpu,
                        &format!("{p}.mlp.down_proj.weight"), config.dim, config.hidden_dim)?,
                    moe,
                }));
            }
        }
    }
    eprintln!("gemma4: loaded all {} layers", config.n_layers);

    Ok(Gemma4Weights {
        embed_tokens,
        embd_format,
        lm_head,
        final_norm,
        layers,
    })
}

/// One-time init for the scratch buffers that must hold a constant value
/// across forward passes (notably the ones-filled `v_norm_ones_full`).
/// Call once after `Gemma4Scratch::new` before the first forward pass.
pub fn init_scratch_constants(gpu: &mut Gpu, scratch: &Gemma4Scratch, full_head_dim: usize)
    -> HipResult<()>
{
    let ones: Vec<f32> = vec![1.0; full_head_dim];
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(ones.as_ptr() as *const u8, ones.len() * 4)
    };
    gpu.hip.memcpy_htod(&scratch.v_norm_ones_full.buf, bytes)?;
    Ok(())
}

// ─── Scratch ────────────────────────────────────────────────────────────

use hip_bridge::DeviceBuffer;

/// Per-decode scratch, sized once at model-load time against the MAX of
/// sliding and full attention dimensions so a single buffer works across
/// layer types. 31B target shapes: sliding Q=[32*256]=8192, full Q=[32*512]=16384
/// → size Q at 16384. Sliding KV=[16*256]=4096, full KV=[4*512]=2048 → size at 4096.
pub struct Gemma4Scratch {
    pub x: GpuTensor,           // [dim] — hidden state
    pub residual: GpuTensor,    // [dim] — saved for sandwich residual
    pub tmp: GpuTensor,         // [dim] — norm output scratch

    /// Position buffer (single i32 on device, updated per decode step).
    pub pos_buf: DeviceBuffer,

    // Attention scratch — sized for max(sliding, full)
    pub q: GpuTensor,           // [max(n_heads*head_dim_sliding, n_heads*head_dim_full)]
    pub k: GpuTensor,           // [max(n_kv_heads*head_dim for each layer type)]
    pub v: GpuTensor,           // [same as k]
    pub attn_out: GpuTensor,    // [same as q]

    // MLP scratch
    pub gate_ffn: GpuTensor,    // [hidden_dim]
    pub up_ffn: GpuTensor,      // [hidden_dim]
    pub ffn_hidden: GpuTensor,  // [hidden_dim]
    pub ffn_out: GpuTensor,     // [dim]

    // Output
    pub logits: GpuTensor,      // [vocab_size]
    pub sample_buf: GpuTensor,  // [2] — (token_id, new_rng_state) for GPU sampling
    pub repeat_buf: GpuTensor,  // [1024] — rolling window for repeat penalty

    // Flash attention tile partials. Sized for the LARGER of the two
    // cache shapes: full-attn uses head_dim=512, max_tiles=max_seq/128.
    // Sliding uses head_dim=256, max_tiles=sliding_window/128 (much smaller).
    pub flash_partials: GpuTensor,

    // Pre-computed RoPE cos/sin tables per layer type.
    // Sliding: default RoPE, head_dim=256, theta=10000, n_rot = head_dim.
    pub sliding_cos: GpuTensor, // [max_seq, head_dim]
    pub sliding_sin: GpuTensor,
    // Full: proportional RoPE, head_dim=512, theta=1e6, rotated_dims = 64 of 256-half
    pub full_cos: GpuTensor,
    pub full_sin: GpuTensor,

    // No-scale v_norm ones buffer (full-attn layers compute v_norm without
    // a learned weight — we pass this ones-filled tensor to the existing
    // rmsnorm kernel to get no-scale RMS semantics).
    pub v_norm_ones_full: GpuTensor, // [full_head_dim]

    // ── MoE scratch (26B-A4B only). Zero-sized on dense models. ─────────
    pub moe_cur_mlp: GpuTensor,        // [dim] — rmsnorm(ffn_out, post_norm_1)
    pub moe_pre2: GpuTensor,           // [dim] — rmsnorm(attn_out, pre_norm_2)
    pub moe_router_in: GpuTensor,      // [dim] — router input (post-rmsnorm + scale)
    pub moe_router_logits: GpuTensor,  // [n_experts]
    pub moe_topk_indices: GpuTensor,   // [top_k_experts] — i32 packed in f32 slots
    pub moe_topk_weights: GpuTensor,   // [top_k_experts]
    pub moe_cur_moe: GpuTensor,        // [dim] — accumulator across top-K experts
    pub moe_expert_gate_up: GpuTensor, // [2 * moe_intermediate_size]
    pub moe_expert_hidden: GpuTensor,  // [moe_intermediate_size] — gelu(gate) * up
    pub moe_expert_out: GpuTensor,     // [dim] — single expert's down_proj output
}

impl Gemma4Scratch {
    pub fn new(gpu: &mut Gpu, config: &Gemma4Config, _max_prefill: usize) -> HipResult<Self> {
        let dim = config.dim;
        let q_dim = (config.n_heads * config.sliding_head_dim).max(config.n_heads * config.full_head_dim);
        let kv_dim = (config.sliding_n_kv_heads * config.sliding_head_dim)
            .max(config.full_n_kv_heads * config.full_head_dim);

        let x = gpu.zeros(&[dim], DType::F32)?;
        let residual = gpu.zeros(&[dim], DType::F32)?;
        let tmp = gpu.zeros(&[dim], DType::F32)?;

        let pos_buf = gpu.hip.malloc(4)?;

        let q = gpu.zeros(&[q_dim], DType::F32)?;
        let k = gpu.zeros(&[kv_dim], DType::F32)?;
        let v = gpu.zeros(&[kv_dim], DType::F32)?;
        let attn_out = gpu.zeros(&[q_dim], DType::F32)?;

        let gate_ffn = gpu.zeros(&[config.hidden_dim], DType::F32)?;
        let up_ffn = gpu.zeros(&[config.hidden_dim], DType::F32)?;
        let ffn_hidden = gpu.zeros(&[config.hidden_dim], DType::F32)?;
        let ffn_out = gpu.zeros(&[dim], DType::F32)?;

        let logits = gpu.zeros(&[config.vocab_size], DType::F32)?;
        let sample_buf = gpu.zeros(&[2], DType::F32)?;
        let repeat_buf = gpu.zeros(&[1024], DType::F32)?;

        // Flash partials sizing. Per-head × max_tiles × (2 + head_dim) floats.
        // Sized for FULL attn (head_dim=512 stride 514, vs sliding 256 stride 258);
        // sliding-layer dispatches use part of the buffer, full-layer dispatches
        // use all of it. Override with HIPFIRE_KV_SEQ to support contexts beyond
        // 32k (Gemma 4 supports up to 128k natively); the daemon must keep
        // kv_cache.max_seq <= this value or dispatch will return a loud Err
        // (see runtime guard in attention_flash_asym3_window). 32k is the
        // hipfire-wide default that matches Qwen3.5 / 3.6 production max.
        const FALLBACK_KV_SEQ: usize = 32768;
        const TILE_SIZE: usize = 128;
        let max_kv_seq: usize = std::env::var("HIPFIRE_KV_SEQ")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n >= 128 && n <= 524_288)
            .unwrap_or(FALLBACK_KV_SEQ);
        let max_tiles_full = (max_kv_seq + TILE_SIZE - 1) / TILE_SIZE;
        let flash_partials_sz = config.n_heads * max_tiles_full * (2 + config.full_head_dim);
        let flash_partials = gpu.zeros(&[flash_partials_sz], DType::F32)?;

        // RoPE tables. Same max_kv_seq cap so a single env override scales
        // partials + tables together. The loader populates these per-model.
        let sliding_cos = gpu.zeros(&[max_kv_seq * config.sliding_head_dim], DType::F32)?;
        let sliding_sin = gpu.zeros(&[max_kv_seq * config.sliding_head_dim], DType::F32)?;
        let full_cos = gpu.zeros(&[max_kv_seq * config.full_head_dim], DType::F32)?;
        let full_sin = gpu.zeros(&[max_kv_seq * config.full_head_dim], DType::F32)?;

        // v_norm ones — populated on first use in the forward pass.
        // Allocated up to the LARGER of sliding_head_dim and full_head_dim
        // since both sliding and full apply no-scale v_norm (the post-rebase
        // fix added v_norm to sliding_layer_decode; sliding head_dim=256,
        // full head_dim=512 → max=512 covers both).
        let v_norm_max = config.sliding_head_dim.max(config.full_head_dim);
        let v_norm_ones_full = gpu.zeros(&[v_norm_max], DType::F32)?;

        // MoE scratch. Allocated unconditionally because the buffers are tiny
        // relative to the model; zero-sized on dense models would just complicate
        // the dispatch path. Sized for 26B-A4B: n_experts=128, top_k=8, mi=704.
        let n_exp = config.num_experts.max(1);
        let mi = config.moe_intermediate_size.max(1);
        let k_top = config.top_k_experts.max(1);
        let moe_cur_mlp = gpu.zeros(&[dim], DType::F32)?;
        let moe_pre2 = gpu.zeros(&[dim], DType::F32)?;
        let moe_router_in = gpu.zeros(&[dim], DType::F32)?;
        let moe_router_logits = gpu.zeros(&[n_exp], DType::F32)?;
        let moe_topk_indices = gpu.zeros(&[k_top], DType::F32)?;
        let moe_topk_weights = gpu.zeros(&[k_top], DType::F32)?;
        let moe_cur_moe = gpu.zeros(&[dim], DType::F32)?;
        let moe_expert_gate_up = gpu.zeros(&[2 * mi], DType::F32)?;
        let moe_expert_hidden = gpu.zeros(&[mi], DType::F32)?;
        let moe_expert_out = gpu.zeros(&[dim], DType::F32)?;

        Ok(Gemma4Scratch {
            x, residual, tmp, pos_buf,
            q, k, v, attn_out,
            gate_ffn, up_ffn, ffn_hidden, ffn_out,
            logits, sample_buf, repeat_buf,
            flash_partials,
            sliding_cos, sliding_sin, full_cos, full_sin,
            v_norm_ones_full,
            moe_cur_mlp, moe_pre2, moe_router_in, moe_router_logits,
            moe_topk_indices, moe_topk_weights, moe_cur_moe,
            moe_expert_gate_up, moe_expert_hidden, moe_expert_out,
        })
    }

    /// Release every GPU allocation owned by this scratch. Mirrors the
    /// Qwen35Scratch / LlamaScratch pattern so `unload_model` in the daemon
    /// can reclaim VRAM on idle eviction.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.x);
        let _ = gpu.free_tensor(self.residual);
        let _ = gpu.free_tensor(self.tmp);
        // pos_buf is a DeviceBuffer, not a GpuTensor; rely on Drop.
        let _ = gpu.free_tensor(self.q);
        let _ = gpu.free_tensor(self.k);
        let _ = gpu.free_tensor(self.v);
        let _ = gpu.free_tensor(self.attn_out);
        let _ = gpu.free_tensor(self.gate_ffn);
        let _ = gpu.free_tensor(self.up_ffn);
        let _ = gpu.free_tensor(self.ffn_hidden);
        let _ = gpu.free_tensor(self.ffn_out);
        let _ = gpu.free_tensor(self.logits);
        let _ = gpu.free_tensor(self.sample_buf);
        let _ = gpu.free_tensor(self.repeat_buf);
        let _ = gpu.free_tensor(self.flash_partials);
        let _ = gpu.free_tensor(self.sliding_cos);
        let _ = gpu.free_tensor(self.sliding_sin);
        let _ = gpu.free_tensor(self.full_cos);
        let _ = gpu.free_tensor(self.full_sin);
        let _ = gpu.free_tensor(self.v_norm_ones_full);
        let _ = gpu.free_tensor(self.moe_cur_mlp);
        let _ = gpu.free_tensor(self.moe_pre2);
        let _ = gpu.free_tensor(self.moe_router_in);
        let _ = gpu.free_tensor(self.moe_router_logits);
        let _ = gpu.free_tensor(self.moe_topk_indices);
        let _ = gpu.free_tensor(self.moe_topk_weights);
        let _ = gpu.free_tensor(self.moe_cur_moe);
        let _ = gpu.free_tensor(self.moe_expert_gate_up);
        let _ = gpu.free_tensor(self.moe_expert_hidden);
        let _ = gpu.free_tensor(self.moe_expert_out);
    }
}

// ─── Forward pass ───────────────────────────────────────────────────────

/// Apply the Gemma 4 MoE parallel branch (26B-A4B). Called from each layer
/// AFTER `down_proj` produces `scratch.ffn_out`, REPLACING the standalone
/// `post_feedforward_layernorm` call. On exit, `scratch.tmp` holds the
/// combined `post_norm(cur_mlp + cur_moe)`, ready for `x = residual + tmp`.
///
/// Legacy serialized path only (8 experts × 5 launches = 40 launches/layer).
/// The fused indexed-GEMV path (`gemv_hfq4g256_moe_gate_up_k8_indexed`) and
/// fused-down path from origin/gemma4 are NOT yet ported — they require a
/// `rotate_x_mq` + `mq_signs` plumbing the modular crate doesn't have yet.
/// Both produce mathematically identical output; the legacy path is the
/// safety/reference baseline.
///
/// HF reference (modeling_gemma4.py Gemma4MoeBlock + Gemma4MoeMLP):
///   cur_mlp = post_feedforward_layernorm_1(ffn_out)        # standard SwiGLU out, normed
///   pre2    = pre_feedforward_layernorm_2(attn_out)        # MoE branch input
///   router_in    = rmsnorm(attn_out, router_scale) / sqrt(dim)
///   router_logits = router_proj @ router_in
///   topk_idx, topk_w = softmax_topk_renorm(router_logits, k=8)
///   cur_moe = sum_k [ topk_w[k] * per_expert_scale[i_k] *
///                     down_proj_{i_k}( gelu_tanh(gate) * up
///                                      where (gate, up) = split(gate_up_proj_{i_k} @ pre2) ) ]
///   cur_moe = post_feedforward_layernorm_2(cur_moe)
///   tmp     = post_feedforward_layernorm(cur_mlp + cur_moe)
///
/// `attn_out` parameter is the layer's post-attention residual stream
/// (= `scratch.residual` at the call site, since the caller stored
/// `residual = x` after the attention sandwich).
fn apply_moe_branch(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    scratch: &Gemma4Scratch,
    moe: &MoeLayerExtras,
    post_ffn_norm: &GpuTensor,
    attn_out: &GpuTensor,
) -> HipResult<()> {
    let dim = config.dim;
    let dim_bytes = dim * 4;
    let mi = config.moe_intermediate_size;
    let n_exp = config.num_experts;
    let k_top = config.top_k_experts;
    if k_top != 8 {
        return Err(hip_bridge::HipError::new(
            0, &format!("MoE top_k_experts={k_top} unsupported (kernel hardcoded to 8)"),
        ));
    }

    // 1) cur_mlp = post_feedforward_layernorm_1(ffn_out)
    gpu.rmsnorm_f32(&scratch.ffn_out, &moe.post_feedforward_layernorm_1,
        &scratch.moe_cur_mlp, config.norm_eps)?;

    // 2) pre2 = pre_feedforward_layernorm_2(attn_out)
    gpu.rmsnorm_f32(attn_out, &moe.pre_feedforward_layernorm_2,
        &scratch.moe_pre2, config.norm_eps)?;

    // 3) Router input: rmsnorm(attn_out, router_scale) / sqrt(dim).
    //    Equivalent to ref `rms_norm(x) * router_scale / sqrt(dim)` since
    //    rmsnorm_f32(x, w) = w * x / sqrt(mean(x²) + eps) — elementwise commutative.
    gpu.rmsnorm_f32(attn_out, &moe.router_scale,
        &scratch.moe_router_in, config.norm_eps)?;
    gpu.scale_f32(&scratch.moe_router_in, 1.0 / (dim as f32).sqrt())?;

    // 4) Router GEMV → logits [n_exp]
    weight_gemv(gpu, &moe.router_proj, &scratch.moe_router_in, &scratch.moe_router_logits)?;

    // 5) Top-K softmax + renorm on device. Kernel hardcoded to k_top=8.
    gpu.moe_softmax_topk_renorm_k8(
        &scratch.moe_router_logits,
        &scratch.moe_topk_indices,
        &scratch.moe_topk_weights,
        n_exp,
        true,
    )?;

    // 6) D2H topk indices + weights so the CPU loop can index into
    //    `moe.experts[e]`. The fused indexed-GEMV path (not ported here)
    //    keeps these on device and dispatches a single batched GEMV;
    //    legacy path needs CPU access for the per-expert weight lookup.
    let idx_bytes = gpu.download_f32(&scratch.moe_topk_indices)?;
    let topk_indices: Vec<usize> = unsafe {
        std::slice::from_raw_parts(idx_bytes.as_ptr() as *const i32, k_top)
    }.iter().map(|&i| i as usize).collect();
    let topk_weights = gpu.download_f32(&scratch.moe_topk_weights)?;
    for &e in topk_indices.iter().take(k_top) {
        if e >= n_exp {
            return Err(hip_bridge::HipError::new(
                0, &format!("MoE topk index {e} out of range (n_exp={n_exp})"),
            ));
        }
    }

    // 7) Zero accumulator
    gpu.hip.memset(&scratch.moe_cur_moe.buf, 0, dim_bytes)?;

    // 8) Per-expert serialized loop. 8 iterations × 5 launches each.
    for ki in 0..k_top {
        let e = topk_indices[ki];
        let weight = topk_weights[ki] * moe.per_expert_scale_host[e];
        let expert = &moe.experts[e];

        // gate_up[2*mi] = expert.gate_up_proj @ pre2
        weight_gemv(gpu, &expert.gate_up_proj, &scratch.moe_pre2, &scratch.moe_expert_gate_up)?;
        // Split: rows [0, mi) = gate; rows [mi, 2*mi) = up.
        let gate = scratch.moe_expert_gate_up.sub_offset(0, mi);
        let up   = scratch.moe_expert_gate_up.sub_offset(mi, mi);
        // hidden = gelu_tanh(gate) * up
        gpu.gelu_tanh_f32(&gate, &scratch.moe_expert_hidden, mi)?;
        gpu.mul_f32(&scratch.moe_expert_hidden, &up, &scratch.moe_expert_hidden)?;
        // expert_out[dim] = expert.down_proj @ hidden
        weight_gemv(gpu, &expert.down_proj, &scratch.moe_expert_hidden, &scratch.moe_expert_out)?;
        // cur_moe += weight * expert_out
        gpu.scaled_add_inplace_cpu_scalar_f32(&scratch.moe_cur_moe, &scratch.moe_expert_out, weight)?;
    }

    // 9) cur_moe = post_feedforward_layernorm_2(cur_moe) — in-place
    gpu.rmsnorm_f32(&scratch.moe_cur_moe, &moe.post_feedforward_layernorm_2,
        &scratch.moe_cur_moe, config.norm_eps)?;

    // 10) combined = cur_mlp + cur_moe → scratch.tmp
    gpu.add_f32(&scratch.moe_cur_mlp, &scratch.moe_cur_moe, &scratch.tmp)?;

    // 11) tmp = post_feedforward_layernorm(combined)
    gpu.rmsnorm_f32(&scratch.tmp, post_ffn_norm, &scratch.tmp, config.norm_eps)?;

    Ok(())
}

/// Single-token decode. Phase 3 implementation.
///
/// Precondition: `scratch.sliding_cos/sin` + `scratch.full_cos/sin` +
/// `scratch.v_norm_ones_full` must be populated by the loader before the
/// first forward call (one-time init).
pub fn forward_scratch(
    gpu: &mut Gpu,
    weights: &Gemma4Weights,
    config: &Gemma4Config,
    token: u32,
    pos: usize,
    kv_sliding: &mut hipfire_runtime::llama::KvCache,
    kv_full: &mut hipfire_runtime::llama::KvCache,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    let dim = config.dim;

    // 1) Embedding lookup + sqrt(dim) scale.
    //
    // Gemma 4 multiplies the embedding row by sqrt(hidden_size) (bf16-cast
    // in the reference — we do it in fp32 here; the absolute magnitude
    // difference is sub-epsilon for our MQ4 quality target).
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(&weights.embed_tokens, &scratch.x, token, dim)?,
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(&weights.embed_tokens, &scratch.x, token, dim)?,
        EmbeddingFormat::Q8_0    => gpu.embedding_lookup_q8(&weights.embed_tokens, &scratch.x, token, dim)?,
        EmbeddingFormat::F32     => gpu.embedding_lookup(&weights.embed_tokens, &scratch.x, token, dim)?,
        _ => return Err(hip_bridge::HipError::new(0, "unsupported Gemma 4 embed format")),
    }
    gpu.scale_f32(&scratch.x, config.embed_scale)?;

    // 2) Update device pos_buf.
    let pos_i32 = pos as i32;
    gpu.hip.memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;

    // 3) Per-layer forward.
    let mut sliding_kv_idx = 0usize;
    let mut full_kv_idx = 0usize;
    for (layer_idx, layer_type) in config.layer_types.iter().copied().enumerate() {
        match (layer_type, &weights.layers[layer_idx]) {
            (LayerType::Sliding, LayerWeights::Sliding(lw)) => {
                sliding_layer_decode(gpu, config, lw, pos, kv_sliding, sliding_kv_idx, scratch)?;
                sliding_kv_idx += 1;
            }
            (LayerType::Full, LayerWeights::Full(lw)) => {
                full_layer_decode(gpu, config, lw, pos, kv_full, full_kv_idx, scratch)?;
                full_kv_idx += 1;
            }
            _ => return Err(hip_bridge::HipError::new(
                0,
                &format!("Gemma 4 layer {} type/weights mismatch", layer_idx),
            )),
        }
    }

    // 4) Final RMSNorm.
    gpu.rmsnorm_f32(&scratch.x, &weights.final_norm, &scratch.tmp, config.norm_eps)?;

    // 5) LM head → logits (reads tied embed bytes via lm_head.buf alias).
    weight_gemv(gpu, &weights.lm_head, &scratch.tmp, &scratch.logits)?;

    // 6) Final logit softcap (Gemma 4): logits = tanh(logits / cap) * cap.
    if config.final_logit_softcapping > 0.0 {
        gpu.logit_softcap_f32(&scratch.logits, config.vocab_size, config.final_logit_softcapping)?;
    }

    Ok(())
}

/// Single sliding-window attention layer.
///
/// Order matches HF modeling_gemma4.py::Gemma4TextDecoderLayer +
/// Gemma4TextAttention (sliding branch):
///   residual = x
///   x = input_layernorm(x)              — RMSNorm (sandwich pre-attn)
///   q = q_proj(x); q = q_norm(q)        — RMSNorm over head_dim=256
///   k = k_proj(x); k = k_norm(k)
///   v = v_proj(x); v = v_norm(v)         — no_scale RMSNorm (ones buffer)
///   RoPE(q, k) with rotate_half, theta=10000, full head_dim=256
///   write K, V to KV cache at position `pos`
///   attn = flash_attention(q, K, V, window_size=1024, scale=1.0 effective)
///   x = o_proj(attn)
///   x = post_attention_layernorm(x)     — RMSNorm (sandwich post-attn)
///   x = residual + x
///   residual = x
///   x = pre_feedforward_layernorm(x)    — RMSNorm (sandwich pre-FFN)
///   gate = gate_proj(x); up = up_proj(x)
///   ffn = gelu_pytorch_tanh(gate) * up  — SwiGLU
///   x = down_proj(ffn)
///   x = post_feedforward_layernorm(x)   — RMSNorm (sandwich post-FFN)
///   x = residual + x
///   x = x * layer_scalar                — learned per-layer scalar
///
/// Gemma 4 attention uses `scaling=1.0` in HF (see modeling_gemma4.py line 1143).
/// Our flash kernels bake in `scale = 1/sqrt(head_dim)`; we compensate by
/// pre-scaling Q by sqrt(head_dim) so the effective scale is 1.0.
// Sliding-window kernel correctness: the 4 attention_flash_*_window dispatch
// sites here use the kernels modified by commit b608c5f (sliding-window arg
// threaded through 7 attention_flash_*.hip files). Static review of the diff
// is clean — per-position out_of_window predicate writes -1e30 to scores;
// early-tile-skip writes {-1e30, 0, zeros} so reduce sees no contribution;
// all 32 lanes participate in __shfl_xor; window_size=0 = byte-identical
// for Qwen 3.5/3.6. Runtime NRMSE validation: verify_against_torch.rs Phase 2
// (D2.5) — per-layer attention NRMSE against the bf16 PyTorch reference dump.
fn sliding_layer_decode(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    lw: &SlidingLayerWeights,
    pos: usize,
    kv_cache: &mut hipfire_runtime::llama::KvCache,
    kv_layer_idx: usize,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    let dim = config.dim;
    let head_dim = config.sliding_head_dim;
    let n_heads = config.n_heads;
    let n_kv = config.sliding_n_kv_heads;
    let dim_bytes = dim * 4;

    // residual = x
    gpu.hip.memcpy_dtod(&scratch.residual.buf, &scratch.x.buf, dim_bytes)?;

    // tmp = input_layernorm(x)
    gpu.rmsnorm_f32(&scratch.x, &lw.input_layernorm, &scratch.tmp, config.norm_eps)?;

    // Q/K/V projections: q[n_heads*head_dim], k/v[n_kv*head_dim].
    weight_gemv(gpu, &lw.q_proj, &scratch.tmp, &scratch.q)?;
    weight_gemv(gpu, &lw.k_proj, &scratch.tmp, &scratch.k)?;
    weight_gemv(gpu, &lw.v_proj, &scratch.tmp, &scratch.v)?;

    // q_norm + k_norm + no-scale v_norm across head_dim (in-place).
    // v_norm matches HF Gemma 4 `value_states = v_norm(v_proj(x))` (no_scale=True
    // RMSNorm — divide-only, ones buffer as weight). Same pattern as full_layer_decode.
    // Omitting v_norm compounds attention-output bias across 48 sliding layers and
    // produces single-token garbage end-to-end while still passing Phase 2 kernel
    // NRMSE (which tests q_norm + k_norm but not v_norm — see d2.5-results.md).
    gpu.rmsnorm_batched(&scratch.q, &lw.q_norm, &scratch.q, n_heads, head_dim, config.norm_eps)?;
    gpu.rmsnorm_batched(&scratch.k, &lw.k_norm, &scratch.k, n_kv, head_dim, config.norm_eps)?;
    gpu.rmsnorm_batched(&scratch.v, &scratch.v_norm_ones_full, &scratch.v,
        n_kv, head_dim, config.norm_eps)?;

    // Pre-scale Q by sqrt(head_dim) so the flash-attn kernel's internal
    // 1/sqrt(head_dim) cancels, leaving the effective Gemma 4 scale of 1.0.
    // Only the first n_heads*head_dim elements of scratch.q are live.
    gpu.scale_f32(&scratch.q, (head_dim as f32).sqrt())?;

    // Full rotate_half RoPE, theta=10000, head_dim=256 (all dims rotate).
    gpu.rope_f32(&scratch.q, &scratch.k, &scratch.pos_buf,
        n_heads, n_kv, head_dim, config.sliding_rope_theta)?;

    // KV cache write + flash attention with window_size=1024.
    // Branch on cache quant mode, same as qwen35::run_fa_layer_body.
    if kv_cache.quant_asym3 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        // Ring-buffer cache_capacity = sliding_window: writes wrap into a
        // sliding_window-sized cache; reads address slots via (t % cap). This
        // lets the sliding KV cache stay at constant size regardless of seq
        // length — required to fit 128k context on a 17 GB GPU.
        let sliding_cap = config.sliding_window as u32;
        gpu.kv_cache_write_asym3_fused(
            &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.k, &scratch.v, &scratch.pos_buf, ct, st, n_kv, head_dim,
            sliding_cap)?;
        gpu.attention_flash_asym3_window(
            &scratch.q, &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.attn_out, &scratch.pos_buf, ct, st, pos + 1,
            n_heads, n_kv, head_dim, kv_cache.max_seq,
            &scratch.flash_partials,
            sliding_cap,
            sliding_cap,
        )?;
    } else if kv_cache.quant_asym4 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        gpu.kv_cache_write_asym4_fused(
            &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.k, &scratch.v, &scratch.pos_buf, ct, st, n_kv, head_dim)?;
        gpu.attention_flash_asym4_window(
            &scratch.q, &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.attn_out, &scratch.pos_buf, ct, st, pos + 1,
            n_heads, n_kv, head_dim, kv_cache.max_seq,
            &scratch.flash_partials,
            config.sliding_window as u32,
        )?;
    } else if kv_cache.quant_asym2 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        gpu.kv_cache_write_asym2_fused(
            &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.k, &scratch.v, &scratch.pos_buf, ct, st, n_kv, head_dim)?;
        gpu.attention_flash_asym2_window(
            &scratch.q, &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.attn_out, &scratch.pos_buf, ct, st, pos + 1,
            n_heads, n_kv, head_dim, kv_cache.max_seq,
            &scratch.flash_partials,
            config.sliding_window as u32,
        )?;
    } else if kv_cache.quant_q8 {
        gpu.kv_cache_write_q8_0(&kv_cache.k_gpu[kv_layer_idx], &scratch.k, &scratch.pos_buf, n_kv, head_dim, 0)?;
        gpu.kv_cache_write_q8_0(&kv_cache.v_gpu[kv_layer_idx], &scratch.v, &scratch.pos_buf, n_kv, head_dim, 0)?;
        gpu.attention_flash_q8_0_window(
            &scratch.q, &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.attn_out, &scratch.pos_buf, pos + 1,
            n_heads, n_kv, head_dim, kv_cache.max_seq,
            &scratch.flash_partials,
            config.sliding_window as u32,
        )?;
    } else {
        // Plain FP32 KV path (kvf16 / kvfp32).
        let kv_dim = n_kv * head_dim;
        gpu.kv_cache_write(&kv_cache.k_gpu[kv_layer_idx], &scratch.k, &scratch.pos_buf, kv_dim)?;
        gpu.kv_cache_write(&kv_cache.v_gpu[kv_layer_idx], &scratch.v, &scratch.pos_buf, kv_dim)?;
        // No sliding-window support in the plain attention_f32 kernel; this
        // path is used only for debugging (mostly Qwen3.5 kvf16 mode).
        return Err(hip_bridge::HipError::new(
            0,
            "gemma4 requires a quantized KV cache (asym2/asym3/asym4/q8); kvf16 lacks sliding-window support",
        ));
    }

    // o_proj → tmp (reuse tmp, overwriting input_layernorm output).
    weight_gemv(gpu, &lw.o_proj, &scratch.attn_out, &scratch.tmp)?;

    // Sandwich post-attn norm (in-place on tmp).
    gpu.rmsnorm_f32(&scratch.tmp, &lw.post_attention_layernorm, &scratch.tmp, config.norm_eps)?;

    // x = residual + tmp. (Reset x first since earlier ops mutated it.)
    gpu.hip.memcpy_dtod(&scratch.x.buf, &scratch.residual.buf, dim_bytes)?;
    gpu.add_inplace_f32(&scratch.x, &scratch.tmp)?;

    // residual = x (for the FFN residual stream).
    gpu.hip.memcpy_dtod(&scratch.residual.buf, &scratch.x.buf, dim_bytes)?;

    // Pre-FFN norm.
    gpu.rmsnorm_f32(&scratch.x, &lw.pre_feedforward_layernorm, &scratch.tmp, config.norm_eps)?;

    // SwiGLU(gelu_pytorch_tanh): gate_proj, up_proj, gelu_tanh(gate) * up → down_proj.
    weight_gemv(gpu, &lw.gate_proj, &scratch.tmp, &scratch.gate_ffn)?;
    weight_gemv(gpu, &lw.up_proj, &scratch.tmp, &scratch.up_ffn)?;
    gpu.gelu_tanh_f32(&scratch.gate_ffn, &scratch.ffn_hidden, config.hidden_dim)?;
    gpu.mul_f32(&scratch.ffn_hidden, &scratch.up_ffn, &scratch.ffn_hidden)?;
    weight_gemv(gpu, &lw.down_proj, &scratch.ffn_hidden, &scratch.ffn_out)?;

    // Sandwich post-FFN norm. On MoE layers (26B-A4B) this is folded into
    // apply_moe_branch (which adds the parallel MoE branch + sandwich norms
    // 1 and 2 before this outer norm); on dense layers we just call the
    // standalone post_feedforward_layernorm.
    let moe_bypass = std::env::var("HIPFIRE_MOE_BYPASS").ok().as_deref() == Some("1");
    match (lw.moe.as_ref(), moe_bypass) {
        (Some(moe), false) => apply_moe_branch(gpu, config, scratch, moe,
            &lw.post_feedforward_layernorm, &scratch.residual)?,
        _ => gpu.rmsnorm_f32(&scratch.ffn_out, &lw.post_feedforward_layernorm,
            &scratch.tmp, config.norm_eps)?,
    }

    // x = residual + tmp (again, reset x from saved residual).
    gpu.hip.memcpy_dtod(&scratch.x.buf, &scratch.residual.buf, dim_bytes)?;
    gpu.add_inplace_f32(&scratch.x, &scratch.tmp)?;

    // Learned per-layer scalar multiplier.
    gpu.scale_f32(&scratch.x, lw.layer_scalar_host)?;

    Ok(())
}

/// Single full (global) attention layer with K=V weight sharing.
///
/// Key differences from sliding:
///   • head_dim = 512 (global_head_dim), 4 KV heads (vs sliding's 256 / 16).
///   • V is the *pre*-k_norm output of k_proj — CRITICAL ordering (line 1214
///     of modeling_gemma4.py). In Python:
///         key_states = k_proj(x)
///         value_states = v_proj(x) if v_proj else key_states   # bound BEFORE norm
///         key_states   = k_norm(key_states)                    # rebind, value_states holds pre-norm
///         value_states = v_norm(value_states)
///     Our translation: write k_proj output into `scratch.k`, memcpy into
///     `scratch.v`, then apply k_norm in-place on scratch.k.
///   • v_norm is `no_scale=true` RMSNorm — divide only, no learned gain.
///     We call the existing `rmsnorm_batched` with the ones-filled
///     `scratch.v_norm_ones_full` as the weight vector.
///   • RoPE is partial_rotary_factor=0.25 proportional:
///     pairs (i, i+head_dim/2) for i in [0, 64) rotate with theta=1e6;
///     pairs [64, 256) are NoPE (identity). See `rope_partial_halved_f32`.
///   • No sliding window (window_size=0 = full causal).
///   • Attention scale = 1.0 (same as sliding — Gemma 4 sets
///     `self.scaling = 1.0`; we compensate by pre-scaling Q by sqrt(head_dim)).
fn full_layer_decode(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    lw: &FullLayerWeights,
    pos: usize,
    kv_cache: &mut hipfire_runtime::llama::KvCache,
    kv_layer_idx: usize,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    let dim = config.dim;
    let head_dim = config.full_head_dim;
    let n_heads = config.n_heads;
    let n_kv = config.full_n_kv_heads;
    let dim_bytes = dim * 4;
    let kv_bytes = n_kv * head_dim * 4;

    // residual = x
    gpu.hip.memcpy_dtod(&scratch.residual.buf, &scratch.x.buf, dim_bytes)?;

    // tmp = input_layernorm(x)
    gpu.rmsnorm_f32(&scratch.x, &lw.input_layernorm, &scratch.tmp, config.norm_eps)?;

    // Q + K projections. V is derived from K's pre-norm output below.
    weight_gemv(gpu, &lw.q_proj, &scratch.tmp, &scratch.q)?;
    weight_gemv(gpu, &lw.k_proj, &scratch.tmp, &scratch.k)?;

    // CRITICAL: capture pre-k_norm bytes as V before applying k_norm.
    gpu.hip.memcpy_dtod(&scratch.v.buf, &scratch.k.buf, kv_bytes)?;

    // q_norm, k_norm, and no-scale v_norm (all head_dim = 512).
    gpu.rmsnorm_batched(&scratch.q, &lw.q_norm, &scratch.q, n_heads, head_dim, config.norm_eps)?;
    gpu.rmsnorm_batched(&scratch.k, &lw.k_norm, &scratch.k, n_kv, head_dim, config.norm_eps)?;
    gpu.rmsnorm_batched(&scratch.v, &scratch.v_norm_ones_full, &scratch.v,
        n_kv, head_dim, config.norm_eps)?;

    // Pre-scale Q by sqrt(head_dim=512) so the flash kernel's 1/sqrt(head_dim)
    // cancels (Gemma 4 attention scaling is 1.0).
    gpu.scale_f32(&scratch.q, (head_dim as f32).sqrt())?;

    // Proportional RoPE: rotate_half of the first 64 pairs of every 512-dim head.
    let n_rot_pairs = ((head_dim as f32) * config.full_partial_rotary_factor * 0.5) as usize;
    gpu.rope_partial_halved_f32(&scratch.q, &scratch.k, &scratch.pos_buf,
        n_heads, n_kv, head_dim, n_rot_pairs, config.full_rope_theta)?;

    // KV cache write + attention. Full-attn layers (head_dim=512) route to
    // the asym3 hd=512 kernels (origin/gemma4 6f5cb8b + f724be6, ported in
    // D2.5). asym2/asym4/q8 hd=512 siblings are not yet ported — those modes
    // hard-fail here so users hit a loud, specific error rather than silent
    // truncation. window_size=0 = full causal (no sliding on global layers).
    if kv_cache.quant_asym3 {
        let ct = kv_cache.givens_cos.as_ref().unwrap();
        let st = kv_cache.givens_sin.as_ref().unwrap();
        gpu.kv_cache_write_asym3_fused(
            &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.k, &scratch.v, &scratch.pos_buf, ct, st, n_kv, head_dim, 0)?;
        gpu.attention_flash_asym3_window(
            &scratch.q, &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.attn_out, &scratch.pos_buf, ct, st, pos + 1,
            n_heads, n_kv, head_dim, kv_cache.max_seq,
            &scratch.flash_partials,
            0, // window_size: full causal,
            0,
        )?;
    } else if kv_cache.quant_asym4 || kv_cache.quant_asym2 || kv_cache.quant_q8 {
        let mode = if kv_cache.quant_asym4 { "asym4" }
                   else if kv_cache.quant_asym2 { "asym2" }
                   else { "q8" };
        return Err(hip_bridge::HipError::new(
            0,
            &format!("gemma4 full-attn layer (hd=512): kv-mode={} not yet ported. \
                     Use --kv-mode asym3 or fp32. Tracked in spec doc as port-blocked \
                     on missing asym2/asym4/q8 hd=512 kernels.", mode),
        ));
    } else {
        // FP32 KV path (kvf16 / kvfp32). attention_f32 bakes in
        // scale=1/sqrt(head_dim); the pre-scale of Q above cancels it, giving
        // the Gemma 4 scale=1.0 semantics.
        let kv_dim = n_kv * head_dim;
        gpu.kv_cache_write(&kv_cache.k_gpu[kv_layer_idx], &scratch.k, &scratch.pos_buf, kv_dim)?;
        gpu.kv_cache_write(&kv_cache.v_gpu[kv_layer_idx], &scratch.v, &scratch.pos_buf, kv_dim)?;
        gpu.attention_f32(
            &scratch.q, &kv_cache.k_gpu[kv_layer_idx], &kv_cache.v_gpu[kv_layer_idx],
            &scratch.attn_out, &scratch.pos_buf, pos + 1,
            n_heads, n_kv, head_dim, kv_cache.max_seq,
        )?;
    }

    // o_proj → tmp.
    weight_gemv(gpu, &lw.o_proj, &scratch.attn_out, &scratch.tmp)?;

    // Sandwich post-attn norm.
    gpu.rmsnorm_f32(&scratch.tmp, &lw.post_attention_layernorm, &scratch.tmp, config.norm_eps)?;

    // x = residual + tmp.
    gpu.hip.memcpy_dtod(&scratch.x.buf, &scratch.residual.buf, dim_bytes)?;
    gpu.add_inplace_f32(&scratch.x, &scratch.tmp)?;

    // Save new residual.
    gpu.hip.memcpy_dtod(&scratch.residual.buf, &scratch.x.buf, dim_bytes)?;

    // Pre-FFN norm.
    gpu.rmsnorm_f32(&scratch.x, &lw.pre_feedforward_layernorm, &scratch.tmp, config.norm_eps)?;

    // SwiGLU with gelu_pytorch_tanh activation.
    weight_gemv(gpu, &lw.gate_proj, &scratch.tmp, &scratch.gate_ffn)?;
    weight_gemv(gpu, &lw.up_proj, &scratch.tmp, &scratch.up_ffn)?;
    gpu.gelu_tanh_f32(&scratch.gate_ffn, &scratch.ffn_hidden, config.hidden_dim)?;
    gpu.mul_f32(&scratch.ffn_hidden, &scratch.up_ffn, &scratch.ffn_hidden)?;
    weight_gemv(gpu, &lw.down_proj, &scratch.ffn_hidden, &scratch.ffn_out)?;

    // Sandwich post-FFN norm. Same MoE dispatch as sliding_layer_decode.
    let moe_bypass = std::env::var("HIPFIRE_MOE_BYPASS").ok().as_deref() == Some("1");
    match (lw.moe.as_ref(), moe_bypass) {
        (Some(moe), false) => apply_moe_branch(gpu, config, scratch, moe,
            &lw.post_feedforward_layernorm, &scratch.residual)?,
        _ => gpu.rmsnorm_f32(&scratch.ffn_out, &lw.post_feedforward_layernorm,
            &scratch.tmp, config.norm_eps)?,
    }

    // x = residual + tmp.
    gpu.hip.memcpy_dtod(&scratch.x.buf, &scratch.residual.buf, dim_bytes)?;
    gpu.add_inplace_f32(&scratch.x, &scratch.tmp)?;

    // Learned per-layer scalar multiplier.
    gpu.scale_f32(&scratch.x, lw.layer_scalar_host)?;

    Ok(())
}

/// Batched prefill. Phase 4.
pub fn forward_prefill_batch(
    _gpu: &mut Gpu,
    _weights: &Gemma4Weights,
    _config: &Gemma4Config,
    _tokens: &[u32],
    _start_pos: usize,
    _kv_sliding: &mut hipfire_runtime::llama::KvCache,
    _kv_full: &mut hipfire_runtime::llama::KvCache,
    _scratch: &Gemma4Scratch,
) -> HipResult<()> {
    Err(hip_bridge::HipError::new(0, "gemma4::forward_prefill_batch not implemented (Phase 4)"))
}
