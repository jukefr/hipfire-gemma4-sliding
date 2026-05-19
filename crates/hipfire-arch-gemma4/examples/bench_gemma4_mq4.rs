//! Focused perf benchmark for Gemma 4 MQ4 forward pass.
//!
//! Mirrors `bench_qwen35_mq4`'s shape so `scripts/speed-gate.sh` can call
//! either with the same flags. Separates prefill from generation, strips
//! kernel-JIT overhead via warmup, reports prefill_tok_s + decode_tok_s.
//!
//! Usage:
//!   bench_gemma4_mq4 <model.mq4> [--prefill N] [--gen N] [--warmup N]
//!
//! Env knobs:
//!   HIPFIRE_KV_MODE     — sliding KV cache mode (asym3 [default] | q8 | asym4 | asym2)
//!   HIPFIRE_FULL_KV     — full-layer KV cache mode (asym3 [default] | fp32)
//!   HIPFIRE_PREFILL_BATCH — 0 to disable batched-prefill v1 path (default on)
//!   HIPFIRE_KV_SEQ      — full-attention KV cap (default 4096, 128k = 131072)

fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_arch_gemma4::gemma4::{self, Gemma4Scratch};
    use hipfire_runtime::llama::KvCache;
    use std::path::Path;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: bench_gemma4_mq4 <model.mq4> [--prefill N] [--gen N] [--warmup N]");
        std::process::exit(1);
    }
    let model_path = &args[1];

    let mut prefill_len: usize = 32;
    let mut gen_len: usize = 100;
    let mut warmup_len: usize = 5;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--prefill" => { prefill_len = args[i + 1].parse().unwrap(); i += 2; }
            "--gen"     => { gen_len     = args[i + 1].parse().unwrap(); i += 2; }
            "--warmup"  => { warmup_len  = args[i + 1].parse().unwrap(); i += 2; }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(1); }
        }
    }

    eprintln!("=== bench_gemma4_mq4 ===");
    eprintln!("Model: {model_path}");
    eprintln!("Phases: prefill={prefill_len} warmup={warmup_len} gen={gen_len}");

    let mut hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    assert_eq!(hfq.arch_id, 7, "expected arch_id=7 (Gemma 4), got {}", hfq.arch_id);
    let config = gemma4::config_from_hfq(&hfq).expect("read config");
    let n_sliding = config.layer_types.iter()
        .filter(|&&t| t == gemma4::LayerType::Sliding).count();
    let n_full = config.layer_types.iter()
        .filter(|&&t| t == gemma4::LayerType::Full).count();
    eprintln!(
        "Config: dim={} layers={} ({} sliding + {} full) heads={} vocab={}",
        config.dim, config.n_layers, n_sliding, n_full, config.n_heads, config.vocab_size,
    );

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("GPU: {}", gpu.arch);
    let weights = gemma4::load_weights(&mut hfq, &config, &mut gpu).expect("load weights");

    let kv_seq = std::env::var("HIPFIRE_KV_SEQ")
        .ok().and_then(|v| v.parse().ok())
        .unwrap_or((prefill_len + warmup_len + gen_len + 16).max(4096));
    let kv_mode = std::env::var("HIPFIRE_KV_MODE").unwrap_or_else(|_| "asym3".to_string());
    let full_kv_mode = std::env::var("HIPFIRE_FULL_KV").unwrap_or_else(|_| "asym3".to_string());
    let sliding_kv_seq = if kv_mode == "asym3" { config.sliding_window } else { kv_seq };

    let mut kv_sliding = match kv_mode.as_str() {
        "asym4" => KvCache::new_gpu_asym4(&mut gpu, n_sliding, config.sliding_n_kv_heads, config.sliding_head_dim, sliding_kv_seq),
        "asym2" => KvCache::new_gpu_asym2(&mut gpu, n_sliding, config.sliding_n_kv_heads, config.sliding_head_dim, sliding_kv_seq),
        "q8"    => KvCache::new_gpu_q8(&mut gpu, n_sliding, config.sliding_n_kv_heads, config.sliding_head_dim, sliding_kv_seq),
        _       => KvCache::new_gpu_asym3(&mut gpu, n_sliding, config.sliding_n_kv_heads, config.sliding_head_dim, sliding_kv_seq),
    }.expect("kv sliding alloc");
    let mut kv_full = match full_kv_mode.as_str() {
        "fp32" => KvCache::new_gpu(&mut gpu, n_full, config.full_n_kv_heads, config.full_head_dim, kv_seq),
        _      => KvCache::new_gpu_asym3(&mut gpu, n_full, config.full_n_kv_heads, config.full_head_dim, kv_seq),
    }.expect("kv full alloc");

    let scratch = Gemma4Scratch::new(&mut gpu, &config, 128).expect("scratch alloc");
    gemma4::init_scratch_constants(&mut gpu, &scratch, config.full_head_dim)
        .expect("scratch constants init");

    // Synthetic prefill tokens — BOS + repeated filler that exists in any
    // Gemma 4 vocab. Avoid relying on a tokenizer pass; the bench is about
    // dispatch perf, not tokenization quality.
    let tokens: Vec<u32> = std::iter::once(config.bos_token as u32)
        .chain(std::iter::repeat(100u32).take(prefill_len.saturating_sub(1)))
        .collect();
    assert_eq!(tokens.len(), prefill_len);

    // Warmup — drives kernel JIT + cache warming. Throws away timing.
    eprintln!("Warmup ({warmup_len} fwd passes)...");
    for pos in 0..warmup_len {
        let tok = tokens.get(pos).copied().unwrap_or(100);
        gemma4::forward_scratch(&mut gpu, &weights, &config, tok, pos,
            &mut kv_sliding, &mut kv_full, &scratch).expect("warmup forward");
    }
    // Reset KV state by reallocating — easier than tracking the warmup positions.
    let mut kv_sliding = match kv_mode.as_str() {
        "asym4" => KvCache::new_gpu_asym4(&mut gpu, n_sliding, config.sliding_n_kv_heads, config.sliding_head_dim, sliding_kv_seq),
        "asym2" => KvCache::new_gpu_asym2(&mut gpu, n_sliding, config.sliding_n_kv_heads, config.sliding_head_dim, sliding_kv_seq),
        "q8"    => KvCache::new_gpu_q8(&mut gpu, n_sliding, config.sliding_n_kv_heads, config.sliding_head_dim, sliding_kv_seq),
        _       => KvCache::new_gpu_asym3(&mut gpu, n_sliding, config.sliding_n_kv_heads, config.sliding_head_dim, sliding_kv_seq),
    }.expect("kv sliding realloc");
    let mut kv_full = match full_kv_mode.as_str() {
        "fp32" => KvCache::new_gpu(&mut gpu, n_full, config.full_n_kv_heads, config.full_head_dim, kv_seq),
        _      => KvCache::new_gpu_asym3(&mut gpu, n_full, config.full_n_kv_heads, config.full_head_dim, kv_seq),
    }.expect("kv full realloc");

    // Prefill phase — per-token forward_scratch loop. Mirrors what the daemon
    // does in the per-token prefill path (HIPFIRE_PREFILL_BATCH=0). The batched
    // prefill path (gemma4::forward_prefill_batch) is what the daemon uses by
    // default; benching that requires going through daemon.rs.
    eprintln!("Prefill ({prefill_len} tokens)...");
    let t_prefill = Instant::now();
    for (pos, &tok) in tokens.iter().enumerate() {
        gemma4::forward_scratch(&mut gpu, &weights, &config, tok, pos,
            &mut kv_sliding, &mut kv_full, &scratch).expect("prefill forward");
    }
    let prefill_secs = t_prefill.elapsed().as_secs_f64();
    let prefill_tok_s = prefill_len as f64 / prefill_secs;
    eprintln!("Prefill: {prefill_len} tok in {prefill_secs:.3}s  →  prefill_tok_s={prefill_tok_s:.1}");

    // Decode phase — greedy argmax for gen_len tokens.
    eprintln!("Decode ({gen_len} tokens)...");
    let logits_buf = vec![0.0f32; config.vocab_size];
    let mut last_tok = *tokens.last().unwrap();
    let t_decode = Instant::now();
    for step in 0..gen_len {
        let pos = prefill_len + step;
        gemma4::forward_scratch(&mut gpu, &weights, &config, last_tok, pos,
            &mut kv_sliding, &mut kv_full, &scratch).expect("decode forward");
        // Greedy argmax — read logits + pick max. The download is on the
        // critical path for AR decode, same as the daemon.
        let logits = gpu.download_f32(&scratch.logits).expect("download logits");
        let (argmax, _) = logits.iter().enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) });
        last_tok = argmax as u32;
        let _ = &logits_buf; // suppress unused-mut
    }
    let decode_secs = t_decode.elapsed().as_secs_f64();
    let decode_tok_s = gen_len as f64 / decode_secs;
    eprintln!("Decode:  {gen_len} tok in {decode_secs:.3}s   →  decode_tok_s={decode_tok_s:.1}");

    // Speed-gate-friendly summary line — single grep-able row at end.
    println!("prefill_tok_s={prefill_tok_s:.2} decode_tok_s={decode_tok_s:.2} prefill={prefill_len} gen={gen_len} kv_mode={kv_mode} full_kv={full_kv_mode}");
}
