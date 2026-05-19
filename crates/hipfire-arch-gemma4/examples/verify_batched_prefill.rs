//! Compare per-token `weight_gemv` against batched `weight_gemm` for the
//! MQ4G256 fast path that caused the 2026-05-19 batched-prefill regression
//! (bisect landed on `521161f8`).
//!
//! This is exactly the cross-shape NRMSE check that would have caught the
//! bug at commit time. Adding it to the Phase 2 verifier (or a CI smoke)
//! is one of the recommended preventive measures in
//! `docs/investigations/2026-05-19-gemma4-128k-branch-evidence-debt.md`.
//!
//! Usage:
//!   verify_batched_prefill <model.mq4> [--batch N] [--tensor NAME]
//!
//! Picks one MQ4G256 weight tensor from the model (default: layer 0's q_proj),
//! generates random x of shape [N, K], runs:
//!   y_pertoken[i] = weight_gemv(W, x[i])           for i in 0..N
//!   y_batched     = weight_gemm(W, x, batch=N)
//! and prints per-batch NRMSE. PASS if all NRMSE < 1e-3 (FP32-vs-FP32 noise
//! envelope; the dispatch paths differ only in batching, not in math).

fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::{weight_gemv, weight_gemm};
    use hipfire_arch_gemma4::gemma4;
    use rdna_compute::DType;
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: verify_batched_prefill <model.mq4> [--batch N] [--tensor NAME_SUBSTRING]");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let mut batch_size: usize = 8;
    let mut tensor_filter = String::from("layers.0.self_attn.q_proj");
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--batch" => { batch_size = args[i + 1].parse().unwrap(); i += 2; }
            "--tensor" => { tensor_filter = args[i + 1].clone(); i += 2; }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(1); }
        }
    }

    eprintln!("=== verify_batched_prefill ===");
    eprintln!("Model: {model_path}");
    eprintln!("Filter: {tensor_filter}");
    eprintln!("Batch: {batch_size}");

    let mut hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    assert_eq!(hfq.arch_id, 7, "expected arch_id=7 (Gemma 4)");
    let config = gemma4::config_from_hfq(&hfq).expect("read config");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("GPU: {}", gpu.arch);
    let weights = gemma4::load_weights(&mut hfq, &config, &mut gpu).expect("load weights");

    // Pick a Sliding layer 0 q_proj as the test weight — that's MQ4G256
    // (auto-promoted from MG4G256 at arch_id=7). Bail loud if the layer's
    // weight isn't MQ4G256: this verifier is specific to that fast path.
    let lw = match &weights.layers[0] {
        gemma4::LayerWeights::Sliding(lw) => lw,
        _ => { eprintln!("layer 0 is not Sliding; rerun against a model whose layer 0 is sliding"); std::process::exit(1); }
    };
    let w = &lw.q_proj;
    if w.gpu_dtype != DType::MQ4G256 {
        eprintln!("layer 0 q_proj is {:?}, expected MQ4G256 — this test only validates the MQ4 fast path", w.gpu_dtype);
        std::process::exit(1);
    }
    let m = w.m;
    let k = w.k;
    eprintln!("Tensor: layer 0 q_proj  m={m}  k={k}  dtype={:?}", w.gpu_dtype);

    // Generate deterministic-but-non-trivial input x[N, K]. Seed is fixed
    // so cross-run comparison is byte-identical without RNG dependencies.
    let mut x_host: Vec<f32> = Vec::with_capacity(batch_size * k);
    for i in 0..(batch_size * k) {
        // Cheap LCG. Range [-1, 1).
        let s = (i.wrapping_mul(2654435761) ^ 0xdeadbeef) as u32;
        x_host.push((s as f32 / u32::MAX as f32) * 2.0 - 1.0);
    }
    let x_dev = gpu.alloc_tensor(&[batch_size, k], DType::F32).expect("x alloc");
    // bytemuck-free f32 → byte slice (host x has no padding, stride = sizeof f32).
    let x_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(x_host.as_ptr() as *const u8, x_host.len() * 4)
    };
    gpu.hip.memcpy_htod(&x_dev.buf, x_bytes).expect("x upload");

    // Per-token reference: y_pertoken[i] = weight_gemv(w, x[i])
    let y_pertoken_dev = gpu.alloc_tensor(&[batch_size, m], DType::F32).expect("y_pertoken alloc");
    let x_tok = gpu.alloc_tensor(&[k], DType::F32).expect("x_tok alloc");
    let y_tok = gpu.alloc_tensor(&[m], DType::F32).expect("y_tok alloc");
    for b in 0..batch_size {
        gpu.hip.memcpy_dtod_at(&x_tok.buf, 0, &x_dev.buf, b * k * 4, k * 4).expect("x_tok copy");
        weight_gemv(&mut gpu, w, &x_tok, &y_tok).expect("gemv");
        gpu.hip.memcpy_dtod_at(&y_pertoken_dev.buf, b * m * 4, &y_tok.buf, 0, m * 4).expect("y_tok writeback");
    }

    // Batched: y_batched = weight_gemm(w, x, batch_size)
    let y_batched_dev = gpu.alloc_tensor(&[batch_size, m], DType::F32).expect("y_batched alloc");
    weight_gemm(&mut gpu, w, &x_dev, &y_batched_dev, batch_size).expect("gemm");

    // Download + NRMSE per batch element.
    let y_pt = gpu.download_f32(&y_pertoken_dev).expect("pt dl");
    let y_bt = gpu.download_f32(&y_batched_dev).expect("bt dl");

    let mut max_nrmse = 0.0f32;
    let mut all_pass = true;
    for b in 0..batch_size {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        let mut max_abs_diff = 0.0f32;
        for j in 0..m {
            let pt = y_pt[b * m + j];
            let bt = y_bt[b * m + j];
            let diff = pt - bt;
            num += (diff as f64) * (diff as f64);
            den += (pt as f64) * (pt as f64);
            if diff.abs() > max_abs_diff { max_abs_diff = diff.abs(); }
        }
        let nrmse = if den > 0.0 { (num / den).sqrt() as f32 } else { num.sqrt() as f32 };
        if nrmse > max_nrmse { max_nrmse = nrmse; }
        let verdict = if nrmse < 1e-3 { "PASS" } else { all_pass = false; "FAIL" };
        eprintln!("  batch[{:>3}] nrmse={:.6e}  max|Δ|={:.4e}  {}", b, nrmse, max_abs_diff, verdict);
    }
    eprintln!();
    eprintln!("max_nrmse = {max_nrmse:.6e}");
    if all_pass {
        eprintln!("VERDICT: PASS  (all batch rows agree within 1e-3 NRMSE)");
        println!("verify_batched_prefill PASS  max_nrmse={max_nrmse:.6e}");
    } else {
        eprintln!("VERDICT: FAIL  (batched and per-token diverge — likely the bug behind 521161f8)");
        println!("verify_batched_prefill FAIL  max_nrmse={max_nrmse:.6e}");
        std::process::exit(1);
    }
}
