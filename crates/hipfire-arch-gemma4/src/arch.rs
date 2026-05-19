//! `Architecture` trait implementation for Gemma 4.
//!
//! Mirrors PR 11's llama / PR 8's qwen35 pattern: bring-up triple
//! (`config_from_hfq`, `load_weights`, `new_state`) routed through the
//! trait so the daemon can dispatch by `arch_id` without a `match`
//! ladder. Forward-pass calls stay direct on `gemma4::*`.
//!
//! Status (2026-05-07): forward path scaffold is present in
//! `gemma4.rs` but UNVALIDATED on real Gemma 4 weights. Per
//! `docs/investigations/2026-05-07-gemma4-arch-intake/arch-report.md`,
//! the remaining work is daemon dispatch, SPM-BPE tokenizer wiring,
//! quantizer surface, and a coherence-gate-passing forward.

use crate::gemma4::{self, Gemma4Config, Gemma4Scratch, Gemma4Weights};
use hipfire_runtime::arch::{Architecture, EosFilterOverrides, PromptFrameOverrides};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

/// Type marker for the Gemma 4 family. Covers `arch_id = 7` (the value
/// claimed by the Phase 1 scaffolding commit `b1b4afa`).
///
/// Released sub-variants (verified 2026-05-07 on huggingface.co/google):
///   - `google/gemma-4-31B` / `gemma-4-31B-it` (33B dense)
///   - `google/gemma-4-26B-A4B` / `gemma-4-26B-A4B-it` (27B MoE A4B)
///   - `google/gemma-4-E4B-it` (8B Any-to-Any "E" variant)
///   - `google/gemma-4-E2B-it` (5B Any-to-Any "E" variant)
///
/// The Phase 1 scaffold targeted `gemma-4-31B`; per-variant config
/// branches (MoE A4B, E-class) live in `gemma4::config_from_hfq` once
/// the actual configs are inspected with the new arch-intake tooling
/// in `scripts/arch-intake/`.
pub struct Gemma4;

impl Architecture for Gemma4 {
    type Weights = Gemma4Weights;
    type State = Gemma4Scratch;
    type Config = Gemma4Config;

    fn arch_id() -> u32 {
        // Reserved for the Gemma 4 family by the Phase 1 scaffolding
        // commit (`b1b4afa`). `docs/architecture-ids.md` (TODO) tracks
        // the full registry.
        7
    }

    fn name() -> &'static str {
        "gemma4"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        gemma4::config_from_hfq(hfq)
            .ok_or_else(|| "gemma4: failed to parse config from HFQ metadata".to_string())
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        gemma4::load_weights(hfq, cfg, gpu)
            .map_err(|e| format!("gemma4: load_weights failed: {e:?}"))
    }

    fn new_state(gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        // Gemma 4's scratch carries TWO KV caches (sliding + full) plus
        // the standard attention scratch. `_max_prefill` is internal —
        // the constructor reads `max_seq` off the env / config.
        Gemma4Scratch::new(gpu, cfg, 1)
            .map_err(|e| format!("gemma4: Gemma4Scratch::new failed: {e:?}"))
    }

    // ── Per-arch overrides ────────────────────────────────────────────
    //
    // Gemma 4 diverges from Qwen3.5 conventions in two ways the runtime
    // policy hooks can absorb. Both are TODO until the forward pass
    // validates end-to-end; they are documented here so the daemon
    // dispatcher knows to expect them.

    fn prompt_frame_overrides(_cfg: &Self::Config) -> PromptFrameOverrides {
        // NOTE: currently dead code — `daemon.rs` special-cases Gemma 4
        // via `gemma4_build_scaffold` and never consults the Architecture
        // trait for prompt framing. Kept for the eventual trait-dispatch
        // migration.
        //
        // The string literals below describe the documented
        // `<start_of_turn>` / `<end_of_turn>` framing, but the daemon
        // scaffold actually emits `<|turn>` (id 105) / `<turn|>` (id 106)
        // because those are the special tokens present in the shipped
        // `gemma-4-26b-a4b-it.mq4` tokenizer. A future trait-dispatched
        // frame builder must derive literals from the tokenizer's
        // special-token table at load time, not from hard-coded bytes.
        PromptFrameOverrides::default()
    }

    fn eos_filter_overrides(_cfg: &Self::Config) -> EosFilterOverrides {
        // NOTE: dead code (see prompt_frame_overrides above). The shipped
        // `gemma-4-26b-a4b-it.mq4` model closes turns with `<turn|>`
        // (id 106), and other Gemma 4 checkpoints may use `<end_of_turn>`
        // or other tokens. Hard-coding either set here would be wrong for
        // the other. Empty stop_at + a strip_think hint is the only
        // honest default until trait dispatch reads from the tokenizer's
        // special-token table at load time.
        EosFilterOverrides {
            stop_at: vec![],
            holdback_prefixes: vec![],
            strip_think: Some(false), // Gemma 4 is not a thinking-mode model
        }
    }
}
