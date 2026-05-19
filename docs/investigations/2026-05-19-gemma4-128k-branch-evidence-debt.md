# Gemma 4 128k ring-buffer branch — evidence debt

**Date:** 2026-05-19
**Branch:** `gemma4-128k-ring-buffer`
**Scope:** 54 commits since `master`, spanning the arch-port (605374dd…7bd2f43a),
the asym3 hd=512 + ring-buffer KV path, indexed-MoE dispatch, the
prefill batching ladder (60 → 132 tok/s), Phase B routing-bucketed MoE
(opt-in null), and Phase C hipGraph root-cause.

This document records what skill-mandated evidence the branch ships
*without*, so the gaps are visible to future work instead of being
re-discovered.

## Process violations corrected on this pass

| Gap | Status |
|-----|--------|
| `core.hooksPath` was `.git/hooks`, not `.githooks` — coherence-gate + speed-gate never fired automatically on any of the 54 commits | Fixed: `git config core.hooksPath .githooks` |
| Coherence-gate matrix had no Gemma 4 row → branch's main model was never gated | Fixed: added `gemma-4-26b-a4b-it.mq4|gemma4-cap` row to `SHORT_TESTS` in `scripts/coherence-gate.sh` |
| Branch shipped without an Astrea engine fingerprint or model inspection | Fixed: `.codeinsight+research/astrea/gemma4-26b-a4b/{inspect,fingerprint,kv-profile-asym3}.json` |
| **Gemma 4 batched prefill broken end-to-end** — coherence gate caught structural-loop attractor on the canonical "Capital of France?" prompt the moment the row was wired up | Fixed: flipped `HIPFIRE_PREFILL_BATCH` from opt-out to opt-in in `daemon.rs:3517`. Root cause not identified — see "Active regression" below |

## Active regression — Gemma 4 batched prefill

**Symptom.** Default-path daemon output on "What is the capital of France? Answer in one short sentence." (temperature=0, repeat_penalty=1.05):

| Path | prefill_tok_s | Output |
|------|---------------|--------|
| Batched prefill (default before fix) | 131.8 | `Jones,4. The fact is my *1. ### (It's a matter of fact! I'll take it with me,;...;...;...` |
| Per-token prefill (`HIPFIRE_PREFILL_BATCH=0`) | 78.6 | `<|channel>thought\n<channel|>The capital of France is Paris.` |

Same binary, same prompt, same model md5, same flags otherwise. The signature (early-text drift then `;...;...` sentinel-loop) is the classic single-token attractor — same shape the DFlash coherence gate tier 1/2 hard-fails on (`unique_token_ratio < 0.15`, `max_single_token_frequency > 0.50`).

**What's masked.** The +38% (`d2bb8573`) and +55% (`521161f8`) prefill claims that put Gemma 4 at 132 tok/s prefill — those numbers are real, but the path that produces them produces wrong tokens. Every speed-ladder commit since `d2bb8573` validated against the prompt `"Capital of France?"` qualitatively, and "Paris" appeared somewhere in the output for those checks, but a fuller prompt + `repeat_penalty=1.05` flips the path into the attractor.

**STATUS 2026-05-19 evening: ROOT-CAUSED AND FIXED.**

The v2 batched-prefill regression has been root-caused and patched in `kernels/src/gemm_hfq4g128.hip`. The bug was a missed port of the partial-trailing-group fix that `gemv_hfq4g128.hip` got when Gemma 4 landed: the batched GEMM used `groups_per_row = K / 128` (floor) where the per-token GEMV uses `(K + 127) / 128` (ceil). At K=2112 (Gemma 4 26B-A4B-it dense down_proj input dim), floor=16 vs needed 17 — the batched kernel silently dropped 64 input dims per dot product, producing the downstream attractor.

`forward_prefill_batch` now routes to v2 by default. **Measured perf with the fix**:

| Row | Prefill tok/s | Decode tok/s |
|-----|---------------|--------------|
| gemma4-cap (23 prefill tokens) | **180.5** | 71.8 |
| gemma4-longctx (1270 prefill tokens, past 1024 sliding cap) | **136.2** | 59.9 |

The 521161f8 commit's claimed +55% (132 tok/s) was real; with the fix landed we beat it by 36% — 180 tok/s. The bug was masked at commit time because the gemv vs gemm ports happened on different days and the smaller "Capital of France?" qualitative check happened to produce "Paris" before the attractor compounded.

**Diagnostic that landed alongside the fix:**

- `crates/hipfire-arch-gemma4/src/gemma4.rs` — `dbg_dump` helper (env-gated by `HIPFIRE_GEMMA4_DUMP=1`) plus matching dump calls in both v1's `sliding_layer_decode_impl` and v2's sliding-layer block. Walks step-by-step from `pb_residual[0]` through `down_proj`, prints `sum`, `head`, `nan/inf` per step. Zero cost when env not set. Future v1-vs-v2-style divergence isolations should use this pattern.

---

**Bisect result (2026-05-19 morning): `521161f8` is the regressing commit.**

Walked d2bb8573 → 38e62760 → 521161f8 with `HIPFIRE_PREFILL_BATCH=1` and the predicate "output must contain 'Paris'":

| Commit | Result | Notes |
|--------|--------|-------|
| d2bb8573 (claimed good in commit msg) | PASS | `<\|channel>thought\n<channel\|>The capital of France is Paris.` |
| 38e62760 (MAX_PREFILL_BATCH 64→128) | PASS | Same clean output |
| 521161f8 (batched dense projections +55%) | **FAIL** | Same attractor signature as HEAD |

`521161f8` introduces `forward_prefill_batch_v2` (replaces the per-token attn+FFN flow with batched dense projections) and adds the `MQ4G256` fast path to `weight_gemm` via `rotate_x_mq_batched → gemm_hfq4g256`. The +55% prefill speedup is real, but the new path corrupts KV state somewhere. Per-token (`HIPFIRE_PREFILL_BATCH=0`) on the same binary stays clean.

**Suspect mechanisms inside 521161f8 (ranked after diagnostic isolation 2026-05-19):**

A new diagnostic example `crates/hipfire-arch-gemma4/examples/verify_batched_prefill.rs` shipped this pass. It compares per-token `weight_gemv` vs batched `weight_gemm` on layer 0's MQ4G256 q_proj at batch=8:

```
max_nrmse = 1.205368e-7  → PASS
```

That **rules out** the MQ4G256 fast path (rotate_x_mq_batched → gemm_hfq4g256) as the bug. It also rules out the BATCH_TILE=8 partial-tile hypothesis. The math at the dense-projection layer is correct to FP32 noise floor.

**Remaining suspects in v2** (the bug is here somewhere):

1. **`rmsnorm_batched` on `pb_q` / `pb_k` / `pb_v` with `n_batch × n_heads` rows.** The kernel must walk packed `[n_batch, n_heads, head_dim]` correctly. If it instead assumes strided-by-q_dim_max layout, the second-and-later tokens normalize against the wrong slice. Diagnostic: run `rmsnorm_batched` on a known packed input at both n_batch=1 and n_batch=2 and compare to a CPU reference.
2. **`rope_batched_f32` vs per-token `rope_f32` semantics.** The kernels look mathematically equivalent (offsets/freqs identical), but the batched kernel reads `positions[b]` per batch row — verify the `pb_positions` upload writes the right `[start_pos, start_pos+1, …]` array, not stale data.
3. **The per-token attention dispatch loop inside `forward_prefill_batch_v2`.** Each token does memcpy_dtod_at out of pb_q/pb_k/pb_v at offset `i * q_dim_bytes` (packed assumption) into scratch.q/k/v, runs `kv_cache_write_asym3_fused + attention_flash_asym3_window`, copies attn_out back. If `gpu.active_stream` is set and the memcpy is async, ordering vs the kernel launches may not serialize.
4. **`apply_moe_branch_batched`'s `scratch.pb_moe_cur_moe` zeroing.** v2 enters the MoE branch on every layer. The accumulator buffer zeroing (line ~1493: `gpu.hip.memset(...)`) needs to complete before the indexed-down atomicAdd kernels read it.
5. **Full-attention layer "V from K" copy** (full-layer batched path, diff line 271-279): K is copied to V before k_norm. If the copy is async and k_norm fires before the copy completes, V gets garbage. Same race shape as suspect 4.

**Recommended next debug step:** instrument `forward_prefill_batch_v2` with `gpu.synchronize()` (or stream sync) calls between each step and re-run the gate. Whichever step's sync makes the output go from FAIL → PASS identifies the bug.

The bisect predicate is committed (`scripts/coherence-gate.sh`'s `gemma4-cap` row + the new `gemma4-longctx` row). A future contributor can verify any candidate v2 fix via:

```bash
# Temporarily flip forward_prefill_batch() in gemma4.rs to route to v2:
#   forward_prefill_batch_v2(...)  // not v1
HIPFIRE_PREFILL_BATCH=1 ./scripts/coherence-gate.sh   # both gemma4 rows must PASS

# And rerun the cross-shape NRMSE check at multiple batch sizes:
target/release/examples/verify_batched_prefill ~/.hipfire/models/gemma-4-26b-a4b-it.mq4 --batch 8
target/release/examples/verify_batched_prefill ~/.hipfire/models/gemma-4-26b-a4b-it.mq4 --batch 23
target/release/examples/verify_batched_prefill ~/.hipfire/models/gemma-4-26b-a4b-it.mq4 --batch 128
```

**Current workaround (shipped 2026-05-19):** `gemma4::forward_prefill_batch` routes to v1 (`forward_prefill_batch_v1`, the d2bb8573 known-good path: per-token attn+FFN + batched MoE). v2 stays in tree as `#[allow(dead_code)]` for future bisect work. Measured perf on the workaround: **prefill 109 tok/s (vs 132 broken v2, 79 per-token), decode 71 tok/s**. The +38% prefill win over per-token is preserved; the additional +55% from v2 is forfeited until the regression is fixed.

**Why the gate didn't catch this earlier.** `.git/hooks/pre-commit` was the active hook directory; `.githooks/pre-commit` was never on `core.hooksPath`. The hook would have refused every speed-ladder commit until the prompt produced "Paris" again.

## Additional gaps from second-pass review (all 6 hipfire skills loaded)

### Tester-contract md5 discipline missing on every perf claim

Per `CLAUDE.md` / `hipfire-tester`: "Treat benchmark numbers as reportable only with fresh-process runs, prompt md5, binary md5, and decoded-output eyeball checks." Audit of the 8 perf-claim commits on this branch:

| Commit | tok/s claim | prompt md5? | binary md5? |
|--------|-------------|-------------|-------------|
| 6813cab8 | 44 → 59 prefill | no | no |
| 21a2cedd | 44 → 56 / 36 → 57 | no | no |
| 5012a0cb | scaffolding | n/a | no |
| d2bb8573 | 60 → 84 prefill | no | no |
| 38e62760 | 83.8 → 85.1 | no | no |
| 521161f8 | 85 → 132 prefill | no | no |
| 8b0111c7 | ~132 (noise) | no | no |
| 959fb0a5 | 132.0 → 124.5 (-5.7%) | no | no |
| 6d3a44b9 | 132.0 → 132.9 | no | no |

Per `CLAUDE.md`: "agent-to-agent perf claims that lack prompt md5 are unverifiable." Every perf number in this branch's commit log is currently unreproducible by anyone without bench-shell access. The fix is mechanical: future commits should cite the prompt file path + `md5sum` and the binary md5 (`md5sum target/release/examples/daemon`) in the commit body.

### Speed-gate has no Gemma 4 coverage

`scripts/speed-gate.sh` is bound to `bench_qwen35_mq4` and exercises Qwen 3.5 model sizes only. None of this branch's Gemma 4 prefill claims would have been caught by the speed-gate even if the hook had been installed. Adding a `bench_gemma4_mq4` example mirroring `bench_qwen35_mq4` (or extending `bench_qwen35_mq4` to dispatch on arch_id) is the right close.

`tests/speed-baselines/*.txt` carries no Gemma 4 rows on any arch. The first `bench_gemma4_mq4` run on gfx1201 should be committed as the baseline.

Minor: `scripts/speed-gate.sh:82` defaults `MODELS_DIR` to `/home/kaden/ClaudeCode/autorocm/hipfire/models` — stale path from another contributor. The fallback to `$HOME/.hipfire/models` works, but the default should be the HOME path.

### Architecture trait override methods are unreachable

`crates/hipfire-arch-gemma4/src/arch.rs` defines `prompt_frame_overrides` and `eos_filter_overrides`, but a workspace grep shows the runtime never calls either method (only the trait definition site references them). Same pattern in Qwen35 / VL / Llama — the override methods are aspirational for a future trait-dispatch refactor.

Bigger issue on Gemma 4 specifically: the literals encoded in those (unused) methods are `<start_of_turn>` / `<end_of_turn>`, but the model this branch actually ships against (`gemma-4-26b-a4b-it.mq4`) uses `<|turn>` (id 105) and `<turn|>` (id 106) per the special-token table inside the .mq4 file. A future contributor wiring trait dispatch by following the trait comments would silently break decoding on the shipped model.

Fixed on this pass: updated the trait-impl comments in `arch.rs` to flag both the dead-code status and the literal mismatch versus the shipped tokenizer.

### Left-on-table: no HFQ4G128 WMMA sibling

The `gfx12 WMMA pattern` reference (`hipfire-arch-port` skill: commit `6924f2a`) covers HFQ4G256 (`gemm_qkv_hfq4g256_wmma.gfx12.hip`). The new `gemv_hfq4g128_moe_down_*` family added this branch is GEMV-shaped only. For Gemma 4 26B-A4B-it on gfx1201, `down_proj` (HFQ4G128) prefill goes through the wave32 GEMV path even with batching. A `gemm_hfq4g128_*_wmma.gfx12.hip` sibling would likely pull another 2–4× on the down-projection prefill share. Out of scope for this branch but worth a tracking issue.

## Open evidence debt (not corrected — needs hardware iteration)

### 1. Quality lane for MG4G256

`cb0a2b62` introduces MG4G256 (quant_type=19) and auto-promotes on
`arch_id=7`. Layout is asserted byte-identical to HFQ4G256 / MQ4G256
modulo FWHT pre-rotation, and per-kernel NRMSE is validated via the
`verify_against_torch` Phase 2 battery (770/770 PASS at the bf16 floor
in d2.5). What is *not* shipped is the Astrea quality lane: no `inspect
--imatrix`, no `plan` / `calibrate` artifact, no `eval` with KLD/PPL vs
a bf16 reference, no `metrics` row. Per `docs/methodology/astrea-model-policy.md`
core rule: "Do not claim a quant candidate is better without measured
quality evidence." MG4G256 is currently a relayout-of-MQ4G256, but
that's an *engineering* claim, not a quality one.

**To close:** run `python3 scripts/astrea.py plan --model
~/.hipfire/models/gemma-4-26b-a4b-it.mq4 --format mq4 --method
imatrix-scale --imatrix <path> --pretty --out .codeinsight+research/astrea/gemma4-26b-a4b/plan-imatrix.json`
then `eval` against a bf16 reference, then `metrics`.

### 2. KV policy artifact for ring-buffer hd=512

The asym3 ring-buffer sliding-window path with `cache_capacity <
kv_max_seq` is the load-bearing change of this branch (commits b40bb4f0,
be6b93e6, 7740afd5, 26fd2b3b, 9be6ecff, 36cdfde0). Per `astrea`'s
`kv-profile` step: "Run `kv-profile` when a candidate changes KV-cache
behavior or when a model should carry an embedded KV policy."

Captured on this pass:
`.codeinsight+research/astrea/gemma4-26b-a4b/kv-profile-asym3.json`
records the baseline policy shape but does not yet describe the
ring-buffer variant as a distinct mode. Astrea's CLI only enumerates
`q8|asym3|triattn|cask|turbo3|rotor`; "asym3 + ring buffer +
cache_capacity tracking" needs to either be expressed via the
`asym3` baseline + a model-side metadata flag, or as a new mode tag.
Decision deferred — flagged as a `kv-profile` schema follow-up.

### 3. Atlas AR rows for prefill perf claims

The branch ships five distinct prefill perf claims:

| Commit | Claim | Bench |
|--------|-------|-------|
| 6813cab8 | 44 → 59 tok/s prefill (+34%, async memset/memcpy + graph wiring) | median of 3 |
| 21a2cedd | 44 → 56 tok/s prefill, 36 → 57 decode (indexed MoE) | median of 3 |
| d2bb8573 | 60 → 84 tok/s prefill (+38%, token-batched wiring) | median of 3 |
| 521161f8 | 85 → 132 tok/s prefill (+55%, batched dense projections) | median of 3 |
| 6d3a44b9 | bucketed v2 132.0 → 132.9 (within noise) — confirms v1's −5.7% is closed | median of 3 |

The numbers were collected with fresh-prompt curl runs against the
running daemon — that's `docs/methodology/perf-benchmarking.md`'s
fresh-process protocol, which is correct. What's missing per
`docs/methodology/kernel-atlas.md` is the joined ISA Fit View:

```bash
python3 scripts/kernel_atlas.py collect-ar \
  --model ~/.hipfire/models/gemma-4-26b-a4b-it.mq4 \
  --workload gemma4-26b-a4b-it \
  --model-size 26b-a4b \
  --quant mq4 \
  --prefill 6011 --gen 32 \
  --kv-mode asym3 \
  --profile-prefill --profile-decode \
  --isa-dir .hipfire_kernels/gfx1201 \
  --isa-filter 'gemv_hfq4|gemm_hfq4|attention_flash_asym3|moe' \
  --isa-output .codeinsight+research/kernel-atlas/runs/gemma4-isa-gfx1201.json \
  --dispatch-provenance \
  --dispatch-output .codeinsight+research/kernel-atlas/runs/gemma4-dispatch-gfx1201.json \
  --output .codeinsight+research/kernel-atlas/runs/gemma4-ar-gfx1201.jsonl
```

`6d3a44b9`'s "~48 VGPR-per-lane cuts wave occupancy in half" diagnosis
is exactly the claim an ISA Fit View attaches evidence to: HSACO VGPR
budget + theoretical occupancy + observed dispatch shape. Per
`hipfire-kernel-tuning` skill: "If VGPR/SGPR/spills are high, prioritize
register pressure and spill removal before claiming a bandwidth win."

### 4. Cross-arch perf coverage

All bench numbers cite `gfx1201 / R9700` only. The repo carries
speed-baselines for `gfx906/908/942/1013/1030/1100/1100x2/1151/1201`,
but no Gemma 4 row has been collected against any other arch.

The wave32-tuned kernels added on this branch (`attention_flash_asym3_tile_hd512`,
the four indexed/bucketed MoE GEMVs) all use `__launch_bounds__(32, 16)`
and `__shfl_*` reductions with offsets ≤16. They are wave-portable for
correctness (the offsets stay within the active half of a wave64), but
**not** wave-optimal on gfx9xx CDNA. Per `hipfire-kernel-tuning` skill
case-study §3 (wave64 CDNA3 port): the canonical fix is a `gfx94x.hip`
sibling with wave64-shaped reductions.

### 5. Isolated NRMSE for the FWHT + HFQ4G256-batched chain

`521161f8` ships a path where MQ4G256 weights flow through the
HFQ4G256 batched GEMM kernel after the caller pre-rotates `x` via FWHT.
The kernel-isolated NRMSE battery (`verify_against_torch.rs` Phase 2)
validates the *unbatched* GEMV against the dequant reference and
validates the FWHT step separately, but does not exercise
`FWHT(x) → gemm_hfq4g256_batched → y` end-to-end as one unit. The +55%
prefill win is end-to-end-coherence-validated (`"Capital of France?"
→ "Paris is the capital of France."`), which is qualitative.

**To close:** extend `verify_against_torch.rs` Phase 2 with a
"prefill batch row" stage that compares `fwht_batched(x) @ HFQ4G256_W`
to PyTorch's `inverse_fwht(W_dequant) @ x` at N>1 token batches.

### 6. Phase C-iter (`a24bf68f`) non-graph correctness

The Phase C-iter blob conversion was verified by HIPFIRE_GRAPH=1 ABA
test where both sides produce the same `±29.99 saturated softcap`
output — that proves *equal-broken-ness*, not safe-conversion. The
non-graph correctness path was not measured before/after in the commit
message itself. The conversions are likely fine (rest of codebase uses
the same `launch_maybe_blob` path) but the test as run did not
substantiate that.

**To close:** add an explicit non-graph-mode "Capital of France →
Paris" cell before any future hot-path dispatcher conversion, or
better: add `gemma-4-26b-a4b-it.mq4|gemma4-cap-graph|...` row that
exercises HIPFIRE_GRAPH=1 once the kv_len-at-capture fix
(`a3dfd147:gemma4.rs:1568-1611`) is landed.

### 7. Profile timer coverage in `gemma4.rs`

`gemma4.rs` is 2614 net-new lines and contains **zero**
`crate::profile::begin_timer` spans. Every "X is the dominant cost"
claim in the perf-ladder commits is inferred from improvement
direction, not measured. Per `hipfire-kernel-tuning` step 1:
"profiler / `crate::profile` timer flagged a hot kernel as the
bottleneck."

**To close:** mirror `qwen35.rs`'s profile-timer placement around the
weight_gemm / attention_flash / kv_cache_write / apply_moe_branch call
sites in `forward_scratch_inner` and the prefill-batch path.

## What this branch did well (record so the pattern repeats)

- **Fresh-process bench** (median of 3 fresh-prompt curl runs) is the
  correct cross-process A/B protocol per `docs/methodology/perf-benchmarking.md`.
- **`959fb0a5` shipped a negative result honestly** — bucketed Phase B
  v1 was −5.7%, kept opt-in with VGPR-pressure mechanism documented.
  Future-you reading that commit knows the hypothesis + why it
  failed. Exactly the perf-benchmarking-doc negative-result rule.
- **Phase C root cause (`a3dfd147`)** is a textbook bisect:
  structural symptom (`±29.99` softcap saturation) → mechanism
  (scalar-arg capture) → exact scalar (`kv_len = pos + 1`) → exact call
  sites. The load-bearing comment at `gemma4.rs:1568-1611` captures it.
- **The v_norm omission (`7bd2f43a`) was caught and turned into a
  memory** (`feedback_arch_port_v_norm_omission.md`). 770/770 per-kernel
  PASS + end-to-end garbage is the canonical arch-port failure mode and
  now has a named pattern.

## Action items (in priority order)

| # | Action | Status |
|---|--------|--------|
| 1 | Install pre-commit hook + add gemma4 row to gate | **DONE** |
| 2 | Atlas AR collection on gfx1201 | **DONE** (`.codeinsight+research/kernel-atlas/runs/gemma4-ar-gfx1201.jsonl`) |
| 3 | Atlas AR collection on gfx1100 / gfx1151 | DEFERRED (hardware not accessible this session) |
| 4 | Phase 2 batched-prefill verifier row | **DONE** (`verify_batched_prefill.rs` — PASS on MQ4G256 fast path) |
| 5 | `gfx94x.hip` siblings for the wave32-tuned new kernels | DEFERRED (CDNA hardware not accessible, kernel work risky without validation) |
| 6 | Astrea imatrix-scale calibration on gemma-4-26b-a4b | DEFERRED (needs imatrix file; inspect/fingerprint/kv-profile/bundle-plan artifacts shipped) |
| 7 | Profile-timer spans in `gemma4.rs forward_scratch_inner` | DEFERRED — kernel-level timers in `dispatch.rs` already cover the hot kernels; per-section spans are nice-to-have, not blocking |
| 8 | `bench_gemma4_mq4` + speed-gate baseline | **DONE** (bench example + `gemma4_26b_a4b_*` rows in `tests/speed-baselines/gfx1201.txt`) |
| 9 | Root-cause + fix v2 batched-prefill regression | NARROWED, NOT ROOT-CAUSED. `verify_batched_prefill` rules out MQ4G256 fast path; bug repros at n_batch=1; device_sync at function exit doesn't fix; code review didn't reveal it. Needs intermediate-state pb_q/k/v dumps vs v1 to localize. Workaround: v1 routing (109 prefill tok/s, 71 decode tok/s — preserves d2bb8573's +38% over per-token). |
| 10 | hipGraph `kv_len` kernel-signature work (~7 attention kernels) | **NOT THE BUG**. Attention kernels read `seq_len = pos_buf[0] + 1` at runtime; the scalar diagnosis was wrong. Actual fix landed in commit `7fc06c14`: `mul_f32` / `add_f32` ran on `None` stream (NOT recorded into the captured graph at all → FFN `gelu(gate)*up` step silently dropped on every replay → token attractor). `scale_f32` ran on `stream_ref()` but used raw `kernelParams` (stack pointers dangle under ROCm 7.x). All three converted to `launch_maybe_blob`. HIPFIRE_GRAPH=1 now produces clean output on both gemma4-cap and gemma4-longctx (1271 prefill past 1024 sliding cap). **However**, hipGraph is currently a perf REGRESSION on this hardware (gfx1201 / ROCm 7.2): Qwen 9B mq4 decode 93 → 68 tok/s graph-on, Gemma 4 26B-A4B decode 71 → 39 tok/s graph-on. Per-blob graph-executor replay cost dominates. HIPFIRE_GRAPH stays default-off until the ROCm 7.x graph executor regression is investigated. |

## Cross-arch artifacts collected this session (gfx1201 only)

The host is k9lin = gfx1201 (R9700). Cross-arch artifacts require hardware
the session doesn't have access to. The diagnostic infrastructure committed
on this branch is reusable by any contributor with gfx906/908/942/1010/1030/
1100/1101/1102/1151 hardware:

```bash
# Per arch:
python3 scripts/kernel_atlas.py collect-ar \
  --bench ./target/release/examples/bench_gemma4_mq4 \
  --model ~/.hipfire/models/gemma-4-26b-a4b-it.mq4 \
  --workload gemma4-26b-a4b-it --quant mq4 \
  --prefill 32 --gen 50 --warmup 5 --kv-mode asym3 \
  --profile-prefill --profile-decode \
  --isa-dir .hipfire_kernels/<arch> \
  --output .codeinsight+research/kernel-atlas/runs/gemma4-ar-<arch>.jsonl

# Then capture baseline rows in tests/speed-baselines/<arch>.txt
./target/release/examples/bench_gemma4_mq4 ~/.hipfire/models/gemma-4-26b-a4b-it.mq4 \
  --prefill 32 --warmup 5 --gen 50
./target/release/examples/bench_gemma4_mq4 ~/.hipfire/models/gemma-4-26b-a4b-it.mq4 \
  --prefill 128 --warmup 5 --gen 50
```
