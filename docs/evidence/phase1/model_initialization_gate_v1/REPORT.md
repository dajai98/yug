# YUG-100M model initialization gate

Date: 2026-10-08 (Asia/Riyadh).

**Initialization gate: PASS. FREEZE YUG-100M ARCHITECTURE.**

## Minimal change

Only YugTransformer.__init__ in yug/native/model.py changed. Immediately after tying the output head to the embedding, scale that shared matrix once in-place under no_grad by hidden_size**-0.5. Policy: tied_embedding_fan_in_v1. Existing N(0,1) samples become N(0,1/h); standard deviation at h=640 is 0.03952847. No extra RNG draws, no other initial tensor changes, no extra parameters, no forward-time logit scaling. Untied models retain their previous initialization.

This targets order-one logit variance for normalized hidden states under approximate coordinate independence. The old shared matrix also aligned with the current-token residual, causing saturated copying. Architecture, forward computation, state-dict shapes/keys and loaded-checkpoint outputs remain unchanged. Fresh tied-model weights and outputs intentionally change; initialization itself is not legacy-identical. All Linear initializers and RMSNorm scales are unchanged. No depth-dependent residual scaling is introduced: measured residual RMS did not justify that additional change.

## Frozen protocol

acceptance.json predates the pilot and production edit. The prior architecture gate's exact inputs were reused: 64 next-token positions each from code, English and Nepali, seeds 17/23/31, CPU FP32, four threads. Original source is retained and hashed as legacy_model.py. No new corpus/tokenizer data was acquired.

Predeclared per-family/seed screens: finite outputs/RMS; NLL within 2 nats of ln(4096); logit std below 2; mean maximum probability below 5%; entropy above ln(4096)-2; current-token argmax below 10%; every block RMS below 10. These are engineering screens, not guarantees of convergence or task quality. None were relaxed.

The pilot scaled only the original model's shared tensor in memory. Production verification rebuilt old and corrected models from matching seeds and obtained exactly equal outputs between the isolated correction and production implementation for all three seeds. No optimizer updates or YUG-100M training occurred.

## Before / after

Ranges cover nine seed/family combinations, except block RMS which covers three seeds.

| Metric | Original | Corrected |
|---|---:|---:|
| Step-0 next-token NLL, nats | 431.087-449.917 | 8.610-8.909 |
| Logit standard deviation | 26.122-26.292 | 0.984-1.014 |
| Maximum absolute logit | 502.422-525.685 | 3.854-4.563 |
| Mean maximum probability | 100% | 0.357-0.681% |
| Predictive entropy, nats | approximately 0 | 7.793-7.849 |
| Current-token argmax fraction | 100% | 0-1.5625% |
| Mean current-token logit | 441.716-452.638 | 0.501-0.830 |
| First-block RMS | 1.019-1.025 | 0.206-0.211 |
| Final-block RMS | 1.442-1.451 | 1.554-1.571 |

Uniform reference NLL is ln(4096)=8.317766. Corrected predictions need not be exactly uniform. All corrected screens pass. Final residual RMS is slightly higher, not lower: the improvement is output saturation. Per-layer trajectories and per-seed results are in pilot.json and verify.json.

## Correctness and numerical verification

Tests prove exact equality of every other initial tensor and post-construction RNG state, preserved weight aliasing/counts, same-seed reproducibility, and bit-identical outputs after loading serialized original weights. Variants cover tied/untied, learned positions/bias, and LayerNorm/SwiGLU. AST comparison finds only YugTransformer.__init__ changed; all forward methods are unchanged.

One full-size 32-token backward diagnostic produced finite gradients for **162/162 parameter tensors**, gradient L2 **28.8522**, loss **8.726904**. Zero optimizer steps. This is a differentiability/numerics check, not training.

A batch-1 **2048-token** forward boundary check passed: finite logits, std **0.992719**, mean maximum probability **0.60694%**, entropy **7.81071**. Repeated existing token IDs were a shape stress input, not generated training data. This does not prove full-context backward/memory feasibility or throughput. Results: gradient_context_checks.json. Replay helper: extra_checks.py.

Focused: `python -B -m pytest -q tests/test_native_initialization.py tests/test_native_model.py tests/test_native_config_core.py tests/test_native_config_serialization.py` ? **193 passed, 0 failed, 0 skipped**, 8.26 seconds.

Full: `python -B -m pytest -q` ? **1057 passed, 0 failed, 1 skipped**, 102.30 seconds. Seven added test cases explain the increase from 1050. The skip remains tests/test_brain_llama_cpp_adapter.py:136 requiring YUG_TEST_MODEL_PATH. No external model was used. These runs completed before the documentation interruption and were not unnecessarily repeated. The production source hash still matches the verified artifact.

## Exact architecture freeze

- Hidden size 640; layers 20; heads 10; head dimension 64.
- FFN size 2560; ungated two-projection SiLU.
- Context 2048; vocabulary 4096; tied embeddings.
- Pre-RMSNorm plus final RMSNorm, scale-only, epsilon 1e-6.
- RoPE base 10000, adjacent-coordinate pairing, full head dimension, no scaling.
- Bias false; dropout 0.0 (attention-probability placement).
- Initialization tied_embedding_fan_in_v1; other initializers unchanged.
- Verified dtype baseline FP32; mixed precision not approved by this gate.

Exact parameters: shared embedding 2,621,440; attention 32,768,000; FFN 65,536,000; norms 26,240; additional output/positional parameters zero. **Total 100,951,680.**

model_config.json is the frozen ModelConfig payload. freeze.json additionally binds initialization policy, implementation constants/hash and count. Retain both: ModelConfig identity excludes initializer and dtype, because checkpoint-compatible architecture/forward semantics did not change. The freeze-record identity binds the additional initialization specification. Legacy same-seed fresh initialization requires the archived original source; existing trained checkpoints remain compatible.

Frozen tokenizer identity:
`sha256:0b9b2bb9efe58902e3cf3dcb359cf174a25f3d7bf603b14bd3fe8e2d7fec0507`.

Historical reports remain unchanged. This gate supersedes the architecture initialization blocker without changing prior measurements.

## Remaining limits and next action

Sane step-0 behavior does not establish optimization stability, trained capability, learning-rate suitability, mixed-precision safety or 2B-token throughput. CPU FP32 is verified; no cross-device bitwise guarantee. The full-context check was forward-only; backward used 32 tokens. No optimizer/schedule was selected. Existing regression tests include their usual tiny training checks; the new diagnostics performed no optimizer updates.

Single next action: define a bounded training-readiness gate for hardware/precision and optimizer/schedule checks around this frozen architecture. **Corpus build and YUG-100M training remain unauthorized.**

Reproduction: `python -B artifacts/model_initialization_gate_v1/probe.py verify` replaces verify.json; preserve/copy the evidence first. Additional checks: `python -B artifacts/model_initialization_gate_v1/extra_checks.py <fresh-output-path>`.
