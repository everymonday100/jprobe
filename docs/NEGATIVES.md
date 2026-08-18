# Documented Negatives

## Context-free SSAE bottleneck

**Experiment:** Step-level bottleneck autoencoder (Yang et al. SSAE) without context conditioning.

**Result:** Activation sparsity = 0.001, fidelity degraded.

**Why:** SSAE requires end-to-end training with context-aware decoder; naive bottleneck collapses in decoder-only models.

**Lesson:** Step-level bottleneck is not drop-in; needs architectural changes.

## Quantization healer (healer_train.py)

**Experiment:** Denoising autoencoder trained to "heal" quantization noise from residual stream states.

**Result:** Domain mismatch — healed states caused generation collapse or degradation.

**Why:** 
- Healer trained on single-shot noise, but inference compounds error recursively
- Real Q4 quantization uses calibrated ranges; naive per-row rounding destroys model
- Domain gap: clean states (training) vs damaged states (inference)

**Lesson:** Quantization artifacts require QAT/calibration, not hook-based denoising.

## Cross-model chain transfer with gating

**Experiment:** Weak-oracle detector flags errors in receiver model, triggers chain injection.

**Result:** Gated transfer = 3/6 < base 5/6 (harmful)

**Why:**
- Detector trained on 64 samples, miscalibrated (false flags on correct answers)
- Chain injection causes copying artifacts (520530, 353)
- Sun et al. achieved success with >90% accurate detector + same-model recompute

**Lesson:** Cross-model transfer requires receiver-adapted gating with high-accuracy detectors; blind or naive gating is counterproductive.

## Base probes fail on LoRA model

**Experiment:** Apply base-model probes to LoRA-aligned model.

**Result:** Latent readout drops to chance (op 0.27–0.38)

**Why:** LoRA shifts residual stream geometry; probes are alignment-sensitive.

**Lesson:** After any alignment (SFT, RLHF), re-probe from scratch; probes are not invariant.

## Limitations summary

- Small eval (6-prompt TEST) — no CIs
- Single model family (Qwen2.5-3B)
- No matched-budget controls (filler tokens, random perturbations)
- Probes alignment-sensitive

Future work: held-out 200–500 tasks, seed CIs, Wang's 6-arm factorization.
