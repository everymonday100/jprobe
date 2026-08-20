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

## Temporal precedence (Wang mediator)

**Experiment:** Measure lead of latent op-commitment over surface verbalization using margin-based plateau detection; sham control with label-shuffled op-probe.

**Result:** REAL lead = +15.8 steps, SHAM lead = +16.7 steps, Δ = −0.8.

**Why:** Margin-based plateau heuristic captures general confidence dynamics, not operation-specific commitment. Any ridge probe produces similar "plateau" on shuffled labels due to high-dimensional geometry.

**Lesson:** Temporal precedence requires stricter operationalization (e.g., fixed-token alignment, causal intervention); margin heuristics are insufficient in overcomplete latent spaces.

## Circular magnitude probe (Sun et al. geometry)

**Experiment:** Train circular (cos/sin) probe on operand magnitude mod 10; sham control with value-shuffled labels.

**Result:** REAL = 7/8 phase sectors, SHAM = 8/8 sectors, Δ = −1. Both show smooth phase distribution.

**Why:** Ridge regression in high-dimensional overcomplete regime (2048-dim hidden, ~5000 samples) produces smooth circular distributions even on random labels. Phase coverage is a property of the estimator, not the data.

**Lesson:** Circular geometry claims require sham controls; phase-sector counting alone is not evidence of geometric encoding. Operand magnitudes are reliably encoded (corr Δ ≈ +0.9), but NOT as phases on a circle.

## Linear wave interference

**Experiment:** Test whether `(trace(A+B) + trace(A−B))/2 ≈ trace(A)` using cosine similarity with sham (random C) and baseline controls.

**Result:** Sim_target = 0.835, Sim_sham = 0.824, Δ = +0.011. All similarities ~0.8 due to shared prompt context.

**Why:** Transformer latent trajectories have high baseline cosine similarity (~0.8) from shared syntactic structure. Arithmetic computation is non-linear (attention-mediated), not additive superposition.

**Lesson:** Linear wave models of latent computation are falsified for this model/task. Numeric semantics is distributed non-linearly; superposition does not hold.

## Per-digit readout

**Experiment:** Three methods attempted: (1) linear A/B regression, (2) binary digit-presence probes (16/32 prompts), (3) circular phase readout mod 10.

**Result:** All failed. Linear gives uncalibrated values; binary probes produce noise; circular phases collapse to binary ±π/0 (constant-label artifact) or fail sham.

**Why:** Digits are fused into holistic magnitude representations in residual stream. No stable per-digit atom exists at the hidden-state level; digit tokens emerge only at unembedding.

**Lesson:** Per-digit interpretability requires probing at the token/embedding boundary, not mid-layer residual stream. Magnitude-level readout (operand corr) is the correct granularity for this model.

## Wave resonance via causal steering (linear amplification)

**Experiment:** Inject differential vector `trace(A+B) − trace(A−B)` into middle layers at varying α; measure accuracy change with random-vector sham.

**Result:** No monotonic accuracy improvement. At α ≥ 0.2, sharp bifurcation (85→5 for 58+27); SHAM stable. Large α causes hallucination (990, 4, 15).

**Why:** SwiGLU gates act as threshold switches, not linear amplifiers. Differential vector triggers non-linear attractor switching, not smooth modulation. Effect is operation-specific (addition/subtraction only).

**Lesson:** Latent steering produces non-linear bifurcations, not linear resonance. The correct model is attractor dynamics with threshold transitions, not wave amplification. This IS a confirmed causal effect (sham-stable), just not the linear kind initially hypothesized.

## RMT filtering (Marchenko-Pastur / Tracy-Widom signal extraction)

**Experiment:** Filter middle-layer activations (L14–27, last prompt token, N=64)
by zeroing singular components below an adaptive spectral threshold
(MP-informed, median-eigenvalue × 3), then re-measure out-of-sample operand corr
(k-fold CV, no label leakage).

**Result:** Mean OOS corr drops from 0.94 → 0.89 (Δ = −0.047); worst at range
edges (Δ = −0.11 at L14, L24–27). ~17 components retained; still harms generalization.

**Why:** Operand signal is NOT concentrated in a few Tracy-Widom spikes. It is
distributed across the spectrum, so any hard spectral cutoff (MP, PCA, circular)
removes task-relevant directions. Linear/spectral compression is lossy by design here.

**Lesson:** Operand encoding is a distributed subspace, not a low-rank spectral
outlier. Consistent with all prior negatives (temporal, circular, wave): the
signal is non-linear and spread out; only magnitude-level distributed probes work.

## Spectral subtraction ("remove top-k syntax slab")

**Hypothesis (Gemini):** top-17 spectral components are a "syntax slab" masking
interference waves of digits; zeroing them should reveal hidden digit structure.

**Experiment:** Zero top-k singular components (k=4,8,17,32,64), reconstruct,
measure out-of-sample operand corr (k-fold CV).

**Result:** Removing even top-4 collapses OOS corr from 0.94 to −0.65.
k=64 → 0.000 (all-zero, trivial). Any top-k removal destroys the signal.

**Interpretation:** Top spectral components are NOT syntax noise — they carry
the operand signal. There is nothing readable "under the slab". Combined with
the keep-top-17 test (0.94→0.89), this shows operand encoding is concentrated
in a few dominant components, consistent with the GWT ~4-chunk bottleneck.

**Verdict:** Spectral-subtraction hypothesis falsified. Top components = working
memory core, not a mask.

## Linear per-digit readout (three methods)

**Experiment:** Three linear/circular methods attempted:
(1) linear A/B regression, (2) binary digit-presence probes (16/32 prompts),
(3) circular phase readout mod 10.

**Result:** All failed. Linear gives uncalibrated values; binary probes produce
noise; circular phases collapse to binary ±π/0 (constant-label artifact) or fail sham.

**Why:** Digits are encoded non-linearly and crystallize gradually across layers.
Linear probes average across the trajectory and miss layer-specific structure.
MLP next-token prediction (see RESULTS.md) shows digits ARE readable non-linearly
with progressive crystallization (Δ=+0.28 early → +0.82 late).

**Lesson:** Per-digit interpretability requires non-linear probes AND layer-specific
analysis. Linear/circular methods are insufficient for this model's encoding scheme.

## Limitations summary

- Small eval (6-prompt TEST) — no CIs
- Single model family (Qwen2.5-3B)
- No matched-budget controls (filler tokens, random perturbations)
- Probes alignment-sensitive
- Sham controls revealed multiple false positives (temporal precedence, circular geometry); all positive claims now require matched controls

Future work: held-out 200–500 tasks, seed CIs, Wang's 6-arm factorization.
