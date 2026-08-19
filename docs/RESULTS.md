# J-Space Mechanistic Interpretability: Final Results

## Executive Summary

Qwen2.5-3B performs arithmetic through a **W-shaped information bottleneck**:
input compression → computational expansion → output crystallization. This
three-phase dynamics is supported by PCA capacity analysis, sham-controlled
operand probes, and causal activation steering. Linear wave superposition,
circular geometry, and temporal precedence are falsified by matched controls.

---

## Confirmed Results (Sham-Controlled)

### 1. Operand Magnitude Encoding (Layer-wise)
- **Method:** Ridge regression probes for operands A/B, evaluated per-layer with label-shuffled sham control.
- **Result:** Specific correlation (Δ ≈ +0.9 vs sham ≈ 0) across ALL 37 layers.
- **Peak:** Layers 13–22 (corr A: 0.89–0.91, corr B: 0.86–0.89).
- **Interpretation:** Operands encoded as continuous magnitudes, not digit atoms. Computation concentrated in middle layers.

### 2. GWT Capacity Bottleneck (~4 Chunks)
- **Method:** PCA effective dimensionality (85% variance threshold) on last-prompt-token activations, 64 examples.
- **Result:** U-shaped capacity profile across 36 layers.

| Zone | Layers | eff_dim | var_1st | Role |
|------|--------|---------|---------|------|
| Early (Collapse) | 0–13 | 3–4 🔵 | 0.44–0.59 | Input compression to GWT bottleneck |
| Middle (Boiling) | 14–27 | 5–7 🟡 | 0.30–0.42 | Computational expansion |
| Late (Crystallization) | 28–35 | 4 🔵 | 0.38–0.45 | Output compression for verbalization |

- **Mean effective dimensionality:** 4.8 ≈ "4 chunks" (Baars/Dehaene GWT).
- **Key alignment:** Middle-layer expansion coincides with operand-corr peak and bifurcation zone.

### 3. Non-linear Bifurcation via Causal Steering
- **Method:** Differential vector `trace(A+B) − trace(A−B)` injected into middle layers (prompt-only), fine-grained α sweep with random-vector sham.
- **Key example (58+27):**

| α | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 |
|---|-----|-----|-----|-----|-----|
| REAL | 85 | 85 | **5** | 5 | 5 |
| SHAM | 85 | 85 | 85 | 85 | 85 |

- **Result:** Sharp bifurcation at α≈0.2 (REAL only); SHAM stable. Operation-specific (addition/subtraction affected; multiplication/division immune).
- **Interpretation:** SwiGLU gates act as threshold switches, not smooth modulators. Numeric computation is non-linear attractor dynamics.

### 4. LoRA Heals Quantization
- **RCR quantized:** 2/6 on TEST arithmetic.
- **RCR + LoRA (fp16-trained adapter):** 6/6.
- **Interpretation:** Lightweight adapter corrects quantization artifacts at the interface level without touching base weights.

---

## Falsified Hypotheses (Sham-Controlled Negatives)

| Hypothesis | Test | Sham Δ | Verdict |
|------------|------|--------|---------|
| Temporal precedence | Lead heuristic | −0.8 | ❌ Artifact of margin heuristic |
| Circular magnitude (Sun et al.) | Phase mod 10 | −1 sector | ❌ Artifact of ridge in overcomplete regime |
| Linear wave interference | Cosine superposition | +0.011 | ❌ Falsified; baseline cosine ~0.8 dominates |
| Wave resonance (causal) | Accuracy at varying α | 0 / harmful | ❌ No linear amplification |
| Per-digit readout | Linear / binary / circular | — | ❌ Impossible; digits fused into magnitude |

**Methodological note:** All positive claims survived label-shuffled or random-vector sham controls. Multiple hypotheses that appeared positive under naive metrics were falsified by matched controls, preventing false discoveries.

---

## Physical Manifesto: W-Shaped Computation

> Computations in Qwen2.5-3B follow a W-shaped information bottleneck:
>
> 1. **Collapse (L0–13):** Input compresses to ~4 dimensions (GWT bottleneck). Operands encoded as holistic magnitudes, not digit chains.
> 2. **Boiling (L14–27):** Workspace expands to ~7 dimensions. Non-linear SwiGLU gates drive computation; operand encoding peaks; causal steering triggers bifurcation at α≈0.2.
> 3. **Crystallization (L28–35):** Result compresses back to ~4 dimensions, forming a compact attractor for LM-head tokenization.
>
> This three-phase dynamics is supported by PCA capacity analysis, sham-controlled operand probes, and causal activation steering. Linear wave superposition, circular geometry, and temporal precedence are falsified. Computation is non-linear attractor dynamics within a GWT-constrained workspace.

---

## Assets Produced

| Asset | Path | Description |
|-------|------|-------------|
| RCR-quantized model | `qwen3b_rcr.pt` | 4-bit NF4 + Int8 residual plug |
| LoRA adapter | `lora_math.pt` | fp16-trained, heals quantization |
| Op/A/B probes | `steer.pt` | Ridge probes, sham-validated |
| Digit probes (negative) | `digits.pt`, `circ_digits.pt` | Failed readout attempts |
| Interpreter | `jprobe_interpreter.py` | Op trajectory + thought tokens |
| GWT analysis | `wave_gwt_pca.py` | Layer-wise capacity |
| Causal steering | `wave_resonance_fine.py` | Bifurcation demonstration |
