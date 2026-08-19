# jprobe

**Read, steer and align latent reasoning traces (J-space) in LLMs** — probes, latent
recursion, think-loops, chain reuse, and a LoRA alignment leg that closes the
presence≠usage gap.

A mechanistic interpretability toolkit for decoder LLMs (validated on
Qwen2.5-3B-Instruct). Implements a latent-recursion bridge over a frozen base,
circular/linear/MLP readout probes, adaptive steering, latent think-loops with reusable
chain indices, and lightweight alignment. Grounded in the H1 / global-workspace
literature: Wang (2026), Sun et al. (2025), Yang et al. SSAE (2026), Anthropic J-space (2026).

## Install

```bash
pip install -r requirements.txt
```

Point `SF_DIR` in `jspace.py` at a HuggingFace model dir (safetensors).

## Pipeline

```bash
# 1. corpus + bridge
python teacher.py collect "Сколько будет 17*23? Рассуждай пошагово."   # x64
python teacher.py train --bridge deep --epochs 500

# 2. probes
set PROJ=proj_deep.pt
python probe5.py            # latent readout (TEST none)
python probe8.py            # steering-aware linear/MLP
python probe_circular.py    # circular digit geometry
python probe10.py           # voting

# 3. think-loop + transfer
python think_loop.py --steps 48 --second

# 4. alignment (close presence≠usage)
python generate_finetune_data.py
python finetune_math.py
python check_lora.py
python control_lora2.py
```

## Headline results (Qwen2.5-3B-Instruct)

| Block | Result | Status |
|---|---|---|
| Latent readout, TEST none | op 0.85–1.00, A/B 0.6–0.9 | ✅ H1 sufficiency |
| Circular digit geometry | circular beats linear OOD | ✅ reproduces Sun et al. |
| presence≠usage | probes read, surface garbled pre-SFT | ✅ confirmed |
| Data-limited scaling | 16→32→64 monotone gain | ✅ |
| Self-scratchpad (think-loop) | +1 exact (4/6 vs 3/6) | ✅ |
| LoRA alignment | surface 0.225→0.775 / 1.0 | ✅ closes gap |
| Alignment control | latents 0.78–0.98 @ surface 1.0 | ✅ agreement |
| Cross-model chain transfer | blind partial, gated harmful | ⚠️ boundary regime |
| Probe invariance | base probes fail on LoRA | ⚠️ alignment-sensitive |
| Context-free SSAE bottleneck | act=0.001 | ❌ negative |
| Quantization healing (RCR+LoRA) | 2/6 → 6/6 | ✅ perfect recovery |
| Operand encoding (layer-wise, sham-controlled) | Δ ≈ +0.9 all layers | ✅ Confirmed |
| GWT bottleneck (~4 chunks) | eff_dim 3–4 in 22/36 layers | ✅ Confirmed |
| Non-linear bifurcation (causal steering) | α=0.2 threshold, sham-stable | ✅ Confirmed |
| LoRA heals RCR quantization | 2/6 → 6/6 | ✅ Confirmed |
| Temporal precedence | sham Δ = −0.8 | ❌ Artifact |
| Circular geometry (Sun et al.) | sham Δ = −1 sector | ❌ Artifact |
| Linear wave interference | sham Δ = +0.011 | ❌ Falsified |

## File map

See directory listing; each module's docstring states its role. `docs/RESULTS.md`
holds the full per-experiment tables; `docs/NEGATIVES.md` documents failures and
why they are expected under the H1/SSAE/Sun framework.

## Limitations

Small eval sets (6-prompt TEST), single model family, no seed CIs, probes are
alignment-sensitive (re-probe after SFT). See `docs/RESULTS.md`.

## License

Apache 2.0.
