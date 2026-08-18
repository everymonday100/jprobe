# Results

## Latent readout (probe5 TEST none, 32 prompts)

| bin | op | A | B | abstract |
|---|---|---|---|---|
| 0–8 | 1.00 | 0.87 | 0.84 | 0.58 |
| 8–16 | 0.94 | 0.67 | 0.64 | 0.52 |
| 16–32 | 0.92 | 0.72 | 0.79 | 0.60 |
| 32–999 | 0.85 | 0.60 | 0.74 | 0.63 |

Latent stream carries operation and operands OOD robustly.

## Circular digit geometry (probe_circular, TEST)

Circular probes outperform linear on OOD:
- Circular: 2/6 exact (64, 37)
- Linear: 0/6 exact

Reproduces Sun et al. (2025) in bridge-generated latents.

## presence≠usage

**Pre-SFT (base model):**
- Latent readout: op 0.85–1.00
- Surface accuracy: 0.225 (eval), 0.5 (TEST)

**Post-SFT (LoRA):**
- Surface accuracy: 0.775 (eval), 1.0 (TEST)
- Latent readout (re-probed): op 0.78–0.98, A/B 0.28–0.69

Gap closed by lightweight alignment.

## Self-scratchpad (think_loop.py)

Base: 3/6 exact → Think: 4/6 exact (+1)

Injecting own latent chain improves surface output.

## Cross-model transfer (3B→0.5B)

Blind transfer: 3/6 exact (partial)
Gated transfer: 3/6 (harmful — detector miscalibrated)

Consistent with SSAE Table 6 (Yang et al.) boundary.

## Alignment control (control_lora2.py)

Surface acc = 1.0
Latent readout (LoRA-trained probes): op 0.78–0.98, A/B 0.28–0.69

Latents and surface agree post-alignment.
