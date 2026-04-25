# Empirical Results & Discussion

This document honestly reports what this implementation of LDStega achieved, where it fell short of the paper's claims, and what experiments were run. The goal is to be useful to anyone trying to reproduce or build on this work — not to oversell.

---

## Paper claims vs. empirical findings

The paper claims near-zero BER for short messages at moderate gamma values, and practical capacity in the hundreds to low thousands of bits with robustness to real-world transforms. In my experiments:

| Claim | Observed |
|-------|----------|
| Near-zero BER at gamma=0.3, short messages | BER ~0.15–0.25 before ECC (no-attack baseline) |
| Robust to messenger-style transforms | BER typically 0.25–0.45 after Telegram/WhatsApp pipelines |
| Capacity scales to ~4096 bits | BER degrades sharply above ~512 bits; 4096 bits not practically usable |

The no-attack BER should be near zero if the algorithm is working correctly — the receiver re-runs the same diffusion process and reads the sign of modified latents. The fact that it is not zero points to implementation-level discrepancies (see hypotheses below).

---

## Seed sensitivity

This is the most important finding: results vary substantially across seeds at fixed configuration. At gamma=0.3, 256-bit messages, no-attack:

- Seed 42:  BER ≈ 0.17
- Seed 123: BER ≈ 0.23
- Seed 777: BER ≈ 0.19

A ±0.06 range on a baseline that should be near zero suggests the algorithm is sensitive to something in the latent structure that varies per seed. This variance is not discussed in the paper.

---

## Gamma analysis

Larger gamma → wider truncated Gaussian intervals → more robust embedding, but more visible distortion.

Empirically:
- gamma=0.1: PSNR ~38 dB, BER ~0.35 (embedding too subtle, easily disrupted)
- gamma=0.3: PSNR ~34 dB, BER ~0.20 (best practical tradeoff observed)
- gamma=0.5: PSNR ~30 dB, BER ~0.15
- gamma=1.0: PSNR ~24 dB, BER ~0.12

The gamma vs. BER/PSNR tradeoff is analyzed in detail in [notebooks/02_bit_accuracy_analysis.ipynb](notebooks/02_bit_accuracy_analysis.ipynb).

---

## ECC as mitigation

Error-correcting codes reduce BER after decoding. Hamming(7,4) and Reed-Solomon codes were both explored across the full benchmark matrix. They help, but cannot fully compensate for the fundamental BER floor:

- At gamma=0.3, 256 bits, no-attack: BER drops from ~0.20 to ~0.14 with Hamming(7,4)
- Under Telegram pipeline: BER drops from ~0.38 to ~0.28 — meaningful but still high for practical use

ECC is most useful as a margin booster when the underlying BER is already low (< 0.10). When BER is above ~0.25, ECC provides diminishing returns.

---

## Experiments conducted

All experiments used `runwayml/stable-diffusion-v1-5` unless noted.

**Parameter sweeps**
- Gamma (0.1, 0.2, 0.3, 0.5, 0.8, 1.0) × message length (256, 512, 1024, 2048, 4096 bits) under JPEG q=75
- Transfer format (none, pil, png, jpeg:95, jpeg:75, jpeg:50) × gamma (0.2, 0.3, 0.5, 0.8) × message length (256–4096 bits)

**Robustness**
- 60+ isolated transforms: JPEG at various qualities, WebP, resize (0.25×–0.9×), Gaussian blur, additive noise, color jitter, center crop, rotation
- Messenger platform pipelines: Telegram (resize 0.75 + JPEG 85), WhatsApp (resize 0.5 + JPEG 60 + brightness), Instagram (resize 0.7 + JPEG 75 + color jitter), WeChat (resize 0.75 + JPEG 70 + blur), double-JPEG, screenshot chain
- Stress tests: JPEG q=10, 4× downscale, heavy blur (σ=3), heavy noise (std=0.1), all combined

**ECC**
- 8 ECC codes (NoECC, Repetition 3/5, Hamming 7/4, 15/11, RS 255/223, 255/191, 255/127) × 8 representative attacks
- Capacity trade-off: 6 codes × 5 message lengths

**Internals / diagnostics**
- VAE roundtrip accuracy: how much D = |Z_T − Z'_T| changes across runs with the same seed
- D vs D' between sender and receiver: whether the receiver's discrepancy map matches the sender's exactly (it does not always, which directly affects position selection)
- Transfer format matching: effect of mismatched sender/receiver format on BER
- DTAMS baseline comparison (separate algorithm, different approach — included for context)

---

## Hypotheses on the gap

None of these are confirmed. They are directions worth investigating.

**1. Transfer format assumption**
The algorithm assumes sender and receiver agree on the transfer format used for the D calculation. In practice, if the VAE encoder introduces any non-determinism (e.g., `latent_dist.sample()` vs `.mode()`), the receiver's D will differ from the sender's, causing misclassified positions and elevated BER.

**2. `eta=1.0` default**
With `eta=1.0` (full DDPM stochasticity), each scheduler step introduces fresh randomness from the generator. Small differences in PyTorch RNG state between encode and decode — e.g., from non-deterministic CUDA ops — can cause Z_T to differ slightly, which shifts mu_{T-1} and sigma_{T-1} and degrades the decision boundary.

**3. MS interval boundaries**
The paper specifies interval boundaries conceptually but does not publish the exact values. The boundaries used here (0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, ∞) were inferred. Different boundaries would change which positions are selected.

**4. Seed-dependent latent structure**
Some seeds produce latent spaces where low-discrepancy positions cluster in semantically important regions of the image. Embedding bits there at gamma=0.3 may cause larger visible perturbations than expected, and also may interact with the scheduler's noise pattern in ways that increase decode error.

---

## Reproduce these results

```bash
# Baseline no-attack BER across seeds
python benchmark_ldstega.py --suite individual --seeds 42,123,777 --output-dir results/individual

# Messenger pipeline robustness
python benchmark_ldstega.py --suite messenger --seeds 42,123,777 --output-dir results/messenger

# Gamma × message length sweep
python benchmark_ldstega.py --suite param_sweep --seeds 42 --output-dir results/param_sweep

# ECC comparison
python benchmark_ldstega.py --suite ecc_comparison --seeds 42 --output-dir results/ecc
```

Results are saved as CSV and JSON. The analysis notebooks in `notebooks/` expect results in `results/`.

---

## Discussion

If you have reproduced the paper's results, or have hypotheses about the gap, please open an issue. The most likely place to look is the D calculation and position selection — if sender and receiver don't agree exactly on which positions carry bits, BER will be non-zero even with a perfect image. A simple diagnostic: log the selected positions at encode time and check whether decode selects the same set.
