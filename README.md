# LDStega: Unofficial PyTorch Implementation

![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

Unofficial implementation of **"LDStega: Practical and Robust Generative Image Steganography based on Latent Diffusion Models"** — Jiang et al., [ACM MM 2024](https://dl.acm.org/doi/10.1145/3664647.3681063).

> This is not an official release. See [RESULTS.md](RESULTS.md) for a frank discussion of what was reproduced and what wasn't.

---

## How it works

Secret bits are embedded into diffusion-generated images by manipulating latent values at the final DDIM step. For each latent position, a discrepancy score D = |Z_T − Z'_T| is computed by round-tripping the decoded image back through the VAE — positions with low discrepancy are the most stable across image transforms and are used first. Bits are encoded by sampling from a truncated Gaussian: bit 0 pulls the latent below the predicted mean, bit 1 pushes it above. The receiver decodes by re-running the same diffusion process with the same seed and prompt, then reading the sign of each modified latent relative to the mean.

---

## Results

These are empirical results from this implementation. **They do not match the theoretical bounds claimed in the paper.**

| gamma | msg length | BER (no ECC) | BER (Hamming 7,4) | PSNR (dB) | model |
|-------|-----------|-------------|-------------------|-----------|-------|
| 0.1   | 256 bits  | ~0.35       | ~0.28             | ~38       | SD 1.5 |
| 0.3   | 256 bits  | ~0.20       | ~0.14             | ~34       | SD 1.5 |
| 0.5   | 256 bits  | ~0.15       | ~0.10             | ~30       | SD 1.5 |
| 0.3   | 512 bits  | ~0.28       | ~0.20             | ~34       | SD 1.5 |
| 0.3   | 1024 bits | ~0.38       | ~0.30             | ~34       | SD 1.5 |

Results varied substantially across seeds (BER range of ±0.08–0.12 at fixed configuration). Numbers above are approximate means across seeds 42, 123, 777.

The theoretical capacity and near-zero BER claimed in the paper were not reproduced. See [RESULTS.md](RESULTS.md) for a full discussion and hypotheses on why.

---

## Supported models

| Model | HuggingFace ID | Status |
|-------|---------------|--------|
| Stable Diffusion 1.5 | `runwayml/stable-diffusion-v1-5` | Tested |
| Stable Diffusion 2.1 | `stabilityai/stable-diffusion-2-1` | Tested |
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` | Tested |
| LDM | `CompVis/ldm-text2im-large-256` | Tested |

---

## Installation

```bash
git clone https://github.com/a3agalyan/ldstega-unofficial
cd ldstega-unofficial
pip install -e .
```

---

## Quick start

```python
from ldstega import LDStega, StegoConfig

config = StegoConfig(model_id="runwayml/stable-diffusion-v1-5", truncation_gamma=0.3)
stega = LDStega(config)

secret = LDStega.text_to_bits("I love steganography!!!")
stego_image = stega.encode("a sunset over mountains", secret_bits=secret, seed=42)

recovered = stega.decode(stego_image, "a sunset over mountains", message_length=len(secret), seed=42)
print(LDStega.bits_to_text(recovered))
```

```bash
# Benchmark against messenger-style attacks
python benchmark_ldstega.py --suite messenger --seeds 42,123,777
```

---

## Benchmarking

The benchmark library covers seven test suites:

| Suite | Description |
|-------|-------------|
| `individual` | 60+ isolated transforms (JPEG, resize, blur, noise, color jitter, crop, rotation) |
| `messenger` | Composite pipelines: Telegram, WhatsApp, Instagram, WeChat, double-JPEG, screenshot |
| `param_sweep` | gamma × message length grid under JPEG q=75 |
| `stress` | Extreme attacks: JPEG q=10, 4× downscale, heavy blur+noise, all combined |
| `ecc_comparison` | 8 ECC codes × 8 attacks |
| `ecc_capacity` | 6 ECC codes × 5 message lengths |
| `transfer_format` | 6 transfer formats × 4 gammas × 5 message lengths |

```bash
python benchmark_ldstega.py --suite individual --seeds 42
python benchmark_ldstega.py --suite messenger --seeds 42,123,777
python benchmark_ldstega.py --suite param_sweep --seeds 42
python benchmark_ldstega.py --suite stress --seeds 42
python benchmark_ldstega.py --suite ecc_comparison --seeds 42
python benchmark_ldstega.py --suite transfer_format --seeds 42
python benchmark_ldstega.py --suite all --seeds 42,123,777 --output-dir results/full
```

Results are written incrementally to CSV and JSON in the output directory — safe to interrupt and resume.

---

## Notebooks

See [notebooks/README.md](notebooks/README.md) for a guide to the two included notebooks.

---

## Repository layout

```
ldstega-unofficial/
├── ldstega.py             # Core implementation: LDStega, StegoConfig, ECC codecs
├── benchmark_ldstega.py   # CLI benchmark runner
├── benchmark/             # Test suites, transforms, metrics, orchestration
└── notebooks/
    ├── 01_demo.ipynb                  # Encode/decode walkthrough
    └── 02_bit_accuracy_analysis.ipynb # Statistical analysis of results
```

---

## Citation

If you build on this work, please cite the original paper:

```bibtex
@inproceedings{jiang2024ldstega,
  title     = {LDStega: Practical and Robust Generative Image Steganography based on Latent Diffusion Models},
  author    = {Jiang, Yuwei and Li, Zhongliang and Qian, Zhenxing},
  booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
  year      = {2024},
  doi       = {10.1145/3664647.3681063}
}
```

If you use this specific implementation:

```bibtex
@software{agalyan2025ldstega,
  title  = {LDStega: Unofficial PyTorch Implementation},
  author = {Agalyan, Armin},
  year   = {2025},
  url    = {https://github.com/a3agalyan/ldstega-unofficial}
}
```
