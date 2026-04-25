# Notebooks

## Start here

**[01_demo.ipynb](01_demo.ipynb)** — Full encode/decode walkthrough. Initializes a Stable Diffusion pipeline, hides a message in a generated image, visualizes the latent discrepancy heatmap and MS interval distribution, decodes the message, and shows a side-by-side comparison of original vs. stego image. No prior reading required.

**[02_bit_accuracy_analysis.ipynb](02_bit_accuracy_analysis.ipynb)** — Statistical analysis of benchmark results. Covers bit accuracy by transform type, BER vs. gamma, BER vs. message length, gamma vs. image quality (PSNR/SSIM), and a correlation matrix of all variables. This notebook produces the charts referenced in [RESULTS.md](../RESULTS.md) — run the benchmarks first and point the notebook at your `results/` directory.

## Running the notebooks

```bash
pip install -e ..          # install from repo root
jupyter notebook           # or: jupyter lab
```

The demo notebook downloads model weights (~4 GB) on first run.
The analysis notebook expects benchmark output in `../results/` — run `benchmark_ldstega.py` first.

## assets/

Static images committed to the repo, embedded in README.md and RESULTS.md.
