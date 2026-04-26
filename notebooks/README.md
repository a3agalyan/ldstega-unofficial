# Notebooks

**[01_demo.ipynb](01_demo.ipynb)** — Full demo visualizes the latent discrepancy heatmap and MS interval distribution, decodes the message, and shows a side-by-side comparison of original vs. stego image. 

**[02_bit_accuracy_analysis.ipynb](02_bit_accuracy_analysis.ipynb)** — Benchmarks for BER vs. gamma, BER vs. message length, gamma vs. image quality (PSNR/SSIM). This notebook produces the charts referenced in [WRITEUP.md](../WRITEUP.md) — run the benchmarks first and point the notebook at your `results/` directory.

## Running the notebooks

```bash
pip install -e ..          # install from repo root
jupyter notebook           # or: jupyter lab
```

The demo notebook downloads model weights (~4-12 GB) on first run.
The analysis notebook expects benchmark output in `../results/` — run `benchmark_ldstega.py` first.

