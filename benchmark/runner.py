"""
Benchmark runner: encode -> transform -> decode -> measure -> log.

Handles model loading, stego-image caching, gamma re-wrapping, and
incremental CSV logging so partial results survive crashes.
"""

import csv
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from ldstega import LDStega, StegoConfig, TransferFormat
from benchmark.metrics import compute_ber, compute_psnr, compute_ssim
from benchmark.suites import TestConfig


class BenchmarkRunner:
    """Orchestrates the full benchmark pipeline.

    Loads the model once, caches stego images keyed by (gamma, msg_len, seed),
    and streams results to CSV row-by-row.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        prompt: str = "A beautiful sunset over mountains",
        seeds: List[int] = None,
        output_dir: str = "results",
        device: str = "auto",
        show_progress: bool = True,
    ):
        self.model_id = model_id
        self.prompt = prompt
        self.seeds = seeds or [42, 123, 777]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.show_progress = show_progress

        # Will hold the active LDStega instance and its gamma / transfer format
        self._stega: Optional[LDStega] = None
        self._current_gamma: Optional[float] = None
        self._current_transfer_format: Optional[str] = None

        # Cache: keyed by (gamma, msg_len, seed, transfer_format)
        self._cache: Dict = {}

        # CSV setup
        self._csv_path = self.output_dir / "benchmark_results.csv"
        self._fieldnames = [
            "suite", "name", "transform_params",
            "seed", "gamma", "message_length", "message_chars",
            "transfer_format",
            "ber", "bit_accuracy",
            "psnr_stego", "ssim_stego",
            "psnr_transformed", "ssim_transformed",
            "encode_time_s", "decode_time_s",
        ]
        self._results: List[Dict] = []

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def _ensure_stega(self, gamma: float, transfer_format: str = "pil") -> LDStega:
        """Return an LDStega with the requested gamma and transfer format."""
        tf = TransferFormat.parse(transfer_format)
        if self._stega is None:
            config = StegoConfig(
                model_id=self.model_id,
                truncation_gamma=gamma,
                device=self.device,
                transfer_format=tf,
            )
            self._stega = LDStega(config=config)
            self._current_gamma = gamma
            self._current_transfer_format = transfer_format
        else:
            if self._current_gamma != gamma:
                self._stega.gamma = gamma
                self._stega.config.truncation_gamma = gamma
                self._current_gamma = gamma
            if self._current_transfer_format != transfer_format:
                self._stega.config.transfer_format = tf
                self._current_transfer_format = transfer_format
        return self._stega

    # ------------------------------------------------------------------
    # Encoding with cache
    # ------------------------------------------------------------------

    def _get_stego(
        self, gamma: float, msg_len: int, seed: int,
    ) -> Tuple[Image.Image, Image.Image, List[int]]:
        """Encode or return cached (stego_image, original_image, secret_bits)."""
        key = (gamma, msg_len, seed)
        if key in self._cache:
            return self._cache[key]

        stega = self._ensure_stega(gamma)

        # Deterministic secret bits from seed
        rng = np.random.default_rng(seed)
        secret_bits = rng.integers(0, 2, size=msg_len).tolist()

        t0 = time.perf_counter()
        stego_image = stega.encode(
            self.prompt, secret_bits, seed, show_progress=self.show_progress,
        )
        encode_time = time.perf_counter() - t0

        original_image = stega._last_original_image

        self._cache[key] = (stego_image, original_image, secret_bits)
        # Store encode time separately so it can be looked up
        self._cache[("_time", key)] = encode_time
        return stego_image, original_image, secret_bits

    # ------------------------------------------------------------------
    # CSV I/O
    # ------------------------------------------------------------------

    def _init_csv(self):
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()

    def _append_csv(self, row: Dict):
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow(row)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, configs: List[TestConfig]) -> List[Dict]:
        """Execute all test configs and return list of result dicts."""
        self._init_csv()

        total = len(configs) * len(self.seeds)
        pbar = tqdm(total=total, desc="Benchmark", disable=not self.show_progress)

        for cfg in configs:
            for seed in self.seeds:
                row = self._run_single(cfg, seed)
                self._results.append(row)
                self._append_csv(row)
                pbar.update(1)
                pbar.set_postfix(
                    suite=cfg.suite, test=cfg.name, ber=f"{row['ber']:.4f}",
                )

        pbar.close()

        # Save full results as JSON too
        json_path = self.output_dir / "benchmark_results.json"
        with open(json_path, "w") as f:
            json.dump(self._results, f, indent=2, default=str)

        self._print_summary()
        return self._results

    def _run_single(self, cfg: TestConfig, seed: int) -> Dict:
        """Run one (config, seed) pair and return a result dict."""
        gamma = cfg.gamma
        msg_len = cfg.message_length
        transfer_fmt = getattr(cfg, "transfer_format", "pil")

        # Deterministic secret bits from seed
        rng = np.random.default_rng(seed)
        secret_bits = rng.integers(0, 2, size=msg_len).tolist()

        # Stego encode (or fetch from cache)
        cache_key = (gamma, msg_len, seed, transfer_fmt)
        if cache_key in self._cache:
            stego_output, original_image, _ = self._cache[cache_key]
            encode_time = self._cache.get(("_time", cache_key), 0.0)
        else:
            stega = self._ensure_stega(gamma, transfer_fmt)
            t0 = time.perf_counter()
            stego_output = stega.encode(
                self.prompt, secret_bits, seed, show_progress=self.show_progress,
            )
            encode_time = time.perf_counter() - t0
            original_image = stega._last_original_image
            self._cache[cache_key] = (stego_output, original_image, secret_bits)
            self._cache[("_time", cache_key)] = encode_time

        # Handle tensor output from "none" format
        is_tensor = isinstance(stego_output, torch.Tensor)
        if is_tensor:
            stego_pil = LDStega._tensor_to_pil(stego_output)
            decode_input = stego_output  # pass tensor directly to decode
        else:
            stego_pil = stego_output
            decode_input = stego_output

        # Apply transform (on PIL image)
        transformed_pil = cfg.transform_fn(stego_pil)

        # For non-"none" formats, use the transformed PIL as decode input
        if not is_tensor:
            decode_input = transformed_pil

        # Stego decode
        stega = self._ensure_stega(gamma, transfer_fmt)

        # Ensure correct size for decode (some transforms may change size)
        if isinstance(decode_input, Image.Image):
            target_size = tuple(stega.config.image_size)
            if decode_input.size != target_size:
                decode_input = decode_input.resize(target_size, Image.BICUBIC)

        t0 = time.perf_counter()
        extracted_bits = stega.decode(
            decode_input, self.prompt, msg_len, seed, show_progress=False,
        )
        decode_time = time.perf_counter() - t0

        ber = compute_ber(secret_bits, extracted_bits)

        # Image quality metrics (convert to PIL if needed)
        if isinstance(original_image, torch.Tensor):
            original_pil = LDStega._tensor_to_pil(original_image)
        else:
            original_pil = original_image
        psnr_stego = compute_psnr(original_pil, stego_pil)
        ssim_stego = compute_ssim(original_pil, stego_pil)
        psnr_trans = compute_psnr(original_pil, transformed_pil)
        ssim_trans = compute_ssim(original_pil, transformed_pil)

        return {
            "suite": cfg.suite,
            "name": cfg.name,
            "transform_params": json.dumps(cfg.transform_params),
            "seed": seed,
            "gamma": gamma,
            "message_length": msg_len,
            "message_chars": msg_len // 8,
            "transfer_format": transfer_fmt,
            "ber": round(ber, 6),
            "bit_accuracy": round(1.0 - ber, 6),
            "psnr_stego": round(psnr_stego, 2),
            "ssim_stego": round(ssim_stego, 4),
            "psnr_transformed": round(psnr_trans, 2),
            "ssim_transformed": round(ssim_trans, 4),
            "encode_time_s": round(encode_time, 2),
            "decode_time_s": round(decode_time, 2),
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _print_summary(self):
        """Print a quick summary table to stdout."""
        import pandas as pd
        if not self._results:
            return
        df = pd.DataFrame(self._results)
        print("\n" + "=" * 72)
        print("BENCHMARK SUMMARY")
        print("=" * 72)

        for suite in df["suite"].unique():
            sub = df[df["suite"] == suite]
            print(f"\n--- {suite.upper()} ---")
            agg_dict = {
                "msg_bits": ("message_length", "first"),
                "msg_chars": ("message_chars", "first"),
                "ber_mean": ("ber", "mean"),
                "ber_std": ("ber", "std"),
                "bit_acc_mean": ("bit_accuracy", "mean"),
                "psnr_trans": ("psnr_transformed", "mean"),
                "ssim_trans": ("ssim_transformed", "mean"),
            }
            summary = (
                sub.groupby("name")
                .agg(**agg_dict)
                .sort_values("ber_mean")
            )
            print(summary.to_string())

        print(f"\nFull results saved to: {self._csv_path}")
        print(f"JSON results saved to: {self.output_dir / 'benchmark_results.json'}")
