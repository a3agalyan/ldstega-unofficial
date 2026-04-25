#!/usr/bin/env python3
"""
LDStega Robustness Benchmark — CLI entry point.

Usage:
    python benchmark_ldstega.py                    # run all suites
    python benchmark_ldstega.py --suite individual # single suite
    python benchmark_ldstega.py --suite messenger --seeds 42,123
    python benchmark_ldstega.py --list-suites      # show available suites
"""

import argparse
import sys

from benchmark.suites import build_suite, list_suites
from benchmark.runner import BenchmarkRunner


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark LDStega robustness against image transformations",
    )
    p.add_argument(
        "--suite", default="all",
        help="Suite to run: individual, messenger, param_sweep, stress, "
             "transfer_format, all (default: all)",
    )
    p.add_argument(
        "--model", default="runwayml/stable-diffusion-v1-5",
        help="HuggingFace model ID (default: runwayml/stable-diffusion-v1-5)",
    )
    p.add_argument(
        "--prompt", default="A beautiful sunset over mountains",
        help="Text prompt for image generation",
    )
    p.add_argument(
        "--seeds", default="42,123,777",
        help="Comma-separated list of random seeds (default: 42,123,777)",
    )
    p.add_argument(
        "--output-dir", default="results",
        help="Directory for CSV/JSON output (default: results/)",
    )
    p.add_argument(
        "--device", default="auto",
        help="Device: cuda, cpu, or auto (default: auto)",
    )
    p.add_argument(
        "--transfer-format", default=None,
        help="Override transfer format for all configs: none, pil, png, jpeg, "
             "jpeg:85 (default: per-config, typically 'pil')",
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="Disable stego-image caching (re-encode for every test)",
    )
    p.add_argument(
        "--list-suites", action="store_true",
        help="List available test suites and exit",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress bars",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_suites:
        print("Available suites:", ", ".join(list_suites() + ["all"]))
        sys.exit(0)

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    configs = build_suite(args.suite)

    # Global --transfer-format override
    if args.transfer_format:
        for cfg in configs:
            cfg.transfer_format = args.transfer_format

    tf_label = args.transfer_format or "(per-config)"
    print(f"Model:    {args.model}")
    print(f"Suite:    {args.suite}")
    print(f"Transfer: {tf_label}")
    print(f"Configs:  {len(configs)}")
    print(f"Seeds:    {seeds}")
    print(f"Total runs: {len(configs) * len(seeds)}")
    print(f"Output:   {args.output_dir}/")
    print()

    runner = BenchmarkRunner(
        model_id=args.model,
        prompt=args.prompt,
        seeds=seeds,
        output_dir=args.output_dir,
        device=args.device,
        show_progress=not args.quiet,
    )

    results = runner.run(configs)
    print(f"\nDone. {len(results)} results written.")


if __name__ == "__main__":
    main()
