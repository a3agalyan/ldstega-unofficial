"""
Benchmark metrics: BER, PSNR, SSIM.
"""

import numpy as np
from PIL import Image
from typing import List


def compute_ber(original_bits: List[int], extracted_bits: List[int]) -> float:
    """Bit Error Rate = fraction of mismatched bits."""
    orig = np.array(original_bits, dtype=np.int32)
    ext = np.array(extracted_bits, dtype=np.int32)
    n = min(len(orig), len(ext))
    if n == 0:
        return 1.0
    mismatched = np.sum(orig[:n] != ext[:n])
    # Count missing bits as errors too
    mismatched += abs(len(orig) - len(ext))
    return float(mismatched) / len(orig)


def compute_psnr(image_a: Image.Image, image_b: Image.Image) -> float:
    """Peak Signal-to-Noise Ratio between two PIL images (dB)."""
    if image_b.size != image_a.size:
        image_b = image_b.resize(image_a.size, Image.LANCZOS)
    a = np.array(image_a).astype(np.float64)
    b = np.array(image_b).astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def compute_ssim(image_a: Image.Image, image_b: Image.Image) -> float:
    """Structural Similarity Index between two PIL images."""
    from skimage.metrics import structural_similarity
    if image_b.size != image_a.size:
        image_b = image_b.resize(image_a.size, Image.LANCZOS)
    a = np.array(image_a)
    b = np.array(image_b)
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))
