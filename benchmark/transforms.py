"""
Image transforms simulating messenger platform perturbations.

Every public function has signature: (PIL.Image) -> PIL.Image
Use functools.partial or the factory helpers to bind parameters.
"""

import io
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from functools import partial
from typing import Callable, List


# ---------------------------------------------------------------------------
# JPEG / WebP compression
# ---------------------------------------------------------------------------

def jpeg_compress(image: Image.Image, quality: int = 75) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def webp_compress(image: Image.Image, quality: int = 75) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="WEBP", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------------------
# Geometric transforms
# ---------------------------------------------------------------------------

def resize_attack(image: Image.Image, scale: float = 0.5,
                  interpolation: int = Image.BICUBIC) -> Image.Image:
    """Down-scale then up-scale back to original size."""
    w, h = image.size
    small = image.resize((int(w * scale), int(h * scale)), interpolation)
    return small.resize((w, h), interpolation)


def center_crop_resize(image: Image.Image, crop_ratio: float = 0.9) -> Image.Image:
    """Crop the center portion, then resize back to original dimensions."""
    w, h = image.size
    new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    cropped = image.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BICUBIC)


def rotation_attack(image: Image.Image, angle: float = 2.0) -> Image.Image:
    """Rotate by a small angle, crop black borders, resize back."""
    w, h = image.size
    rotated = image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(128, 128, 128))
    # Crop back to remove expanded borders (center crop to original aspect)
    rw, rh = rotated.size
    left = (rw - w) // 2
    top = (rh - h) // 2
    cropped = rotated.crop((left, top, left + w, top + h))
    return cropped.resize((w, h), Image.BICUBIC)


# ---------------------------------------------------------------------------
# Pixel-domain noise & blur
# ---------------------------------------------------------------------------

def gaussian_noise(image: Image.Image, std: float = 0.01) -> Image.Image:
    """Add Gaussian noise with given standard deviation (image in [0,1] scale)."""
    arr = np.array(image).astype(np.float32) / 255.0
    rng = np.random.default_rng(0)
    noise = rng.normal(0, std, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 1)
    return Image.fromarray((noisy * 255).round().astype(np.uint8))


def gaussian_blur(image: Image.Image, kernel_size: int = 3,
                  sigma: float = 1.0) -> Image.Image:
    """Apply Gaussian blur via PIL."""
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def gaussian_blur_cv2(image: Image.Image, kernel_size: int = 3,
                      sigma: float = 1.0) -> Image.Image:
    """Apply Gaussian blur via OpenCV (more control over kernel size)."""
    import cv2
    arr = np.array(image)
    # kernel_size must be odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(arr, (kernel_size, kernel_size), sigma)
    return Image.fromarray(blurred)


# ---------------------------------------------------------------------------
# Color transforms
# ---------------------------------------------------------------------------

def color_jitter(image: Image.Image, brightness: float = 0.0,
                 contrast: float = 0.0, saturation: float = 0.0) -> Image.Image:
    """Adjust brightness, contrast, saturation.

    Values are *additive* factors: output = base_factor + value.
    So brightness=0.1 means 10% brighter, brightness=-0.1 means 10% darker.
    """
    if brightness != 0.0:
        image = ImageEnhance.Brightness(image).enhance(1.0 + brightness)
    if contrast != 0.0:
        image = ImageEnhance.Contrast(image).enhance(1.0 + contrast)
    if saturation != 0.0:
        image = ImageEnhance.Color(image).enhance(1.0 + saturation)
    return image


def color_quantize(image: Image.Image, levels: int = 64) -> Image.Image:
    """Reduce color depth by quantizing to fewer levels per channel."""
    arr = np.array(image).astype(np.float32)
    step = 256.0 / levels
    quantized = (np.floor(arr / step) * step + step / 2).clip(0, 255).astype(np.uint8)
    return Image.fromarray(quantized)


# ---------------------------------------------------------------------------
# Identity (baseline)
# ---------------------------------------------------------------------------

def identity(image: Image.Image) -> Image.Image:
    """No-op transform (baseline measurement)."""
    return image.copy()


# ---------------------------------------------------------------------------
# Chaining combinator
# ---------------------------------------------------------------------------

def chain_transforms(transforms: List[Callable]) -> Callable:
    """Compose a sequence of (Image -> Image) transforms into one callable."""
    def chained(image: Image.Image) -> Image.Image:
        for t in transforms:
            image = t(image)
        return image
    return chained


# ---------------------------------------------------------------------------
# Factory helpers – bind parameters, return (Image -> Image) callables
# ---------------------------------------------------------------------------

def make_jpeg(quality: int) -> Callable:
    return partial(jpeg_compress, quality=quality)

def make_webp(quality: int) -> Callable:
    return partial(webp_compress, quality=quality)

def make_resize(scale: float) -> Callable:
    return partial(resize_attack, scale=scale)

def make_blur(kernel_size: int, sigma: float) -> Callable:
    return partial(gaussian_blur_cv2, kernel_size=kernel_size, sigma=sigma)

def make_noise(std: float) -> Callable:
    return partial(gaussian_noise, std=std)

def make_color_jitter(brightness: float = 0.0, contrast: float = 0.0,
                      saturation: float = 0.0) -> Callable:
    return partial(color_jitter, brightness=brightness, contrast=contrast,
                   saturation=saturation)

def make_crop(ratio: float) -> Callable:
    return partial(center_crop_resize, crop_ratio=ratio)

def make_rotation(angle: float) -> Callable:
    return partial(rotation_attack, angle=angle)

def make_quantize(levels: int) -> Callable:
    return partial(color_quantize, levels=levels)
