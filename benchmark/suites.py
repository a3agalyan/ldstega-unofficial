"""
Test suite definitions.

Each suite returns a list of TestConfig objects that the runner iterates over.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List

from benchmark.transforms import (
    identity,
    chain_transforms,
    make_jpeg, make_webp, make_resize, make_blur, make_noise,
    make_color_jitter, make_crop, make_rotation, make_quantize,
)

DEFAULT_GAMMA = 0.3
DEFAULT_MSG_LEN = 512


@dataclass
class TestConfig:
    suite: str
    name: str
    transform_fn: Callable
    transform_params: Dict
    gamma: float = DEFAULT_GAMMA
    message_length: int = DEFAULT_MSG_LEN
    transfer_format: str = "pil"  # "none", "pil", "png", "jpeg", "jpeg:Q"


# -----------------------------------------------------------------------
# Suite builders
# -----------------------------------------------------------------------

def _build_individual() -> List[TestConfig]:
    """Suite 1: each transform in isolation at varying intensity."""
    configs: List[TestConfig] = []

    # Baseline (no attack)
    configs.append(TestConfig(
        suite="individual", name="identity",
        transform_fn=identity, transform_params={"transform": "none"},
    ))

    # JPEG compression
    for q in [95, 85, 75, 60, 50]:
        configs.append(TestConfig(
            suite="individual", name=f"jpeg_q{q}",
            transform_fn=make_jpeg(q),
            transform_params={"transform": "jpeg", "quality": q},
        ))

    # WebP compression
    for q in [90, 75, 50]:
        configs.append(TestConfig(
            suite="individual", name=f"webp_q{q}",
            transform_fn=make_webp(q),
            transform_params={"transform": "webp", "quality": q},
        ))

    # Resize (down then up)
    for s in [0.9, 0.75, 0.5, 0.25]:
        configs.append(TestConfig(
            suite="individual", name=f"resize_{s}",
            transform_fn=make_resize(s),
            transform_params={"transform": "resize", "scale": s},
        ))

    # Gaussian blur
    for k, sigma in [(3, 0.5), (3, 1.0), (5, 1.0), (5, 2.0), (7, 2.0), (7, 3.0), (11, 3.0)]:
        configs.append(TestConfig(
            suite="individual", name=f"blur_k{k}_s{sigma}",
            transform_fn=make_blur(k, sigma),
            transform_params={"transform": "gaussian_blur", "kernel": k, "sigma": sigma},
        ))

    # Gaussian noise
    for std in [0.005, 0.01, 0.02, 0.05, 0.1]:
        configs.append(TestConfig(
            suite="individual", name=f"noise_{std}",
            transform_fn=make_noise(std),
            transform_params={"transform": "gaussian_noise", "std": std},
        ))

    # Color jitter – brightness
    for b in [0.05, 0.1, 0.2, -0.05, -0.1, -0.2]:
        configs.append(TestConfig(
            suite="individual", name=f"bright_{b:+.2f}",
            transform_fn=make_color_jitter(brightness=b),
            transform_params={"transform": "brightness", "brightness": b},
        ))

    # Color jitter – contrast
    for c in [0.05, 0.1, 0.2, -0.05, -0.1, -0.2]:
        configs.append(TestConfig(
            suite="individual", name=f"contrast_{c:+.2f}",
            transform_fn=make_color_jitter(contrast=c),
            transform_params={"transform": "contrast", "contrast": c},
        ))

    # Color jitter – saturation
    for sat in [0.1, 0.2, -0.1, -0.2]:
        configs.append(TestConfig(
            suite="individual", name=f"saturation_{sat:+.2f}",
            transform_fn=make_color_jitter(saturation=sat),
            transform_params={"transform": "saturation", "saturation": sat},
        ))

    # Center crop + resize
    for r in [0.95, 0.9, 0.8, 0.7]:
        configs.append(TestConfig(
            suite="individual", name=f"crop_{r}",
            transform_fn=make_crop(r),
            transform_params={"transform": "center_crop", "ratio": r},
        ))

    # Rotation
    for a in [1, 2, 5, 10]:
        configs.append(TestConfig(
            suite="individual", name=f"rotate_{a}deg",
            transform_fn=make_rotation(a),
            transform_params={"transform": "rotation", "angle": a},
        ))

    return configs


def _build_messenger() -> List[TestConfig]:
    """Suite 2: realistic messenger platform pipelines."""
    configs: List[TestConfig] = []

    # Telegram: resize(0.75) -> JPEG(85)
    configs.append(TestConfig(
        suite="messenger", name="telegram",
        transform_fn=chain_transforms([make_resize(0.75), make_jpeg(85)]),
        transform_params={"platform": "telegram", "resize": 0.75, "jpeg": 85},
    ))

    # WhatsApp: resize(0.5) -> JPEG(60) -> slight brightness shift
    configs.append(TestConfig(
        suite="messenger", name="whatsapp",
        transform_fn=chain_transforms([
            make_resize(0.5), make_jpeg(60), make_color_jitter(brightness=0.03),
        ]),
        transform_params={"platform": "whatsapp", "resize": 0.5, "jpeg": 60,
                          "brightness": 0.03},
    ))

    # Instagram: resize(0.7) -> JPEG(75) -> color jitter
    configs.append(TestConfig(
        suite="messenger", name="instagram",
        transform_fn=chain_transforms([
            make_resize(0.7), make_jpeg(75),
            make_color_jitter(brightness=0.05, saturation=0.1),
        ]),
        transform_params={"platform": "instagram", "resize": 0.7, "jpeg": 75,
                          "brightness": 0.05, "saturation": 0.1},
    ))

    # WeChat: resize(0.75) -> JPEG(70) -> blur(k=3, sigma=0.5)
    configs.append(TestConfig(
        suite="messenger", name="wechat",
        transform_fn=chain_transforms([
            make_resize(0.75), make_jpeg(70), make_blur(3, 0.5),
        ]),
        transform_params={"platform": "wechat", "resize": 0.75, "jpeg": 70,
                          "blur_sigma": 0.5},
    ))

    # Double compression: JPEG(85) -> JPEG(70)
    configs.append(TestConfig(
        suite="messenger", name="double_jpeg",
        transform_fn=chain_transforms([make_jpeg(85), make_jpeg(70)]),
        transform_params={"platform": "double_jpeg", "jpeg1": 85, "jpeg2": 70},
    ))

    # Screenshot: blur(0.3) -> resize(0.9) -> JPEG(90) -> quantize(64)
    configs.append(TestConfig(
        suite="messenger", name="screenshot",
        transform_fn=chain_transforms([
            make_blur(3, 0.3), make_resize(0.9), make_jpeg(90), make_quantize(64),
        ]),
        transform_params={"platform": "screenshot", "blur_sigma": 0.3,
                          "resize": 0.9, "jpeg": 90, "quantize_levels": 64},
    ))

    return configs


def _build_param_sweep() -> List[TestConfig]:
    """Suite 3: gamma x message_length grid under JPEG(75)."""
    configs: List[TestConfig] = []
    jpeg75 = make_jpeg(75)

    for gamma in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        for msg_len in [256, 512, 1024, 2048, 4096]:
            configs.append(TestConfig(
                suite="param_sweep",
                name=f"gamma{gamma}_len{msg_len}",
                transform_fn=jpeg75,
                transform_params={"transform": "jpeg", "quality": 75,
                                  "gamma": gamma, "message_length": msg_len},
                gamma=gamma,
                message_length=msg_len,
            ))

    return configs


def _build_stress() -> List[TestConfig]:
    """Suite 4: extreme attacks to find breaking points."""
    configs: List[TestConfig] = []

    configs.append(TestConfig(
        suite="stress", name="jpeg_q10",
        transform_fn=make_jpeg(10),
        transform_params={"transform": "jpeg", "quality": 10},
    ))
    configs.append(TestConfig(
        suite="stress", name="resize_0.25",
        transform_fn=make_resize(0.25),
        transform_params={"transform": "resize", "scale": 0.25},
    ))
    configs.append(TestConfig(
        suite="stress", name="heavy_blur",
        transform_fn=make_blur(11, 3.0),
        transform_params={"transform": "gaussian_blur", "kernel": 11, "sigma": 3.0},
    ))
    configs.append(TestConfig(
        suite="stress", name="heavy_noise",
        transform_fn=make_noise(0.1),
        transform_params={"transform": "gaussian_noise", "std": 0.1},
    ))
    configs.append(TestConfig(
        suite="stress", name="all_combined",
        transform_fn=chain_transforms([
            make_jpeg(10), make_resize(0.25), make_blur(11, 3.0), make_noise(0.1),
        ]),
        transform_params={"transform": "combined_stress"},
    ))

    return configs


def _build_transfer_format_sweep() -> List[TestConfig]:
    """Suite 7: transfer format x message_length x gamma grid.

    Tests how different transfer formats affect D calculation accuracy
    and bit recovery, with no additional attack (identity transform).
    """
    configs: List[TestConfig] = []

    formats = ["none", "pil", "png", "jpeg:95", "jpeg:75", "jpeg:50"]
    gammas = [0.2, 0.3, 0.5, 0.8]
    msg_lens = [256, 512, 1024, 2048, 4096]

    for fmt in formats:
        for gamma in gammas:
            for msg_len in msg_lens:
                fmt_label = fmt.replace(":", "_q")  # "jpeg:85" -> "jpeg_q85"
                configs.append(TestConfig(
                    suite="transfer_format",
                    name=f"{fmt_label}_gamma{gamma}_len{msg_len}",
                    transform_fn=identity,
                    transform_params={
                        "transfer_format": fmt,
                        "gamma": gamma,
                        "message_length": msg_len,
                    },
                    gamma=gamma,
                    message_length=msg_len,
                    transfer_format=fmt,
                ))

    return configs


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

_SUITE_BUILDERS = {
    "individual": _build_individual,
    "messenger": _build_messenger,
    "param_sweep": _build_param_sweep,
    "stress": _build_stress,
    "transfer_format": _build_transfer_format_sweep,
}


def list_suites() -> List[str]:
    return list(_SUITE_BUILDERS.keys())


def build_suite(name: str) -> List[TestConfig]:
    """Build a named suite. Use 'all' to get every suite concatenated."""
    if name == "all":
        configs = []
        for builder in _SUITE_BUILDERS.values():
            configs.extend(builder())
        return configs
    if name not in _SUITE_BUILDERS:
        raise ValueError(f"Unknown suite '{name}'. Choose from: {list_suites() + ['all']}")
    return _SUITE_BUILDERS[name]()
