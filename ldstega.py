"""
LDStega: Latent Diffusion Steganography

Unofficial implementation of "LDStega: Practical and Robust Generative Image
Steganography based on Latent Diffusion Models" (ACM MM '24).

This module provides steganographic capabilities using Latent Diffusion Models,
allowing secret binary data to be hidden within generated images.
"""

import io
import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy.stats import truncnorm
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, LDMTextToImagePipeline, DiffusionPipeline, DDIMScheduler, AutoencoderKL
from diffusers.models import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

# Fix compatibility between diffusers' LDMBertModel and transformers 5.x:
# LDMBertModel.__init__ doesn't call post_init(), which transformers 5.x
# requires to set all_tied_weights_keys.
from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion import LDMBertModel as _LDMBertModel
_original_ldmbert_init = _LDMBertModel.__init__
def _patched_ldmbert_init(self, config):
    _original_ldmbert_init(self, config)
    self.post_init()
_LDMBertModel.__init__ = _patched_ldmbert_init

CUDA_SEED = 11
# =====================================================================
# Transfer Format
# =====================================================================

@dataclass
class TransferFormat:
    """Specifies how the stego image is transmitted between sender and receiver.

    The same format distortion is applied during the VAE round-trip for D
    calculation, so that sender and receiver agree on position classification.

    Attributes:
        kind: "none" (raw float tensor), "pil" (uint8 quantization),
              "png" (uint8 + PNG lossless), "jpeg" (uint8 + JPEG lossy)
        jpeg_quality: JPEG quality 1-100, only used when kind="jpeg"
    """
    kind: str = "pil"
    jpeg_quality: int = 95

    _VALID_KINDS = {"none", "pil", "png", "jpeg"}

    def __post_init__(self):
        if self.kind not in self._VALID_KINDS:
            raise ValueError(
                f"Unknown transfer format '{self.kind}'. Valid: {sorted(self._VALID_KINDS)}"
            )
        if self.kind == "jpeg" and not (1 <= self.jpeg_quality <= 100):
            raise ValueError(f"JPEG quality must be 1-100, got {self.jpeg_quality}")

    @staticmethod
    def parse(spec: str) -> "TransferFormat":
        """Parse a string like 'none', 'pil', 'png', 'jpeg', or 'jpeg:85'."""
        if spec.startswith("jpeg:"):
            return TransferFormat(kind="jpeg", jpeg_quality=int(spec.split(":")[1]))
        if spec == "jpeg":
            return TransferFormat(kind="jpeg", jpeg_quality=95)
        return TransferFormat(kind=spec)

    @property
    def label(self) -> str:
        """Short string for logs/filenames."""
        if self.kind == "jpeg":
            return f"jpeg_q{self.jpeg_quality}"
        return self.kind


# =====================================================================
# LDStega Configuration & Core
# =====================================================================

@dataclass
class StegoConfig:
    """
    Configuration for LDStega steganography.

    Attributes:
        model_id: HuggingFace model ID for Stable Diffusion
        image_size: Output image size (width, height)
        latent_size: Latent space size (height, width)
        latent_channels: Number of latent channels
        truncation_gamma: Truncation factor for Gaussian sampling (γ in paper)
        num_inference_steps: Number of DDIM diffusion steps
        guidance_scale: Classifier-free guidance scale
        device: Device to run on ("cuda", "cpu", or "auto")
        torch_dtype: PyTorch dtype for model weights
        ms_intervals: MS intervals for position classification by discrepancy
        eta: Noise scale for DDIM scheduler step. 0.0 = deterministic (pure DDIM),
            1.0 = full DDPM-like stochasticity. Must be > 0 for the mapping function H
            to draw meaningful truncated-Gaussian samples; 1.0 is recommended.
        message_length: Number of secret bits to hide. Must be set to the same value
            at encode and decode time. Used by _select_positions() to pre-determine
            exactly which latent positions carry bits.
    """
    model_id: str = "runwayml/stable-diffusion-v1-5"
    image_size: Tuple[int, int] = (512, 512)
    latent_size: Tuple[int, int] = (64, 64)
    latent_channels: int = 4
    truncation_gamma: float = 0.3
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    device: str = "auto"
    torch_dtype: torch.dtype = torch.float16
    transfer_format: TransferFormat = field(default_factory=lambda: TransferFormat("pil"))
    eta: float = 1.0
    ms_intervals: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 0.05),    # MS_0
        (0.05, 0.1),    # MS_1
        (0.1, 0.15),    # MS_2
        (0.15, 0.2),    # MS_3
        (0.2, 0.25),    # MS_4
        (0.25, 0.3),    # MS_5
        (0.3, float('inf')),  # MS_6
    ])
    message_length: int = 0  # bits to hide; must match len(secret_bits) at encode time

    def __post_init__(self):
        """Resolve 'auto' device to actual device."""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def capacity_bits(self) -> int:
        """Maximum steganographic capacity in bits."""
        return self.latent_size[0] * self.latent_size[1] * self.latent_channels

    @property
    def capacity_bytes(self) -> int:
        """Maximum steganographic capacity in bytes."""
        return self.capacity_bits // 8


class LDStega:
    """
    LDStega: Latent Diffusion Steganography

    Hides secret binary data within images generated by Latent Diffusion Models.
    Uses truncated Gaussian sampling to encode bits in the latent space during
    the denoising process.

    Args:
        config: StegoConfig instance with all parameters
        model_id: HuggingFace model ID (deprecated, use config)
        device: Device to run on (deprecated, use config)
        gamma: Truncation factor (deprecated, use config)
        num_inference_steps: DDIM steps (deprecated, use config)
    """

    def __init__(
        self,
        config: Optional[StegoConfig] = None,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        gamma: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
    ):
        # Support both config object and legacy parameters
        if config is None:
            config = StegoConfig(
                model_id=model_id or "runwayml/stable-diffusion-v1-5",
                device=device or "auto",
                truncation_gamma=gamma or 0.3,
                num_inference_steps=num_inference_steps or 50,
            )

        self.config = config
        self.device = config.device
        self.gamma = config.truncation_gamma
        self.num_inference_steps = config.num_inference_steps
        self.model_id = config.model_id

        # Auto-detect pipeline type (SD 1.x/2.x vs SDXL)
        self.pipe = DiffusionPipeline.from_pretrained(
            config.model_id,
            torch_dtype=config.torch_dtype,
        ).to(self.device)
        self.is_sdxl = isinstance(self.pipe, StableDiffusionXLPipeline)
        self.is_ldm = isinstance(self.pipe, LDMTextToImagePipeline)

        # Disable safety checker for SD 1.x/2.x only
        if not self.is_sdxl and not self.is_ldm and hasattr(self.pipe, 'safety_checker'):
            self.pipe.safety_checker = None
            self.pipe.requires_safety_checker = False

        # Auto-adjust config based on pipeline type
        if self.is_sdxl and config.image_size == (512, 512):
            config.image_size = (1024, 1024)
            config.latent_size = (128, 128)
        elif self.is_ldm and config.latent_size == (64, 64):
            config.image_size = (256, 256)
            config.latent_size = (32, 32)

        # Use DDIM scheduler for deterministic generation
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        # Store components for direct access
        # LDM uses 'vqvae' and 'bert'; SD uses 'vae' and 'text_encoder'
        if self.is_ldm:
            self.pipe.vqvae.to(dtype=torch.float32)
            self.vae = self.pipe.vqvae
            self.text_encoder = self.pipe.bert
        else:
            self.pipe.vae.to(dtype=torch.float32)
            self.vae = self.pipe.vae
            self.text_encoder = self.pipe.text_encoder

        self.unet = self.pipe.unet
        self.tokenizer = self.pipe.tokenizer
        self.scheduler = self.pipe.scheduler

        # SDXL has a second text encoder and tokenizer
        if self.is_sdxl:
            self.tokenizer_2 = self.pipe.tokenizer_2
            self.text_encoder_2 = self.pipe.text_encoder_2

        # State for visualization
        self._last_discrepancy = None
        self._last_original_image = None
        self._last_stego_image = None
        self._last_hidden_positions = None

    @staticmethod
    def _seed_everything(seed: int):
        """Set all RNG seeds for full CUDA reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def _tensor_to_pil(X: torch.Tensor) -> Image.Image:
        """Convert a [-1,1] float tensor (1,3,H,W) to a PIL Image."""
        img = (X / 2 + 0.5).clamp(0, 1)
        img = img.cpu().permute(0, 2, 3, 1).numpy()[0]
        img = (img * 255).round().astype(np.uint8)
        return Image.fromarray(img)

    def _pil_to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        """Convert a PIL Image to a [-1,1] float tensor on self.device."""
        arr = np.array(pil_img).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        t = (t * 2) - 1
        return t.to(self.device, dtype=torch.float32)

    def _apply_transfer_format(self, X: torch.Tensor) -> torch.Tensor:
        """Apply transfer format distortion to a VAE-decoded image tensor.

        Simulates the quantization/compression that occurs when the stego
        image is transmitted.  The same distortion is applied during the D
        calculation in both encode() and decode() so that sender and receiver
        agree on position classification.

        Args:
            X: Raw float tensor from VAE.decode(), shape (1,3,H,W), range [-1,1]

        Returns:
            Tensor of same shape with format distortion applied, range [-1,1]
        """
        fmt = self.config.transfer_format

        if fmt.kind == "none":
            return X.clone().float()

        # Convert to PIL (uint8 quantization)
        pil_img = self._tensor_to_pil(X)

        if fmt.kind == "pil":
            return self._pil_to_tensor(pil_img)

        # PNG or JPEG: serialize to bytes and reload
        buf = io.BytesIO()
        if fmt.kind == "png":
            pil_img.save(buf, format="PNG")
        elif fmt.kind == "jpeg":
            pil_img.save(buf, format="JPEG", quality=fmt.jpeg_quality)
        buf.seek(0)
        pil_img = Image.open(buf).convert("RGB")

        return self._pil_to_tensor(pil_img)

    def _format_stego_output(self, stego_raw: torch.Tensor) -> Union[Image.Image, torch.Tensor]:
        """Convert raw stego tensor to the configured output format.

        Returns:
            torch.Tensor for 'none' format, PIL.Image for all others.
        """
        fmt = self.config.transfer_format

        if fmt.kind == "none":
            return stego_raw.clone().float()

        pil_img = self._tensor_to_pil(stego_raw)

        if fmt.kind == "pil":
            return pil_img

        # PNG or JPEG: serialize and reload
        buf = io.BytesIO()
        if fmt.kind == "png":
            pil_img.save(buf, format="PNG")
        elif fmt.kind == "jpeg":
            pil_img.save(buf, format="JPEG", quality=fmt.jpeg_quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def _encode_prompt(self, prompt: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode text prompt to condition embedding.

        Returns:
            Tuple of (prompt_embeds, pooled_prompt_embeds).
            pooled_prompt_embeds is None for SD 1.5 / LDM, a tensor for SDXL.
        """
        if self.is_sdxl:
            return self._encode_prompt_sdxl(prompt)
        return self._encode_prompt_sd15(prompt)

    def _encode_prompt_sd15(self, prompt: str) -> Tuple[torch.Tensor, None]:
        """SD 1.5 / LDM prompt encoding (single text encoder)."""
        # LDM's BertTokenizer has model_max_length=512 but LDMBertModel's
        # position embeddings only support 77 positions.
        if self.is_ldm:
            max_length = self.text_encoder.config.max_position_embeddings
        else:
            max_length = self.tokenizer.model_max_length

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input_ids)[0]

        # Get unconditional embeddings for classifier-free guidance
        uncond_input = self.tokenizer(
            "",
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Concatenate for classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings, None

    def _encode_prompt_sdxl(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """SDXL prompt encoding (dual text encoders with pooled output)."""
        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]

        prompt_embeds_list = []
        uncond_embeds_list = []
        pooled_prompt_embeds = None
        pooled_uncond_embeds = None

        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            # Encode prompt
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(self.device)

            with torch.no_grad():
                outputs = text_encoder(text_input_ids, output_hidden_states=True)

            # Use penultimate hidden state
            prompt_embeds = outputs.hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)

            # Extract pooled output from the second encoder (CLIPTextModelWithProjection)
            if text_encoder is self.text_encoder_2:
                pooled_prompt_embeds = outputs[0]

            # Encode unconditional (empty string)
            uncond_inputs = tokenizer(
                "",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                uncond_outputs = text_encoder(
                    uncond_inputs.input_ids.to(self.device),
                    output_hidden_states=True,
                )

            uncond_embeds = uncond_outputs.hidden_states[-2]
            uncond_embeds_list.append(uncond_embeds)

            if text_encoder is self.text_encoder_2:
                pooled_uncond_embeds = uncond_outputs[0]

        # Concatenate embeddings from both encoders along feature dim
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        uncond_embeds = torch.concat(uncond_embeds_list, dim=-1)

        # Concatenate for classifier-free guidance (uncond first, then cond)
        text_embeddings = torch.cat([uncond_embeds, prompt_embeds])
        pooled_embeds = torch.cat([pooled_uncond_embeds, pooled_prompt_embeds])

        return text_embeddings, pooled_embeds

    def _sample_truncated_gaussian(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Sample from truncated Gaussian distribution.

        Args:
            mu: Mean values
            sigma: Standard deviation values
            lower: Lower bounds
            upper: Upper bounds
            rng: Random number generator

        Returns:
            Samples from truncated Gaussian
        """
        # Normalize bounds
        a = (lower - mu) / sigma
        b = (upper - mu) / sigma

        # Sample using scipy's truncnorm
        samples = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=1, random_state=rng)
        return samples

    def _mapping_function_H(
        self,
        bits: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Algorithm 2: Mapping function H(·)

        Maps secret bits to latent space values using truncated Gaussian sampling.

        Args:
            bits: Binary data (0 or 1) to hide
            mu: Mean of Gaussian at each position (mu_{T-1})
            sigma: Standard deviation at each position (sigma_{T-1})
            rng: Random number generator

        Returns:
            Modified latent values Z^s_T
        """
        result = np.zeros_like(mu)

        for i, bit in enumerate(bits):
            if bit == 0:
                # pool_0 = (-inf, mu - gamma)
                lower = -np.inf
                upper = mu[i] - self.gamma
            else:
                # pool_1 = (mu + gamma, +inf)
                lower = mu[i] + self.gamma
                upper = np.inf

            result[i] = self._sample_truncated_gaussian(
                np.array([mu[i]]),
                np.array([sigma[i]]),
                np.array([lower]),
                np.array([upper]),
                rng,
            )[0]

        return result

    def _classify_positions_by_discrepancy(
        self,
        D: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Classify latent positions into MS intervals based on discrepancy D.

        Args:
            D: Absolute discrepancy |Z_T - Z'_T| flattened

        Returns:
            List of position indices for each MS interval
        """
        positions_by_interval = []

        for lower, upper in self.config.ms_intervals:
            if upper == float('inf'):
                mask = D >= lower
            else:
                mask = (D >= lower) & (D < upper)
            positions_by_interval.append(np.where(mask)[0])

        return positions_by_interval

    def _select_positions(
        self,
        positions_by_interval: List[np.ndarray],
        message_length: int,
        seed: int,
    ) -> List[int]:
        """
        Select which scalar positions in the flattened latent space carry secret bits.

        Positions are drawn from MS intervals in priority order (MS_0 first, then MS_1,
        etc.), but within each interval they are shuffled pseudo-randomly so that
        modulations are spread across all channels and spatial locations rather than
        clustering in the lowest-indexed region of the image.

        Properties:
            1. Pseudo-random: positions within each interval are permuted so
               modulations don't cluster visibly in one region of the image.
            2. Deterministically reproducible from the seed alone, so decode()
               recovers the identical position list without any side-channel.
            3. The seed offset (+12345) separates the position RNG from the noise
               RNG (which uses the raw seed), preventing correlations between which
               latent scalars are chosen and the initial noise pattern.

        Args:
            positions_by_interval: output of _classify_positions_by_discrepancy(),
                a list of numpy arrays (one per MS interval).
            message_length: total number of bits to place.
            seed: the encode/decode seed (same value used for diffusion).

        Returns:
            Ordered list of flat latent indices, length == min(message_length, capacity).
        """
        rng = np.random.default_rng(seed + 12345)
        selected: List[int] = []

        for positions in positions_by_interval:
            remaining = message_length - len(selected)
            if remaining <= 0:
                break
            shuffled = rng.permutation(positions)  # random order within interval
            selected.extend(shuffled[:remaining].tolist())

        return selected

    def _get_add_time_ids(self) -> torch.Tensor:
        """Build SDXL time_ids: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]."""
        h, w = self.config.image_size
        add_time_ids = torch.tensor(
            [[h, w, 0, 0, h, w]],
            dtype=self.pipe.unet.dtype,
            device=self.device,
        )
        return add_time_ids

    def _run_diffusion_process(
        self,
        prompt_embeds: torch.Tensor,
        seed: int,
        guidance_scale: float = 7.5,
        show_progress: bool = True,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the DDIM diffusion process and capture intermediate states.

        Args:
            prompt_embeds: Text embeddings for conditioning
            seed: Random seed for reproducibility
            guidance_scale: Classifier-free guidance scale
            show_progress: Whether to show progress bar
            pooled_prompt_embeds: Pooled text embeddings (required for SDXL)

        Returns:
            Tuple of (Z_T, mu_{T-1}, sigma_{T-1})
        """
        # Set scheduler timesteps
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)

        # Initialize latent with seed
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latent_shape = (1, self.config.latent_channels,
                        self.config.latent_size[0], self.config.latent_size[1])
        latents = torch.randn(latent_shape, generator=generator, device=self.device, dtype=self.config.torch_dtype)

        # Scale initial noise
        latents = latents * self.scheduler.init_noise_sigma

        # Build added_cond_kwargs for SDXL
        added_cond_kwargs = None
        if self.is_sdxl and pooled_prompt_embeds is not None:
            add_time_ids = self._get_add_time_ids()
            # Duplicate for classifier-free guidance (uncond + cond)
            add_time_ids = torch.cat([add_time_ids] * 2)
            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            }

        # Store intermediate values
        mu_T_minus_1 = None
        sigma_T_minus_1 = None

        # Run diffusion with optional progress bar
        timesteps_iter = enumerate(self.scheduler.timesteps)
        if show_progress:
            timesteps_iter = tqdm(
                list(enumerate(self.scheduler.timesteps)),
                desc="Diffusion steps",
                leave=False
            )

        for i, t in timesteps_iter:
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

            # Classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Store values from second-to-last step (T-1)
            if i == len(self.scheduler.timesteps) - 2:
                t_prev = self.scheduler.timesteps[i + 1]
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[t_prev]
                beta_prod_t = 1 - alpha_prod_t

                # Predicted x_0
                pred_original_sample = (latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5

                # DDPM posterior sigma: sqrt((1-ā_{t-1})/(1-ā_t) * (1 - ā_t/ā_{t-1}))
                # This is always non-zero at intermediate timesteps and is the physically
                # meaningful spread used for the truncated-Gaussian embedding intervals.
                sigma_t_ddpm = torch.sqrt(
                    (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
                    * (1 - alpha_prod_t / alpha_prod_t_prev)
                )

                # Actual step sigma = eta * sigma_ddpm (eta controls stochasticity)
                sigma_t = self.config.eta * sigma_t_ddpm

                # Noise-direction coefficient: sqrt(1 - ā_{t-1} - σ_t²)
                noise_direction_coeff = torch.sqrt(
                    torch.clamp(1 - alpha_prod_t_prev - sigma_t ** 2, min=0.0)
                )

                # Full DDIM/DDPM mean: sqrt(ā_{t-1})*x0_pred + noise_direction*ε_pred
                mu_T_minus_1 = (
                    alpha_prod_t_prev ** 0.5 * pred_original_sample
                    + noise_direction_coeff * noise_pred
                )

                # Use DDPM posterior sigma for the embedding distribution
                sigma_T_minus_1 = sigma_t_ddpm * torch.ones_like(mu_T_minus_1)

            # Scheduler step — eta and generator make each step reproducible
            latents = self.scheduler.step(
                noise_pred, t, latents,
                eta=self.config.eta,
                generator=generator,
            ).prev_sample

        return latents, mu_T_minus_1, sigma_T_minus_1

    def encode(
        self,
        prompt: str,
        secret_bits: List[int],
        seed: int,
        guidance_scale: Optional[float] = None,
        show_progress: bool = True,
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Encode secret bits into a stego image (Algorithm 1).

        The configured transfer format (self.config.transfer_format) controls
        both the D calculation and the output format.

        Args:
            prompt: Text prompt for image generation
            secret_bits: List of binary values (0 or 1) to hide
            seed: Random seed (must be shared with receiver)
            guidance_scale: Classifier-free guidance scale
            show_progress: Whether to show diffusion progress bar

        Returns:
            PIL Image for pil/png/jpeg formats, torch.Tensor for 'none' format
        """
        if guidance_scale is None:
            guidance_scale = self.config.guidance_scale
        self._seed_everything(seed)
        secret_bits = np.array(secret_bits, dtype=np.int32)
        rng = np.random.default_rng(seed)

        # Step 1-2: Encode prompt
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(prompt)

        # Step 3: Run diffusion to get Z_T and intermediate values
        Z_T, mu_T_minus_1, sigma_T_minus_1 = self._run_diffusion_process(
            prompt_embeds, seed, guidance_scale, show_progress,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )

        # Step 4: Pre-generate image X = D(Z_T)
        with torch.no_grad():
            X = self.vae.decode(Z_T.float() / self.vae.config.scaling_factor).sample

        # Step 5: Apply transfer format, then reconstruct latent Z'_T = E(transfer(X))
        with torch.no_grad():
            X_transferred = self._apply_transfer_format(X)
            Z_T_prime = self.vae.encode(X_transferred).latent_dist.mode()
            Z_T_prime = Z_T_prime * self.vae.config.scaling_factor

        # Step 6: Compute discrepancy D = |Z_T - Z'_T|
        D = torch.abs(Z_T.float() - Z_T_prime).cpu().numpy().flatten()
        # Store for visualization
        self._last_discrepancy = D.reshape(Z_T.shape[1:])

        # Step 7: Classify positions into MS intervals
        positions_by_interval = self._classify_positions_by_discrepancy(D)

        # Step 8: Hide bits in positions (low discrepancy first)
        Z_T_stego = Z_T.cpu().numpy().flatten().copy()
        mu_flat = mu_T_minus_1.cpu().numpy().flatten()
        sigma_flat = sigma_T_minus_1.cpu().numpy().flatten()

        # Select positions: pseudo-random spread within each MS interval,
        # MS-priority ordering preserved, deterministic from seed + 12345.
        selected_positions = self._select_positions(
            positions_by_interval, len(secret_bits), seed
        )

        if len(selected_positions) < len(secret_bits):
            print(f"Warning: Could only hide {len(selected_positions)} of {len(secret_bits)} bits")

        hidden_positions = []
        for bit_idx, pos in enumerate(selected_positions):
            # Apply mapping function H
            Z_T_stego[pos] = self._mapping_function_H(
                np.array([secret_bits[bit_idx]]),
                np.array([mu_flat[pos]]),
                np.array([sigma_flat[pos]]),
                rng,
            )[0]
            hidden_positions.append(pos)

        self._last_hidden_positions = hidden_positions

        # Step 9: Generate stego image X^s = D(Z^s_T)
        Z_T_stego_tensor = torch.from_numpy(
            Z_T_stego.reshape(Z_T.shape)
        ).to(self.device, dtype=torch.float32)

        with torch.no_grad():
            stego_raw = self.vae.decode(
                Z_T_stego_tensor / self.vae.config.scaling_factor
            ).sample

        # Apply transfer format to stego output
        stego_output = self._format_stego_output(stego_raw)

        # Store in native format (tensor for "none", PIL for others)
        self._last_original_image = self._format_stego_output(X)
        self._last_stego_image = stego_output

        return stego_output

    def decode(
        self,
        stego_image: Union[Image.Image, torch.Tensor, str],
        prompt: str,
        message_length: int,
        seed: int,
        guidance_scale: Optional[float] = None,
        show_progress: bool = True,
    ) -> List[int]:
        """
        Extract secret bits from a stego image.

        Args:
            stego_image: PIL Image, torch.Tensor (for 'none' format), or path
            prompt: Text prompt (must match encoding)
            message_length: Number of bits to extract
            seed: Random seed (must match encoding)
            guidance_scale: Classifier-free guidance scale (must match encoding)
            show_progress: Whether to show diffusion progress bar

        Returns:
            List of extracted binary values
        """
        if guidance_scale is None:
            guidance_scale = self.config.guidance_scale
        self._seed_everything(seed)

        # Convert input to tensor
        if isinstance(stego_image, str):
            stego_image = Image.open(stego_image).convert("RGB")

        if isinstance(stego_image, torch.Tensor):
            stego_tensor = stego_image.to(self.device, dtype=torch.float32)
        elif isinstance(stego_image, Image.Image):
            stego_tensor = self._pil_to_tensor(stego_image)
        else:
            raise TypeError(
                f"stego_image must be PIL.Image, torch.Tensor, or path string, "
                f"got {type(stego_image)}"
            )

        # Step 1: Encode received image Z'^s_T = E(X^r)
        with torch.no_grad():
            Z_T_stego_prime = self.vae.encode(stego_tensor).latent_dist.mode()
            Z_T_stego_prime = Z_T_stego_prime * self.vae.config.scaling_factor

        # Step 2-3: Regenerate Z_{T-1} using shared seed and prompt
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(prompt)
        Z_T, mu_T_minus_1, _ = self._run_diffusion_process(
            prompt_embeds, seed, guidance_scale, show_progress=False,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )

        # Apply same transfer format for D calculation as was used during encode
        with torch.no_grad():
            X = self.vae.decode(Z_T.float() / self.vae.config.scaling_factor).sample
            X_transferred = self._apply_transfer_format(X)
            Z_T_prime = self.vae.encode(X_transferred).latent_dist.mode()
            Z_T_prime = Z_T_prime * self.vae.config.scaling_factor

        D = torch.abs(Z_T - Z_T_prime).cpu().numpy().flatten()
        positions_by_interval = self._classify_positions_by_discrepancy(D)

        # Get flat arrays
        Z_stego_flat = Z_T_stego_prime.cpu().numpy().flatten()
        mu_flat = mu_T_minus_1.cpu().numpy().flatten()

        # Step 4: Reproduce identical position list and extract bits
        msg_len = message_length if message_length > 0 else self.config.message_length
        selected_positions = self._select_positions(positions_by_interval, msg_len, seed)

        extracted_bits = []
        for pos in selected_positions:
            # Decision rule: Z'^s_T[i] <= mu[i] -> 0, else -> 1
            extracted_bits.append(0 if Z_stego_flat[pos] <= mu_flat[pos] else 1)

        return extracted_bits

    @staticmethod
    def text_to_bits(text: str) -> List[int]:
        """Convert text string to list of bits."""
        bits = []
        for char in text.encode('utf-8'):
            for i in range(7, -1, -1):
                bits.append((char >> i) & 1)
        return bits

    @staticmethod
    def bits_to_text(bits: List[int]) -> str:
        """Convert list of bits to text string."""
        # Pad to multiple of 8
        while len(bits) % 8 != 0:
            bits.append(0)

        bytes_list = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            bytes_list.append(byte)

        return bytes(bytes_list).decode('utf-8', errors='replace')

    def visualize_discrepancy(self, channel: int = 0, figsize: Tuple[int, int] = (10, 8)):
        """
        Visualize the latent discrepancy D as a heatmap.

        Args:
            channel: Which latent channel to visualize (0-3)
            figsize: Figure size
        """
        if self._last_discrepancy is None:
            print("No encoding has been performed yet. Run encode() first.")
            return

        fig, ax = plt.subplots(figsize=figsize)

        # Get single channel
        D_channel = self._last_discrepancy[channel]

        sns.heatmap(
            D_channel,
            ax=ax,
            cmap='viridis',
            cbar_kws={'label': 'Discrepancy |Z_T - Z\'_T|'}
        )

        ax.set_title(f'Latent Discrepancy (Channel {channel})')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')

        plt.tight_layout()
        plt.show()

        return fig

    def visualize_ms_distribution(self, figsize: Tuple[int, int] = (12, 5)):
        """
        Visualize the distribution of positions across MS intervals.
        """
        if self._last_discrepancy is None:
            print("No encoding has been performed yet. Run encode() first.")
            return

        D_flat = self._last_discrepancy.flatten()
        positions_by_interval = self._classify_positions_by_discrepancy(D_flat)

        counts = [len(pos) for pos in positions_by_interval]
        labels = [f'MS_{i}\n{self.config.ms_intervals[i]}' for i in range(len(self.config.ms_intervals))]

        fig, ax = plt.subplots(figsize=figsize)

        colors = sns.color_palette("Blues_d", len(counts))
        bars = ax.bar(range(len(counts)), counts, color=colors)

        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(labels)
        ax.set_xlabel('MS Interval')
        ax.set_ylabel('Number of Positions')
        ax.set_title('Distribution of Latent Positions by Discrepancy Interval')

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 50,
                str(count),
                ha='center',
                va='bottom'
            )

        plt.tight_layout()
        plt.show()

        return fig

    @staticmethod
    def _to_numpy_display(img: Union[Image.Image, torch.Tensor]) -> np.ndarray:
        """Convert PIL Image or tensor to (H,W,3) uint8 numpy for display."""
        if isinstance(img, torch.Tensor):
            arr = (img / 2 + 0.5).clamp(0, 1)
            arr = arr.cpu().permute(0, 2, 3, 1).numpy()[0]
            return (arr * 255).round().astype(np.uint8)
        return np.array(img)

    def compare_images(self, figsize: Tuple[int, int] = (14, 6)):
        """
        Display original and stego images side-by-side.
        """
        if self._last_original_image is None or self._last_stego_image is None:
            print("No encoding has been performed yet. Run encode() first.")
            return

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        orig_np = self._to_numpy_display(self._last_original_image)
        stego_np = self._to_numpy_display(self._last_stego_image)

        # Original image
        axes[0].imshow(orig_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Stego image
        axes[1].imshow(stego_np)
        axes[1].set_title('Stego Image')
        axes[1].axis('off')

        # Difference (amplified)
        diff = np.abs(orig_np.astype(np.float32) - stego_np.astype(np.float32))
        diff_amplified = np.clip(diff * 10, 0, 255).astype(np.uint8)

        axes[2].imshow(diff_amplified)
        axes[2].set_title('Difference (10x amplified)')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        return fig

    def get_capacity(self) -> int:
        """
        Get the maximum steganographic capacity in bits.

        For a 64x64x4 latent space, the theoretical maximum is 16384 bits.
        In practice, positions with high discrepancy are avoided.
        """
        return self.config.capacity_bits


if __name__ == "__main__":
    # Quick test
    print("LDStega module loaded successfully.")
    print("Use LDStega class to encode/decode messages in images.")
