"""
ComfyUI Node Definitions for the DFM-Solver
=============================================

Provides two nodes:

1. **DFM Sampler** — General-purpose flow-matching sampler with RK4 integration
   and FFT detail injection.
2. **DFM Inpaint Sampler** — Extended variant with mask input and covariance
   matching for seamless inpainting blending.

Both nodes accept standard ComfyUI types (MODEL, CONDITIONING, LATENT) and
output a LATENT.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .dfm_sampler import DFMSolver, build_model_fn


# =========================================================================
#  Shared Constants
# =========================================================================

CATEGORY = "sampling/dfm"


# =========================================================================
#  Helper — prepare conditioning for the model wrapper
# =========================================================================

def _prepare_cond(conditioning):
    """Extract the tensor payload from ComfyUI conditioning format.

    ComfyUI conditioning is a list of ``[tensor, dict]`` pairs. For simple
    use-cases we only need the first entry's tensor.
    """
    if isinstance(conditioning, list) and len(conditioning) > 0:
        return conditioning[0][0]
    return conditioning


# =========================================================================
#  Node 1: DFM Sampler (general-purpose)
# =========================================================================

class DFMSamplerNode:
    """ComfyUI node wrapping the Dilated Flow-Matching Solver for standard
    text-to-image and image-to-image generation.

    This replaces the KSampler in workflows using flow-matching / velocity-field
    models (MMDiT, Flux, SD3, etc.).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Random seed for the initial noise.",
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Number of solver steps. More steps = higher fidelity.",
                }),
                "cfg": ("FLOAT", {
                    "default": 7.5,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.5,
                    "tooltip": "Classifier-Free Guidance scale.",
                }),
                "dilation_strength": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Timestep dilation strength. 0 = uniform steps, "
                        "1 = full cosine dilation (concentrates steps in the "
                        "high-curvature mid-region of the ODE trajectory)."
                    ),
                }),
                "fft_injection_strength": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": (
                        "High-frequency detail injection via spectral amplification. "
                        "0 = disabled, 0.1–0.25 = subtle sharpening, >0.4 = aggressive."
                    ),
                }),
                "fft_highpass_ratio": ("FLOAT", {
                    "default": 0.35,
                    "min": 0.05,
                    "max": 0.95,
                    "step": 0.05,
                    "tooltip": (
                        "Cutoff for the high-pass filter as a fraction of the max "
                        "frequency radius. Lower = more frequencies affected."
                    ),
                }),
                "use_rk4": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Use 4th-order Runge-Kutta integrator (4 model evals/step). "
                        "Disable to fall back to Euler (1 eval/step, lower quality)."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = CATEGORY

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed: int,
        steps: int,
        cfg: float,
        dilation_strength: float,
        fft_injection_strength: float,
        fft_highpass_ratio: float,
        use_rk4: bool,
    ):
        """Execute the DFM sampling loop and return the denoised latent."""
        # --- Unpack the ComfyUI latent dict ---
        latent = latent_image["samples"].clone()
        device = model.load_device
        dtype = model.model.model_dtype() if hasattr(model.model, "model_dtype") else torch.float32

        # --- Seed & initial noise ---
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(
            latent.shape, generator=generator, device="cpu", dtype=dtype,
        )
        x = noise.to(device)

        # --- Build model function with CFG ---
        pos_cond = _prepare_cond(positive)
        neg_cond = _prepare_cond(negative)
        model_fn = build_model_fn(model, pos_cond, neg_cond, cfg_scale=cfg)

        # --- Initialise solver ---
        solver = DFMSolver(
            steps=steps,
            dilation_strength=dilation_strength,
            fft_injection_strength=fft_injection_strength,
            fft_highpass_ratio=fft_highpass_ratio,
            covariance_weight=0.0,  # no inpainting for this node
        )

        # --- Run ---
        result = solver.sample(
            model_fn=model_fn,
            x=x,
            sigma_min=0.0,
            sigma_max=1.0,
            use_rk4=use_rk4,
        )

        return ({"samples": result.cpu()},)


# =========================================================================
#  Node 2: DFM Inpaint Sampler
# =========================================================================

class DFMInpaintSamplerNode:
    """ComfyUI node for inpainting with the DFM-Solver.

    Extends the base sampler with mask-aware covariance matching that forces
    newly generated regions to adopt the color and lighting distribution of
    the unmasked original.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "mask": ("MASK",),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Random seed for the initial noise.",
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Number of solver steps.",
                }),
                "cfg": ("FLOAT", {
                    "default": 7.5,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.5,
                    "tooltip": "Classifier-Free Guidance scale.",
                }),
                "dilation_strength": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Timestep dilation strength.",
                }),
                "fft_injection_strength": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "High-frequency spectral injection strength.",
                }),
                "fft_highpass_ratio": ("FLOAT", {
                    "default": 0.35,
                    "min": 0.05,
                    "max": 0.95,
                    "step": 0.05,
                    "tooltip": "High-pass filter cutoff ratio.",
                }),
                "covariance_weight": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "How strongly to match the generated region's color/lighting "
                        "to the original. 0 = disabled, 0.4–0.7 = recommended for "
                        "natural blending, 1.0 = full statistical matching."
                    ),
                }),
                "use_rk4": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use RK4 (4 evals/step) or Euler (1 eval/step).",
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = CATEGORY

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        mask,
        seed: int,
        steps: int,
        cfg: float,
        dilation_strength: float,
        fft_injection_strength: float,
        fft_highpass_ratio: float,
        covariance_weight: float,
        use_rk4: bool,
    ):
        """Execute the DFM inpainting sampling loop."""
        # --- Unpack ---
        latent = latent_image["samples"].clone()
        device = model.load_device
        dtype = model.model.model_dtype() if hasattr(model.model, "model_dtype") else torch.float32

        # --- Prepare mask [B, 1, H, W] ---
        mask_tensor = mask.clone().to(device=device, dtype=dtype)
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        elif mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(1)
        # Resize mask to match latent spatial dims if needed
        if mask_tensor.shape[-2:] != latent.shape[-2:]:
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor, size=latent.shape[-2:], mode="nearest",
            )

        # --- Seed & initial noise ---
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(
            latent.shape, generator=generator, device="cpu", dtype=dtype,
        )
        # Start from noise in masked region, original content in unmasked
        original_latent = latent.to(device)
        x = noise.to(device)

        # --- Build model function ---
        pos_cond = _prepare_cond(positive)
        neg_cond = _prepare_cond(negative)
        model_fn = build_model_fn(model, pos_cond, neg_cond, cfg_scale=cfg)

        # --- Initialise solver ---
        solver = DFMSolver(
            steps=steps,
            dilation_strength=dilation_strength,
            fft_injection_strength=fft_injection_strength,
            fft_highpass_ratio=fft_highpass_ratio,
            covariance_weight=covariance_weight,
        )

        # --- Run ---
        result = solver.sample(
            model_fn=model_fn,
            x=x,
            sigma_min=0.0,
            sigma_max=1.0,
            mask=mask_tensor,
            original_latent=original_latent,
            use_rk4=use_rk4,
        )

        return ({"samples": result.cpu()},)
