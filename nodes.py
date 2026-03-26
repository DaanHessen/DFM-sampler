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

import comfy.model_management  # type: ignore[import-untyped]

from .dfm_sampler import DFMSolver


CATEGORY = "sampling/dfm"


# =========================================================================
#  Node 1: DFM Sampler (general-purpose)
# =========================================================================

class DFMSamplerNode:
    """ComfyUI node wrapping the Dilated Flow-Matching Solver for standard
    text-to-image and image-to-image generation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Random seed for the initial noise.",
                }),
                "steps": ("INT", {
                    "default": 20, "min": 1, "max": 200, "step": 1,
                    "tooltip": "Number of solver steps. More = higher fidelity.",
                }),
                "cfg": ("FLOAT", {
                    "default": 7.5, "min": 0.0, "max": 30.0, "step": 0.5,
                    "tooltip": "Classifier-Free Guidance scale.",
                }),
                "dilation_strength": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Timestep dilation. 0 = uniform, 1 = full cosine.",
                }),
                "fft_injection_strength": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "High-frequency detail injection. 0 = off.",
                }),
                "fft_highpass_ratio": ("FLOAT", {
                    "default": 0.35, "min": 0.05, "max": 0.95, "step": 0.05,
                    "tooltip": "High-pass filter cutoff ratio.",
                }),
                "use_rk4": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "RK4 (4 evals/step) or Euler (1 eval/step).",
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
        device = model.load_device
        latent = latent_image["samples"].clone()

        # --- Get sigma range from the model's sampling config ---
        model_sampling = model.get_model_object("model_sampling")
        sigma_max = model_sampling.sigma_max.item()
        sigma_min = model_sampling.sigma_min.item()

        # --- Generate noise ---
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(latent.shape, generator=generator, device="cpu")
        x = (noise * sigma_max).to(device)

        # --- Initialise solver ---
        solver = DFMSolver(
            steps=steps,
            dilation_strength=dilation_strength,
            fft_injection_strength=fft_injection_strength,
            fft_highpass_ratio=fft_highpass_ratio,
            covariance_weight=0.0,
        )

        sigmas = solver.compute_sigmas(sigma_min, sigma_max, device)

        # --- Run ---
        result = solver.sample(
            model=model,
            x=x,
            sigmas=sigmas,
            positive=positive,
            negative=negative,
            cfg_scale=cfg,
            use_rk4=use_rk4,
        )

        return ({"samples": result.cpu()},)


# =========================================================================
#  Node 2: DFM Inpaint Sampler
# =========================================================================

class DFMInpaintSamplerNode:
    """ComfyUI node for inpainting with the DFM-Solver.

    Adds mask-aware covariance matching for color/lighting coherence.
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
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Random seed for the initial noise.",
                }),
                "steps": ("INT", {
                    "default": 20, "min": 1, "max": 200, "step": 1,
                }),
                "cfg": ("FLOAT", {
                    "default": 7.5, "min": 0.0, "max": 30.0, "step": 0.5,
                }),
                "dilation_strength": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "fft_injection_strength": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "fft_highpass_ratio": ("FLOAT", {
                    "default": 0.35, "min": 0.05, "max": 0.95, "step": 0.05,
                }),
                "covariance_weight": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Color/lighting matching strength. 0 = off.",
                }),
                "use_rk4": ("BOOLEAN", {
                    "default": True,
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
        device = model.load_device
        latent = latent_image["samples"].clone()

        # --- Get sigma range ---
        model_sampling = model.get_model_object("model_sampling")
        sigma_max = model_sampling.sigma_max.item()
        sigma_min = model_sampling.sigma_min.item()

        # --- Prepare mask to match latent spatial dims ---
        mask_tensor = mask.clone().to(device=device, dtype=torch.float32)
        # Normalise to [B, 1, H, W] regardless of input shape
        if mask_tensor.ndim == 2:       # [H, W]
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        elif mask_tensor.ndim == 3:     # [B, H, W]
            mask_tensor = mask_tensor.unsqueeze(1)
        # mask_tensor is now [B, 1, H, W] — resize spatially
        latent_spatial = latent.shape[-2:]
        if mask_tensor.shape[-2:] != latent_spatial:
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor, size=latent_spatial, mode="nearest",
            )
        # For 5-D latents (e.g. Wan / video models: [B, C, T, H, W]),
        # insert a temporal dim so mask becomes [B, 1, 1, H, W]
        if latent.ndim == 5 and mask_tensor.ndim == 4:
            mask_tensor = mask_tensor.unsqueeze(2)

        # --- Seed & initial noise ---
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(latent.shape, generator=generator, device="cpu")
        original_latent = latent.to(device)
        x = (noise * sigma_max).to(device)

        # --- Solver ---
        solver = DFMSolver(
            steps=steps,
            dilation_strength=dilation_strength,
            fft_injection_strength=fft_injection_strength,
            fft_highpass_ratio=fft_highpass_ratio,
            covariance_weight=covariance_weight,
        )

        sigmas = solver.compute_sigmas(sigma_min, sigma_max, device)

        result = solver.sample(
            model=model,
            x=x,
            sigmas=sigmas,
            positive=positive,
            negative=negative,
            cfg_scale=cfg,
            mask=mask_tensor,
            original_latent=original_latent,
            use_rk4=use_rk4,
        )

        return ({"samples": result.cpu()},)
