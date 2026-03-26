"""
ComfyUI Node Definitions for the DFM-Solver
=============================================

Uses ComfyUI's CFGGuider + custom Sampler architecture so all conditioning
preprocessing (convert_cond, process_conds, area/mask resolution, ControlNet,
hooks, etc.) is handled natively by ComfyUI.  We only supply the ODE
integration logic.
"""

from __future__ import annotations

import torch

import comfy.samplers  # type: ignore[import-untyped]
import comfy.sample  # type: ignore[import-untyped]
import latent_preview

from .dfm_sampler import DFMSampler, apply_dilation


CATEGORY = "sampling/dfm"


# =========================================================================
#  Node 1: DFM Sampler (general-purpose)
# =========================================================================

class DFMSamplerNode:
    """Drop-in KSampler replacement for flow-matching models."""

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
                }),
                "steps": ("INT", {
                    "default": 20, "min": 1, "max": 200, "step": 1,
                }),
                "cfg": ("FLOAT", {
                    "default": 7.5, "min": 0.0, "max": 30.0, "step": 0.5,
                }),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
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
        scheduler: str,
        dilation_strength: float,
        fft_injection_strength: float,
        fft_highpass_ratio: float,
        use_rk4: bool,
    ):
        latent = latent_image["samples"].clone()
        device = model.load_device

        # --- Noise ---
        noise = comfy.sample.prepare_noise(latent, seed)

        # --- Noise mask (from InpaintModelConditioning etc.) ---
        noise_mask = latent_image.get("noise_mask", None)

        # --- Build our custom sampler ---
        sampler = DFMSampler(
            dilation_strength=dilation_strength,
            fft_injection_strength=fft_injection_strength,
            fft_highpass_ratio=fft_highpass_ratio,
            covariance_weight=0.0,
            use_rk4=use_rk4,
        )

        # --- Compute dilated sigma schedule ---
        model_sampling = model.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps).to(device)
        sigmas = apply_dilation(sigmas, dilation_strength)

        # --- Use CFGGuider (handles all conditioning preprocessing) ---
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)

        # Fix empty latent channels if needed
        latent = comfy.sample.fix_empty_latent_channels(model, latent)

        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1)

        # --- Run sampling through the full ComfyUI pipeline ---
        result = guider.sample(
            noise, latent, sampler, sigmas,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=False,
            seed=seed,
        )

        result = result.to(comfy.model_management.intermediate_device())
        return ({"samples": result},)


# =========================================================================
#  Node 2: DFM Inpaint Sampler
# =========================================================================

class DFMInpaintSamplerNode:
    """DFM Sampler with covariance matching for inpainting blending."""

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
                }),
                "steps": ("INT", {
                    "default": 20, "min": 1, "max": 200, "step": 1,
                }),
                "cfg": ("FLOAT", {
                    "default": 7.5, "min": 0.0, "max": 30.0, "step": 0.5,
                }),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
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
        scheduler: str,
        dilation_strength: float,
        fft_injection_strength: float,
        fft_highpass_ratio: float,
        covariance_weight: float,
        use_rk4: bool,
    ):
        latent = latent_image["samples"].clone()
        device = model.load_device

        # --- Noise ---
        noise = comfy.sample.prepare_noise(latent, seed)

        # --- Noise mask: prefer the one from InpaintModelConditioning,
        #     fall back to the explicit mask input ---
        noise_mask = latent_image.get("noise_mask", None)
        if noise_mask is None:
            noise_mask = mask

        # --- Build sampler ---
        sampler = DFMSampler(
            dilation_strength=dilation_strength,
            fft_injection_strength=fft_injection_strength,
            fft_highpass_ratio=fft_highpass_ratio,
            covariance_weight=covariance_weight,
            use_rk4=use_rk4,
        )

        # --- Sigmas ---
        model_sampling = model.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps).to(device)
        sigmas = apply_dilation(sigmas, dilation_strength)

        # --- CFGGuider ---
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)

        latent = comfy.sample.fix_empty_latent_channels(model, latent)

        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1)

        result = guider.sample(
            noise, latent, sampler, sigmas,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=False,
            seed=seed,
        )

        result = result.to(comfy.model_management.intermediate_device())
        return ({"samples": result},)
