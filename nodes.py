"""
ComfyUI Node Definitions for the DFM-Solver v2
================================================

Uses ComfyUI's CFGGuider + custom Sampler architecture so all conditioning
preprocessing is handled natively by ComfyUI.
"""

from __future__ import annotations

import torch

import comfy.samplers
import comfy.sample
import comfy.model_management
import latent_preview

from .dfm_sampler import DFMSampler


CATEGORY = "sampling/dfm"


# =========================================================================
#  DFM Sampler Node (general-purpose)
# =========================================================================

class DFMSamplerNode:
    """Drop-in KSampler replacement using the DFM-Solver v2 pipeline.

    Combines RF-native interpolation, 2nd-order multi-step prediction,
    dynamic thresholding, and optional restart sampling for superior
    quality on flow-matching models.
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
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                }),
                "steps": ("INT", {
                    "default": 20, "min": 1, "max": 200, "step": 1,
                }),
                "cfg": ("FLOAT", {
                    "default": 2.5, "min": 0.0, "max": 30.0, "step": 0.5,
                }),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                "eta": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Ancestral noise strength. 0 = deterministic ODE, 1 = full stochastic.",
                }),
                "s_noise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Noise multiplier for ancestral steps.",
                }),
                "restart_segments": ("INT", {
                    "default": 2, "min": 1, "max": 5, "step": 1,
                    "tooltip": "Number of sampling passes. 1 = normal, 2+ = restart for quality.",
                }),
                "dynamic_threshold": ("FLOAT", {
                    "default": 0.995, "min": 0.5, "max": 1.0, "step": 0.005,
                    "tooltip": "Percentile for dynamic thresholding. 1.0 = disabled.",
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
        eta: float,
        s_noise: float,
        restart_segments: int,
        dynamic_threshold: float,
    ):
        latent = latent_image["samples"]
        device = model.load_device

        # --- Fix empty latent channels FIRST (must precede prepare_noise) ---
        latent = comfy.sample.fix_empty_latent_channels(model, latent,
                    latent_image.get("downscale_ratio_spacial", None))

        # --- Noise (generated from fixed latent so shapes match) ---
        noise = comfy.sample.prepare_noise(latent, seed)

        # --- Noise mask ---
        noise_mask = latent_image.get("noise_mask", None)

        # --- Build DFM v2 sampler ---
        sampler = DFMSampler(
            eta=eta,
            s_noise=s_noise,
            restart_segments=restart_segments,
            dynamic_threshold=dynamic_threshold,
        )

        # --- Sigma schedule ---
        model_sampling = model.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(
            model_sampling, scheduler, steps
        ).to(device)

        # --- CFGGuider (handles all conditioning) ---
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)

        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1)

        # --- Run sampling ---
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
#  DFM Inpaint Sampler Node
# =========================================================================

class DFMInpaintSamplerNode:
    """DFM-Solver v2 with explicit mask input for inpainting."""

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
                    "default": 2.5, "min": 0.0, "max": 30.0, "step": 0.5,
                }),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                "eta": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Ancestral noise strength. 0 = deterministic, 1 = full stochastic.",
                }),
                "s_noise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "restart_segments": ("INT", {
                    "default": 2, "min": 1, "max": 5, "step": 1,
                    "tooltip": "1 = normal, 2+ = restart for quality.",
                }),
                "dynamic_threshold": ("FLOAT", {
                    "default": 0.995, "min": 0.5, "max": 1.0, "step": 0.005,
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
        eta: float,
        s_noise: float,
        restart_segments: int,
        dynamic_threshold: float,
    ):
        latent = latent_image["samples"]
        device = model.load_device

        # --- Fix empty latent channels FIRST ---
        latent = comfy.sample.fix_empty_latent_channels(model, latent,
                    latent_image.get("downscale_ratio_spacial", None))

        noise = comfy.sample.prepare_noise(latent, seed)

        # Prefer mask from InpaintModelConditioning, else use explicit mask
        noise_mask = latent_image.get("noise_mask", None)
        if noise_mask is None:
            noise_mask = mask

        sampler = DFMSampler(
            eta=eta,
            s_noise=s_noise,
            restart_segments=restart_segments,
            dynamic_threshold=dynamic_threshold,
        )

        model_sampling = model.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(
            model_sampling, scheduler, steps
        ).to(device)

        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
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
