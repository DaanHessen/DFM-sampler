"""
DFM-Solver v2 — Dilated Flow-Matching Sampler
===============================================

A novel sampler for flow-matching / Rectified Flow models (Qwen-image, Flux,
SD3, etc.) that combines four quality-enhancing techniques:

  1. RF-native sigma-ratio interpolation (matches model training)
  2. 2nd-order multi-step predictor (zero extra model evals)
  3. Dynamic thresholding (percentile clamping, prevents CFG artifacts)
  4. Restart sampling (re-noise + re-denoise for detail refinement)

Uses ComfyUI's CFGGuider + Sampler interface so all conditioning is handled
natively by ComfyUI.

Author: DFM-Solver Contributors
License: MIT
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor
import comfy.samplers
import comfy.model_management
import comfy.model_sampling
from comfy.utils import model_trange as trange

log = logging.getLogger("DFM-Solver")


# =========================================================================
#  Utility functions
# =========================================================================

def _get_ancestral_step_rf(sigma_from: Tensor, sigma_to: Tensor, eta: float = 1.0):
    """Compute sigma_down and renoise coefficient for RF (Rectified Flow) ancestral step.

    Adapted from sample_euler_ancestral_RF in k-diffusion.
    """
    if eta <= 0 or sigma_to == 0:
        return sigma_to, torch.zeros_like(sigma_to)

    downstep_ratio = 1 + (sigma_to / sigma_from - 1) * eta
    sigma_down = sigma_to * downstep_ratio
    alpha_next = 1 - sigma_to
    alpha_down = 1 - sigma_down
    renoise_coeff = (sigma_to ** 2 - sigma_down ** 2 * alpha_next ** 2 / alpha_down ** 2) ** 0.5
    return sigma_down, renoise_coeff


def _default_noise_sampler(x: Tensor, seed: Optional[int] = None):
    """Create a reproducible noise sampler matching k-diffusion's convention."""
    if seed is not None:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed + (1 if x.device == torch.device("cpu") else 0))
    else:
        generator = None

    def sampler(sigma, sigma_next):
        return torch.randn(x.size(), dtype=x.dtype, layout=x.layout,
                           device=x.device, generator=generator)
    return sampler


def _dynamic_threshold(denoised: Tensor, percentile: float = 0.995) -> Tensor:
    """Percentile-based dynamic thresholding (Imagen technique).

    Clamps the denoised prediction to the `percentile`-th magnitude,
    then rescales so values fill the valid range.  Prevents CFG-induced
    oversaturation and color drift without destroying detail.
    """
    if percentile >= 1.0:
        return denoised

    # Compute per-sample threshold
    flat = denoised.flatten(1).abs()
    s = torch.quantile(flat, percentile, dim=1)
    s = torch.clamp(s, min=1.0)  # don't shrink values that are already fine
    s = s.reshape(-1, *([1] * (denoised.ndim - 1)))

    return denoised.clamp(-s, s) / s


# =========================================================================
#  DFM Sampler v2
# =========================================================================

class DFMSampler(comfy.samplers.Sampler):
    """ComfyUI Sampler implementing the DFM-Solver v2 pipeline.

    Combines RF-native interpolation, 2nd-order multi-step prediction,
    dynamic thresholding, and restart sampling.
    """

    def __init__(
        self,
        eta: float = 0.5,
        s_noise: float = 1.0,
        restart_segments: int = 1,
        dynamic_threshold: float = 0.995,
    ) -> None:
        self.eta = max(0.0, eta)
        self.s_noise = max(0.0, s_noise)
        self.restart_segments = max(1, int(restart_segments))
        self.dynamic_threshold_pct = min(1.0, max(0.0, dynamic_threshold))

    # ------------------------------------------------------------------ #
    #  Sampler interface (called by CFGGuider.inner_sample)
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ) -> Tensor:
        # --- Inpaint wrapping ---
        model_k = comfy.samplers.KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        model_k.noise = noise
        extra_args = dict(extra_args) if extra_args is not None else {}
        extra_args["denoise_mask"] = denoise_mask

        # --- Initial noise scaling ---
        x = model_wrap.inner_model.model_sampling.noise_scaling(
            sigmas[0], noise, latent_image,
            self.max_denoise(model_wrap, sigmas),
        )

        log.info("DFM-Solver v2: x=%s, sigmas=[%.4f → %.4f], %d steps, "
                 "eta=%.2f, restarts=%d, threshold=%.3f",
                 list(x.shape), sigmas[0].item(), sigmas[-1].item(),
                 len(sigmas) - 1, self.eta, self.restart_segments,
                 self.dynamic_threshold_pct)

        # --- Setup ---
        seed = extra_args.get("seed", None)
        noise_sampler = _default_noise_sampler(x, seed=seed)
        s_in = x.new_ones([x.shape[0]])
        n_steps = len(sigmas) - 1

        # --- Split sigmas into restart segments ---
        segments = self._split_into_segments(sigmas, self.restart_segments)

        global_step = 0
        for seg_idx, seg_sigmas in enumerate(segments):
            # On restart segments (seg_idx > 0), add noise back
            if seg_idx > 0:
                restart_sigma = seg_sigmas[0]
                log.info("DFM restart: re-noising to sigma=%.4f", restart_sigma.item())
                # Re-noise: x_noisy = sigma * noise + (1 - sigma) * x_clean
                fresh_noise = noise_sampler(restart_sigma, restart_sigma)
                x = model_wrap.inner_model.model_sampling.noise_scaling(
                    restart_sigma, fresh_noise, x, max_denoise=False,
                )

            # Run the segment
            x, global_step = self._run_segment(
                model_k, x, seg_sigmas, extra_args,
                callback, noise_sampler, s_in,
                n_steps, global_step, disable_pbar,
            )

        # --- Final inverse noise scaling ---
        x = model_wrap.inner_model.model_sampling.inverse_noise_scaling(
            sigmas[-1], x,
        )
        return x

    # ------------------------------------------------------------------ #
    #  Core sampling loop for one segment
    # ------------------------------------------------------------------ #

    def _run_segment(
        self,
        model_k,
        x: Tensor,
        sigmas: Tensor,
        extra_args: dict,
        callback,
        noise_sampler,
        s_in: Tensor,
        total_steps: int,
        global_step: int,
        disable_pbar: bool,
    ) -> tuple[Tensor, int]:
        """Run RF-native 2nd-order sampling over one sigma segment."""
        old_denoised = None
        old_sigma = None
        seg_steps = len(sigmas) - 1

        for i in trange(seg_steps, disable=disable_pbar):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            # --- Model evaluation ---
            denoised = model_k(x, sigma * s_in, **extra_args)

            # --- Dynamic thresholding on the denoised prediction ---
            denoised = _dynamic_threshold(denoised, self.dynamic_threshold_pct)

            # --- Progress callback ---
            # latent_preview.prepare_callback expects: (step, x0, x, total_steps)
            if callback is not None:
                callback(global_step, denoised, x, total_steps)

            # --- Terminal step: just output denoised ---
            if sigma_next == 0:
                x = denoised
                global_step += 1
                old_denoised = denoised
                old_sigma = sigma
                continue

            # --- Compute ancestral step (noise level + renoise amount) ---
            sigma_down, renoise_coeff = _get_ancestral_step_rf(
                sigma, sigma_next, eta=self.eta,
            )

            # --- RF-native step with 2nd-order correction ---
            sigma_ratio = sigma_down / sigma

            if old_denoised is not None and old_sigma is not None:
                # 2nd-order multi-step: use previous denoised for correction
                # Based on DPM-Solver++(2M) adapted for RF interpolation
                #
                # The key insight: we can use the previous denoised prediction
                # to build a 2nd-order estimate WITHOUT any extra model evals.
                h = sigma_down - sigma
                h_last = sigma - old_sigma
                r = h_last / h if abs(h.item()) > 1e-8 else torch.tensor(1.0)

                # 2nd-order corrected denoised
                denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised

                # RF-native interpolation with corrected prediction
                x = sigma_ratio * x + (1 - sigma_ratio) * denoised_d
            else:
                # 1st step: simple RF interpolation (Euler equivalent)
                x = sigma_ratio * x + (1 - sigma_ratio) * denoised

            # --- Ancestral noise injection ---
            if self.eta > 0 and renoise_coeff > 0:
                alpha_next = 1 - sigma_next
                alpha_down = 1 - sigma_down
                x = (alpha_next / alpha_down) * x + \
                    noise_sampler(sigma, sigma_next) * self.s_noise * renoise_coeff

            # --- Bookkeeping ---
            old_denoised = denoised
            old_sigma = sigma
            global_step += 1

            comfy.model_management.soft_empty_cache()

        return x, global_step

    # ------------------------------------------------------------------ #
    #  Restart segment splitting
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_into_segments(sigmas: Tensor, n_segments: int) -> list[Tensor]:
        """Split a sigma schedule into N restart segments.

        For n_segments=1, returns [sigmas] (no restart).
        For n_segments=2, splits at the midpoint: the first segment runs to
        the midpoint, then restart re-noises to that level and re-runs.
        """
        if n_segments <= 1:
            return [sigmas]

        n_steps = len(sigmas) - 1
        if n_steps < 2:
            return [sigmas]

        segments = []
        # First segment: full schedule (gets the overall structure right)
        segments.append(sigmas)

        # Restart segments: re-run from a midpoint in the schedule
        for seg in range(1, n_segments):
            # Restart from progressively later points to refine details
            # e.g. with 2 segments: restart from ~40% through schedule
            # e.g. with 3 segments: restart from ~40% and ~70%
            fraction = 0.3 + 0.3 * (seg / n_segments)
            restart_idx = max(1, min(n_steps - 1, int(fraction * n_steps)))
            segments.append(sigmas[restart_idx:])

        return segments
