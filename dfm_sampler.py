"""
Dilated Flow-Matching Solver (DFM-Solver)
==========================================

A higher-order ODE solver for flow-matching / velocity-field models (MMDiT, Flux, SD3, etc.)
with integrated FFT high-frequency detail injection and continuous covariance matching
for seamless inpainting blending.

Uses ComfyUI's CFGGuider + Sampler interface so all conditioning preprocessing
(convert_cond, process_conds, encode_model_conds, area/mask resolution, hooks,
ControlNet, etc.) is handled natively by ComfyUI.

Author: DFM-Solver Contributors
License: MIT
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from torch import Tensor

import comfy.samplers  # type: ignore[import-untyped]


class DFMSampler(comfy.samplers.Sampler):
    """ComfyUI Sampler subclass implementing the DFM-Solver ODE integration.

    Plugs into CFGGuider.outer_sample() which handles all conditioning
    preprocessing automatically.
    """

    def __init__(
        self,
        dilation_strength: float = 0.7,
        fft_injection_strength: float = 0.15,
        fft_highpass_ratio: float = 0.35,
        covariance_weight: float = 0.0,
        use_rk4: bool = True,
    ) -> None:
        self.dilation_strength = max(0.0, min(dilation_strength, 1.0))
        self.fft_injection_strength = max(0.0, fft_injection_strength)
        self.fft_highpass_ratio = max(0.0, min(fft_highpass_ratio, 1.0))
        self.covariance_weight = max(0.0, min(covariance_weight, 1.0))
        self.use_rk4 = use_rk4

    # ------------------------------------------------------------------ #
    #  Sampler interface (called by CFGGuider.inner_sample)
    # ------------------------------------------------------------------ #

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
        """Entry point called by CFGGuider after conditioning is prepared.

        Args:
            model_wrap: CFGGuider callable  ``(x, sigma, **extra_args) → denoised``.
            sigmas: 1-D tensor of sigma values (high → 0).
            extra_args: Dict with ``model_options``, ``seed``, etc.
            callback: Progress callback ``(step_info_dict) → None``.
            noise: Initial noise tensor.
            latent_image: Original latent for img2img / inpainting.
            denoise_mask: Mask from ComfyUI's inpainting pipeline.
            disable_pbar: (unused, for API compat).

        Returns:
            Denoised samples tensor.
        """
        # --- KSamplerX0Inpaint wrapping (handles denoise_mask compositing) ---
        model_k = comfy.samplers.KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        model_k.noise = noise
        extra_args["denoise_mask"] = denoise_mask

        # --- Apply noise scaling (same as built-in KSampler) ---
        x = model_wrap.inner_model.model_sampling.noise_scaling(
            sigmas[0], noise, latent_image,
            self.max_denoise(model_wrap, sigmas),
        )

        n_steps = len(sigmas) - 1
        model_options = extra_args.get("model_options", {})
        seed = extra_args.get("seed", None)

        for i in range(n_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            # --- ODE integration step ---
            if self.use_rk4:
                x = self._rk4_step(model_k, x, sigma, sigma_next, model_options, seed, denoise_mask)
            else:
                x = self._euler_step(model_k, x, sigma, sigma_next, model_options, seed, denoise_mask)

            # --- FFT high-frequency detail injection ---
            x = self._fft_detail_injection(x)

            # --- Covariance matching (when mask is present) ---
            if denoise_mask is not None and self.covariance_weight > 0.0:
                x = self._covariance_match(x, denoise_mask, self.covariance_weight)

            # --- Progress callback ---
            if callback is not None:
                callback({
                    "x": x,
                    "i": i,
                    "sigma": sigma,
                    "sigma_hat": sigma,
                    "denoised": x,
                })

        x = model_wrap.inner_model.model_sampling.inverse_noise_scaling(
            sigmas[-1], x,
        )
        return x

    # ------------------------------------------------------------------ #
    #  ODE integration steps
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_d(x: Tensor, sigma: Tensor, denoised: Tensor) -> Tensor:
        """Convert denoised prediction → ODE derivative: dx/dσ = (x − x̂₀)/σ"""
        # Broadcast sigma to x's shape
        s = sigma.reshape(-1, *([1] * (x.ndim - 1)))
        return (x - denoised) / s.clamp(min=1e-7)

    def _rk4_step(
        self, model_k, x: Tensor, sigma: Tensor, sigma_next: Tensor,
        model_options: dict, seed, denoise_mask,
    ) -> Tensor:
        """4th-order Runge-Kutta integration step (4 model evaluations)."""
        h = sigma_next - sigma
        s_in = x.new_ones([x.shape[0]])

        def f(x_in: Tensor, s: Tensor) -> Tensor:
            s_safe = s.clamp(min=1e-7)
            denoised = model_k(x_in, s_safe * s_in, denoise_mask, model_options=model_options, seed=seed)
            return self._to_d(x_in, s_safe * s_in, denoised)

        s_mid = sigma + 0.5 * h
        k1 = f(x, sigma)
        k2 = f(x + 0.5 * h * k1, s_mid)
        k3 = f(x + 0.5 * h * k2, s_mid)
        k4 = f(x + h * k3, sigma_next)

        return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _euler_step(
        self, model_k, x: Tensor, sigma: Tensor, sigma_next: Tensor,
        model_options: dict, seed, denoise_mask,
    ) -> Tensor:
        """1st-order Euler step (1 model evaluation)."""
        s_in = x.new_ones([x.shape[0]])
        denoised = model_k(x, sigma * s_in, denoise_mask, model_options=model_options, seed=seed)
        d = self._to_d(x, sigma * s_in, denoised)
        return x + d * (sigma_next - sigma)

    # ------------------------------------------------------------------ #
    #  FFT High-Frequency Detail Injection
    # ------------------------------------------------------------------ #

    def _fft_detail_injection(self, x: Tensor) -> Tensor:
        """Amplify high-frequency content via spectral high-pass filtering."""
        if self.fft_injection_strength <= 0.0:
            return x

        H, W = x.shape[-2:]
        x_freq = torch.fft.rfft2(x, norm="ortho")

        freq_h = torch.fft.fftfreq(H, device=x.device, dtype=x.dtype)
        freq_w = torch.fft.rfftfreq(W, device=x.device, dtype=x.dtype)
        grid_h, grid_w = torch.meshgrid(freq_h, freq_w, indexing="ij")
        radius = torch.sqrt(grid_h ** 2 + grid_w ** 2)

        max_radius = math.sqrt(0.5)
        radius_norm = radius / max_radius

        sharpness = 10.0
        highpass_mask = torch.sigmoid(sharpness * (radius_norm - self.fft_highpass_ratio))
        for _ in range(x.ndim - 2):
            highpass_mask = highpass_mask.unsqueeze(0)

        amplification = 1.0 + self.fft_injection_strength * highpass_mask
        x_freq_boosted = x_freq * amplification
        return torch.fft.irfft2(x_freq_boosted, s=(H, W), norm="ortho")

    # ------------------------------------------------------------------ #
    #  Continuous Covariance Matching
    # ------------------------------------------------------------------ #

    @staticmethod
    def _covariance_match(x: Tensor, mask: Tensor, weight: float) -> Tensor:
        """Per-channel affine covariance matching between masked/unmasked regions."""
        if weight <= 0.0:
            return x

        mask_f = mask.float()
        if mask_f.ndim < x.ndim:
            # Expand mask dims to match x ([B, H, W] → [B, 1, ..., H, W])
            while mask_f.ndim < x.ndim:
                mask_f = mask_f.unsqueeze(1)
        mask_f = mask_f.expand_as(x)
        inv_mask_f = 1.0 - mask_f
        eps = 1e-6

        spatial_dims = tuple(range(2, x.ndim))

        orig_count = inv_mask_f.sum(dim=spatial_dims, keepdim=True).clamp(min=1.0)
        orig_mean = (x * inv_mask_f).sum(dim=spatial_dims, keepdim=True) / orig_count
        orig_var = (((x - orig_mean) ** 2) * inv_mask_f).sum(dim=spatial_dims, keepdim=True) / orig_count
        orig_std = (orig_var + eps).sqrt()

        gen_count = mask_f.sum(dim=spatial_dims, keepdim=True).clamp(min=1.0)
        gen_mean = (x * mask_f).sum(dim=spatial_dims, keepdim=True) / gen_count
        gen_var = (((x - gen_mean) ** 2) * mask_f).sum(dim=spatial_dims, keepdim=True) / gen_count
        gen_std = (gen_var + eps).sqrt()

        x_normalised = (x - gen_mean) / gen_std
        x_matched = x_normalised * orig_std + orig_mean

        mask_bool = mask_f.bool()
        x_out = x.clone()
        x_out[mask_bool] = x[mask_bool] * (1.0 - weight) + x_matched[mask_bool] * weight
        return x_out


# =========================================================================
#  Sigma schedule with cosine dilation
# =========================================================================

def compute_dilated_sigmas(
    model_sampling,
    steps: int,
    dilation_strength: float,
    device: torch.device,
) -> Tensor:
    """Compute a cosine-dilated sigma schedule using the model's sigma range.

    Concentrates evaluations in the high-curvature mid-region.
    """
    d = max(0.0, min(dilation_strength, 1.0))

    # Get sigmas from model's normal schedule, then warp them
    sigma_max = float(model_sampling.sigma_max)
    sigma_min = float(model_sampling.sigma_min)

    u = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float64)
    linear_part = u
    cosine_part = (1.0 - torch.cos(math.pi * u)) / 2.0
    warped = (1.0 - d) * linear_part + d * cosine_part

    sigmas = sigma_max + (sigma_min - sigma_max) * warped
    # Append a final 0.0 sigma if sigma_min is very close to 0
    sigmas_list = sigmas.tolist()
    if sigmas_list[-1] > 1e-6:
        sigmas_list.append(0.0)

    return torch.FloatTensor(sigmas_list).to(device)
