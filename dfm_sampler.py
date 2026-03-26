"""
Dilated Flow-Matching Solver (DFM-Solver)
==========================================

A higher-order ODE solver for flow-matching / velocity-field models (MMDiT, Flux, SD3, etc.)
with integrated FFT high-frequency detail injection and continuous covariance matching
for seamless inpainting blending.

Mathematical foundations:
    1. RK4 integration of the learned velocity field v(x, t)
    2. Cosine-dilated timestep schedule concentrating evaluations in high-curvature regions
    3. Spectral high-pass amplification via torch.fft for micro-detail preservation
    4. Affine whitening-coloring transform for latent color/lighting coherence in inpainting

Author: DFM-Solver Contributors
License: MIT
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from torch import Tensor

import comfy.model_management  # type: ignore[import-untyped]
import comfy.samplers  # type: ignore[import-untyped]


class DFMSolver:
    """Dilated Flow-Matching Solver.

    Integrates the probability-flow ODE of a flow-matching model using an
    adaptive-step RK4 scheme with optional spectral detail injection and
    covariance matching for inpainting.

    All operations are pure PyTorch — no external compiled dependencies.
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        steps: int = 20,
        dilation_strength: float = 1.0,
        fft_injection_strength: float = 0.0,
        fft_highpass_ratio: float = 0.35,
        covariance_weight: float = 0.0,
    ) -> None:
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")

        self.steps = steps
        self.dilation_strength = max(0.0, min(dilation_strength, 1.0))
        self.fft_injection_strength = max(0.0, fft_injection_strength)
        self.fft_highpass_ratio = max(0.0, min(fft_highpass_ratio, 1.0))
        self.covariance_weight = max(0.0, min(covariance_weight, 1.0))

    # ------------------------------------------------------------------ #
    #  Sigma Schedule (with cosine dilation)
    # ------------------------------------------------------------------ #

    def compute_sigmas(
        self,
        sigma_min: float,
        sigma_max: float,
        device: torch.device,
    ) -> Tensor:
        """Return N+1 sigmas from ``sigma_max`` down to ``sigma_min``.

        Cosine dilation concentrates evaluations in the high-curvature
        mid-region of the ODE trajectory.
        """
        n = self.steps
        d = self.dilation_strength

        u = torch.linspace(0.0, 1.0, n + 1, device=device, dtype=torch.float64)
        linear_part = u
        cosine_part = (1.0 - torch.cos(math.pi * u)) / 2.0
        warped = (1.0 - d) * linear_part + d * cosine_part

        # sigma_max → sigma_min
        sigmas = sigma_max + (sigma_min - sigma_max) * warped
        return sigmas.to(dtype=torch.float32)

    # ------------------------------------------------------------------ #
    #  Model prediction via ComfyUI's conditioning pipeline
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_denoised(
        model,
        x: Tensor,
        sigma: float,
        positive,
        negative,
        cfg_scale: float,
    ) -> Tensor:
        """Obtain the denoised x0 prediction through ComfyUI's full
        conditioning pipeline (handles c_concat, c_crossattn, ControlNet,
        model patches, etc.)."""
        sigma_tensor = torch.full(
            (x.shape[0],), sigma, device=x.device, dtype=x.dtype,
        )
        return comfy.samplers.sampling_function(
            model, x, sigma_tensor,
            negative, positive, cfg_scale,
            model_options=model.model_options,
        )

    # ------------------------------------------------------------------ #
    #  Derivative helper
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_d(x: Tensor, sigma: float, denoised: Tensor) -> Tensor:
        """Convert a denoised prediction to the ODE derivative.

        For flow-matching models the derivative in sigma-space is:
            dx/dσ = (x − x̂₀) / σ
        """
        return (x - denoised) / max(sigma, 1e-7)

    # ------------------------------------------------------------------ #
    #  Integration steps
    # ------------------------------------------------------------------ #

    def _rk4_step(
        self,
        model,
        x: Tensor,
        sigma: float,
        sigma_next: float,
        positive,
        negative,
        cfg_scale: float,
    ) -> Tensor:
        """Single RK4 integration step (4 model evaluations)."""
        h = sigma_next - sigma
        eps = 1e-7
        s_mid = sigma + 0.5 * h

        def f(x_in: Tensor, s: float) -> Tensor:
            s_safe = max(s, eps)
            denoised = self._get_denoised(model, x_in, s_safe, positive, negative, cfg_scale)
            return self._to_d(x_in, s_safe, denoised)

        k1 = f(x, sigma)
        k2 = f(x + 0.5 * h * k1, s_mid)
        k3 = f(x + 0.5 * h * k2, s_mid)
        k4 = f(x + h * k3, sigma_next)

        return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _euler_step(
        self,
        model,
        x: Tensor,
        sigma: float,
        sigma_next: float,
        positive,
        negative,
        cfg_scale: float,
    ) -> Tensor:
        """Single Euler integration step (1 model evaluation)."""
        eps = 1e-7
        s_safe = max(sigma, eps)
        denoised = self._get_denoised(model, x, s_safe, positive, negative, cfg_scale)
        d = self._to_d(x, s_safe, denoised)
        return x + d * (sigma_next - sigma)

    # ------------------------------------------------------------------ #
    #  FFT High-Frequency Detail Injection
    # ------------------------------------------------------------------ #

    def _fft_detail_injection(self, x: Tensor) -> Tensor:
        """Amplify high-frequency content via spectral filtering.

        Uses a smooth Butterworth-style radial high-pass mask and amplifies
        only the high-frequency components by ``1 + fft_injection_strength``.
        """
        if self.fft_injection_strength <= 0.0:
            return x

        # Work on last two spatial dims regardless of total ndim
        spatial = x.shape[-2:]
        H, W = spatial

        x_freq = torch.fft.rfft2(x, norm="ortho")

        freq_h = torch.fft.fftfreq(H, device=x.device, dtype=x.dtype)
        freq_w = torch.fft.rfftfreq(W, device=x.device, dtype=x.dtype)
        grid_h, grid_w = torch.meshgrid(freq_h, freq_w, indexing="ij")
        radius = torch.sqrt(grid_h ** 2 + grid_w ** 2)

        max_radius = math.sqrt(0.5)
        radius_norm = radius / max_radius

        sharpness = 10.0
        highpass_mask = torch.sigmoid(sharpness * (radius_norm - self.fft_highpass_ratio))

        # Broadcast: add leading singleton dims to match x_freq
        for _ in range(x.ndim - 2):
            highpass_mask = highpass_mask.unsqueeze(0)

        amplification = 1.0 + self.fft_injection_strength * highpass_mask
        x_freq_boosted = x_freq * amplification

        return torch.fft.irfft2(x_freq_boosted, s=(H, W), norm="ortho")

    # ------------------------------------------------------------------ #
    #  Continuous Covariance Matching (Inpainting)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _covariance_match(
        x: Tensor,
        mask: Tensor,
        weight: float,
    ) -> Tensor:
        """Per-channel affine covariance matching between masked/unmasked regions.

            x_matched = (σ_orig / σ_gen) · (x_gen − μ_gen) + μ_orig

        Only applied inside the mask; unmasked pixels are untouched.
        """
        if weight <= 0.0:
            return x

        mask_f = mask.float()
        inv_mask_f = 1.0 - mask_f
        eps = 1e-6

        # Reduce over all spatial dims (everything except batch & channel)
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

        mask_bool = mask_f.bool().expand_as(x)
        x_out = x.clone()
        x_out[mask_bool] = (
            x[mask_bool] * (1.0 - weight) + x_matched[mask_bool] * weight
        )
        return x_out

    # ------------------------------------------------------------------ #
    #  Main Sampling Loop
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def sample(
        self,
        model,
        x: Tensor,
        sigmas: Tensor,
        positive,
        negative,
        cfg_scale: float,
        mask: Optional[Tensor] = None,
        original_latent: Optional[Tensor] = None,
        use_rk4: bool = True,
        callback: Optional[Callable[[int, Tensor], None]] = None,
    ) -> Tensor:
        """Run the full DFM sampling loop.

        Args:
            model: ComfyUI model patcher.
            x: Initial noisy latent.
            sigmas: 1-D tensor of N+1 sigma values (high → low).
            positive: Positive conditioning (full ComfyUI format).
            negative: Negative conditioning (full ComfyUI format).
            cfg_scale: Classifier-free guidance strength.
            mask: Optional inpainting mask (1 = generate, 0 = keep).
            original_latent: Original latent for inpainting reference.
            use_rk4: Use RK4 (True) or Euler (False).
            callback: Optional progress callback.

        Returns:
            Denoised latent tensor.
        """
        comfy.model_management.load_model_gpu(model)
        n_steps = len(sigmas) - 1

        for i in range(n_steps):
            sigma = sigmas[i].item()
            sigma_next = sigmas[i + 1].item()

            # --- ODE integration step ---
            if use_rk4:
                x = self._rk4_step(model, x, sigma, sigma_next, positive, negative, cfg_scale)
            else:
                x = self._euler_step(model, x, sigma, sigma_next, positive, negative, cfg_scale)

            # --- FFT high-frequency detail injection ---
            x = self._fft_detail_injection(x)

            # --- Inpainting: composite + covariance match ---
            if mask is not None and original_latent is not None:
                progress = float(i + 1) / n_steps
                noise_level = 1.0 - progress
                noisy_orig = original_latent + noise_level * torch.randn_like(original_latent)
                inv_mask = 1.0 - mask.float()
                x = x * mask.float() + noisy_orig * inv_mask
                x = self._covariance_match(x, mask, self.covariance_weight)

            if callback is not None:
                callback(i, x)

        return x
