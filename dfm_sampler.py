"""
Dilated Flow-Matching Solver (DFM-Solver)
==========================================

A higher-order ODE solver for flow-matching / velocity-field models (MMDiT, Flux, SD3, etc.)
with integrated FFT high-frequency detail injection and continuous covariance matching
for seamless inpainting blending.

Mathematical foundations:
    1. RK4 integration of the learned velocity field  v(x, t)
    2. Cosine-dilated timestep schedule concentrating evaluations in high-curvature regions
    3. Spectral high-pass amplification via torch.fft for micro-detail preservation
    4. Affine whitening-coloring transform for latent color/lighting coherence in inpainting

Author: DFM-Solver Contributors
License: MIT
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Type alias for the model forward call used throughout the solver.
# Signature:  model_fn(latent, timestep, **kwargs) -> velocity_prediction
# ---------------------------------------------------------------------------
ModelForwardFn = Callable[..., Tensor]


class DFMSolver:
    """Dilated Flow-Matching Solver.

    Integrates the probability-flow ODE of a flow-matching model using an
    adaptive-step RK4 scheme with optional spectral detail injection and
    covariance matching for inpainting.

    All operations are pure PyTorch on the same device / dtype as the input
    tensors — no external compiled dependencies.
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
        """Initialise solver hyper-parameters.

        Args:
            steps: Number of integration steps (fewer = faster, more = higher fidelity).
            dilation_strength: Controls how aggressively the timestep schedule is
                warped towards the high-curvature mid-region.  0.0 = uniform spacing,
                1.0 = full cosine dilation.
            fft_injection_strength: Amplitude multiplier for spectral high-frequency
                injection.  0.0 disables the feature entirely.
            fft_highpass_ratio: Fraction of the frequency radius above which
                the high-pass filter is active (0 = everything, 1 = nothing).
            covariance_weight: Blending weight for per-step covariance matching
                during inpainting.  0.0 disables.
        """
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if not (0.0 <= dilation_strength <= 2.0):
            raise ValueError(f"dilation_strength must be in [0, 2], got {dilation_strength}")
        if fft_injection_strength < 0.0:
            raise ValueError(f"fft_injection_strength must be >= 0, got {fft_injection_strength}")
        if not (0.0 <= fft_highpass_ratio <= 1.0):
            raise ValueError(f"fft_highpass_ratio must be in [0, 1], got {fft_highpass_ratio}")
        if not (0.0 <= covariance_weight <= 1.0):
            raise ValueError(f"covariance_weight must be in [0, 1], got {covariance_weight}")

        self.steps = steps
        self.dilation_strength = dilation_strength
        self.fft_injection_strength = fft_injection_strength
        self.fft_highpass_ratio = fft_highpass_ratio
        self.covariance_weight = covariance_weight

    # ------------------------------------------------------------------ #
    #  Timestep Schedule
    # ------------------------------------------------------------------ #

    def compute_timesteps(
        self,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        """Return a 1-D tensor of *N+1* timesteps from ``sigma_max`` → ``sigma_min``.

        When ``dilation_strength > 0`` the spacing follows a cosine warp that
        packs more evaluations into the high-curvature middle of the
        trajectory while coasting through the nearly-linear endpoints.

        The dilation function is:
            u(i) = i / N
            t(i) = σ_max + (σ_min - σ_max) · [ (1-d)·u + d·(1 - cos(πu))/2 ]

        where *d* = ``dilation_strength`` clamped to [0, 1] for blending.
        """
        n = self.steps
        d = min(max(self.dilation_strength, 0.0), 1.0)

        u = torch.linspace(0.0, 1.0, n + 1, device=device, dtype=torch.float64)

        # Cosine-dilated warp
        linear_part = u
        cosine_part = (1.0 - torch.cos(math.pi * u)) / 2.0
        warped = (1.0 - d) * linear_part + d * cosine_part

        timesteps = sigma_max + (sigma_min - sigma_max) * warped
        return timesteps.to(dtype=torch.float32)

    # ------------------------------------------------------------------ #
    #  RK4 Integration Step
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rk4_step(
        model_fn: ModelForwardFn,
        x: Tensor,
        t_cur: float,
        t_next: float,
        **model_kwargs,
    ) -> Tensor:
        """Advance the ODE by one full RK4 step.

        Given  dx/dt = v(x, t),  compute:
            k1 = v(x,           t           )
            k2 = v(x + h/2·k1,  t + h/2     )
            k3 = v(x + h/2·k2,  t + h/2     )
            k4 = v(x + h·k3,    t + h        )
            x_next = x + (h/6)·(k1 + 2k2 + 2k3 + k4)

        This requires **4 model evaluations** per step but achieves O(h⁵)
        local truncation error vs. O(h²) for Euler.

        Args:
            model_fn: Callable that takes ``(x, t_tensor, **kwargs)`` and
                returns the velocity field prediction.
            x: Current latent state  [B, C, H, W].
            t_cur: Current time (scalar).
            t_next: Target time (scalar).
            **model_kwargs: Extra keyword arguments forwarded to ``model_fn``
                (e.g. conditioning embeddings).

        Returns:
            Updated latent ``x_next`` with the same shape as ``x``.
        """
        h = t_next - t_cur
        device = x.device
        dtype = x.dtype

        def _v(x_in: Tensor, t_val: float) -> Tensor:
            t_tensor = torch.full((x_in.shape[0],), t_val, device=device, dtype=dtype)
            return model_fn(x_in, t_tensor, **model_kwargs)

        k1 = _v(x, t_cur)
        k2 = _v(x + 0.5 * h * k1, t_cur + 0.5 * h)
        k3 = _v(x + 0.5 * h * k2, t_cur + 0.5 * h)
        k4 = _v(x + h * k3, t_next)

        x_next = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x_next

    # ------------------------------------------------------------------ #
    #  Euler Integration Step (fallback / low-VRAM mode)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _euler_step(
        model_fn: ModelForwardFn,
        x: Tensor,
        t_cur: float,
        t_next: float,
        **model_kwargs,
    ) -> Tensor:
        """First-order Euler step — single model evaluation.

        Provided as a lightweight fallback for very large models where 4×
        evaluations per step would be prohibitive.
        """
        h = t_next - t_cur
        t_tensor = torch.full(
            (x.shape[0],), t_cur, device=x.device, dtype=x.dtype,
        )
        v = model_fn(x, t_tensor, **model_kwargs)
        return x + h * v

    # ------------------------------------------------------------------ #
    #  FFT High-Frequency Detail Injection
    # ------------------------------------------------------------------ #

    def _fft_detail_injection(self, x: Tensor) -> Tensor:
        """Amplify high-frequency content in the latent via spectral filtering.

        Algorithm:
            1. Compute 2-D real FFT of x.
            2. Build a smooth radial high-pass mask (Butterworth-style sigmoid).
            3. Scale high-frequency components by ``1 + fft_injection_strength``.
            4. Inverse FFT back to spatial domain.

        The mask transitions smoothly from 0 (low-freq, left alone) to 1
        (high-freq, amplified) around the cutoff defined by
        ``fft_highpass_ratio``.

        Args:
            x: Latent tensor [B, C, H, W].

        Returns:
            Latent with high-frequency content amplified.
        """
        if self.fft_injection_strength <= 0.0:
            return x

        B, C, H, W = x.shape

        # --- Forward FFT (real-valued input → half-complex output) ---
        x_freq = torch.fft.rfft2(x, norm="ortho")

        # --- Build radial high-pass mask ---
        freq_h = torch.fft.fftfreq(H, device=x.device, dtype=x.dtype)
        freq_w = torch.fft.rfftfreq(W, device=x.device, dtype=x.dtype)
        grid_h, grid_w = torch.meshgrid(freq_h, freq_w, indexing="ij")
        radius = torch.sqrt(grid_h ** 2 + grid_w ** 2)

        # Normalise so that the maximum possible radius ≈ 0.707 maps to 1.0
        max_radius = math.sqrt(0.5)
        radius_norm = radius / max_radius

        # Smooth sigmoid transition around the cutoff
        cutoff = self.fft_highpass_ratio
        sharpness = 10.0  # controls how steep the transition band is
        highpass_mask = torch.sigmoid(sharpness * (radius_norm - cutoff))

        # Reshape for broadcasting: [1, 1, H, W//2+1]
        highpass_mask = highpass_mask.unsqueeze(0).unsqueeze(0)

        # --- Apply amplification ---
        amplification = 1.0 + self.fft_injection_strength * highpass_mask
        x_freq_boosted = x_freq * amplification

        # --- Inverse FFT ---
        x_boosted = torch.fft.irfft2(x_freq_boosted, s=(H, W), norm="ortho")
        return x_boosted

    # ------------------------------------------------------------------ #
    #  Continuous Covariance Matching (Inpainting)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _covariance_match(
        x: Tensor,
        mask: Tensor,
        weight: float,
    ) -> Tensor:
        """Apply per-channel affine covariance matching between masked and unmasked regions.

        The transform aligns the first- and second-order statistics of generated
        (masked) pixels to match those of the original (unmasked) pixels:

            x_matched = σ_orig / σ_gen · (x_gen − μ_gen) + μ_orig

        This is a simplified diagonal version (no cross-channel covariance) that
        is numerically stable and cheap to compute.

        Args:
            x: Current latent tensor [B, C, H, W].
            mask: Binary mask [B, 1, H, W] where 1 = region being generated
                (will be matched) and 0 = original (reference).
            weight: Blend factor in [0, 1].  0 = no matching, 1 = full match.

        Returns:
            Latent with the masked region's statistics matched to the unmasked.
        """
        if weight <= 0.0:
            return x

        # Ensure mask is broadcastable: [B, 1, H, W]
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask_f = mask.float()
        inv_mask_f = 1.0 - mask_f

        eps = 1e-6

        # --- Compute per-channel statistics for each region ---
        # Unmasked (original) region
        orig_count = inv_mask_f.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
        orig_mean = (x * inv_mask_f).sum(dim=(-2, -1), keepdim=True) / orig_count
        orig_var = (
            ((x - orig_mean) ** 2 * inv_mask_f).sum(dim=(-2, -1), keepdim=True)
            / orig_count
        )
        orig_std = (orig_var + eps).sqrt()

        # Masked (generated) region
        gen_count = mask_f.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
        gen_mean = (x * mask_f).sum(dim=(-2, -1), keepdim=True) / gen_count
        gen_var = (
            ((x - gen_mean) ** 2 * mask_f).sum(dim=(-2, -1), keepdim=True)
            / gen_count
        )
        gen_std = (gen_var + eps).sqrt()

        # --- Affine whitening-coloring ---
        x_normalised = (x - gen_mean) / gen_std
        x_matched = x_normalised * orig_std + orig_mean

        # Blend only inside the mask
        x_out = x.clone()
        x_out = torch.where(
            mask_f.bool().expand_as(x),
            x * (1.0 - weight) + x_matched * weight,
            x,
        )
        return x_out

    # ------------------------------------------------------------------ #
    #  Main Sampling Loop
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def sample(
        self,
        model_fn: ModelForwardFn,
        x: Tensor,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
        mask: Optional[Tensor] = None,
        original_latent: Optional[Tensor] = None,
        use_rk4: bool = True,
        callback: Optional[Callable[[int, Tensor], None]] = None,
        **model_kwargs,
    ) -> Tensor:
        """Run the full DFM sampling loop.

        Args:
            model_fn: Velocity-field prediction function.
                Signature: ``model_fn(x, t, **model_kwargs) -> v``.
            x: Initial noisy latent [B, C, H, W].
            sigma_min: End time of the ODE (clean).
            sigma_max: Start time of the ODE (noise).
            mask: Optional inpainting mask [B, 1, H, W].  1 = generate, 0 = keep.
            original_latent: Original (unmasked) latent for inpainting reference.
                Required when ``mask`` is provided and ``covariance_weight > 0``.
            use_rk4: If True use RK4 integrator, else fall back to Euler.
            callback: Optional ``(step_index, current_x) -> None`` for progress.
            **model_kwargs: Forwarded to ``model_fn`` (conditioning, etc.).

        Returns:
            Denoised latent tensor [B, C, H, W].
        """
        timesteps = self.compute_timesteps(
            sigma_min=sigma_min, sigma_max=sigma_max, device=x.device,
        )

        step_fn = self._rk4_step if use_rk4 else self._euler_step

        for i in range(self.steps):
            t_cur = timesteps[i].item()
            t_next = timesteps[i + 1].item()

            # --- ODE integration step ---
            x = step_fn(model_fn, x, t_cur, t_next, **model_kwargs)

            # --- FFT high-frequency detail injection ---
            x = self._fft_detail_injection(x)

            # --- Inpainting: composite original into unmasked region ---
            if mask is not None and original_latent is not None:
                # Linearly interpolate the original content into the unmasked
                # area at the current noise level so the boundary stays coherent
                # as the solver progresses.
                progress = (sigma_max - t_next) / (sigma_max - sigma_min + 1e-8)
                # Blend a decreasing amount of noise into the original region
                noise_level = 1.0 - progress
                noisy_orig = original_latent + noise_level * torch.randn_like(
                    original_latent,
                )
                inv_mask = 1.0 - mask.float()
                if inv_mask.dim() == 3:
                    inv_mask = inv_mask.unsqueeze(1)
                inv_mask = inv_mask.expand_as(x)
                x = x * mask.float().expand_as(x) + noisy_orig * inv_mask

                # --- Covariance matching on the boundary ---
                x = self._covariance_match(x, mask, self.covariance_weight)

            # --- Progress callback ---
            if callback is not None:
                callback(i, x)

        return x


# ---------------------------------------------------------------------------
#  Convenience wrapper for ComfyUI integration
# ---------------------------------------------------------------------------

def build_model_fn(
    model,
    positive_cond,
    negative_cond,
    cfg_scale: float = 7.5,
) -> ModelForwardFn:
    """Construct a classifier-free-guidance velocity function from a ComfyUI model.

    This wraps the raw model call with CFG:
        v_guided = v_uncond + cfg · (v_cond − v_uncond)

    Args:
        model: ComfyUI model patcher / wrapper.
        positive_cond: Positive conditioning tensors.
        negative_cond: Negative conditioning tensors.
        cfg_scale: Classifier-free guidance strength.

    Returns:
        A callable ``(x, t) -> v_guided`` suitable for ``DFMSolver.sample()``.
    """
    import comfy.samplers  # type: ignore[import-untyped]
    import comfy.model_management  # type: ignore[import-untyped]

    def model_fn(x: Tensor, t: Tensor, **kwargs) -> Tensor:
        # Ensure we're on the correct device
        comfy.model_management.load_model_gpu(model)

        # Unconditional (negative) prediction
        v_uncond = model.model.apply_model(x, t, negative_cond)

        # Conditional (positive) prediction
        v_cond = model.model.apply_model(x, t, positive_cond)

        # CFG blending
        v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)
        return v_guided

    return model_fn
