# DFM-Solver — Dilated Flow-Matching Solver for ComfyUI

A production-ready ComfyUI custom node pack that introduces **a fundamentally new sampling architecture** optimised for Multimodal Diffusion Transformers (MMDiTs) and Flow-Matching models such as **Flux**, **SD3**, **Qwen-image-edit**, and similar architectures.

> **TL;DR** — Replace your KSampler with the DFM Sampler node for sharper details, more accurate trajectory tracking, and seamlessly blended inpainting.

---

## ✨ Features

| Feature | Benefit |
|---|---|
| **4th-Order Runge-Kutta ODE Integrator** | Tracks the curved flow trajectory with O(h⁵) local error instead of Euler's O(h²) — dramatically better quality at low step counts |
| **Cosine-Dilated Step Schedule** | Concentrates model evaluations in the high-curvature mid-region of the trajectory where detail is decided |
| **FFT High-Frequency Detail Injection** | Amplifies micro-detail texture via spectral high-pass filtering to prevent the "smooth/melted AI look" |
| **Continuous Covariance Matching** | Forces generated inpaint regions to adopt the exact color & lighting distribution of the surrounding original image |

---

## 🔬 The Math

### 1. Flow-Matching ODE & RK4 Integration

Flow-matching models learn a velocity field **v(x, t)** that defines an ODE:

```
dx/dt = v(x, t),    t ∈ [1, 0]   (noise → clean)
```

Instead of the standard Euler step `x ← x + h·v`, we use classical RK4:

```
k₁ = v(x,            t         )
k₂ = v(x + h/2 · k₁, t + h/2   )
k₃ = v(x + h/2 · k₂, t + h/2   )
k₄ = v(x + h · k₃,   t + h     )

x_next = x + (h/6) · (k₁ + 2k₂ + 2k₃ + k₄)
```

This requires 4 model evaluations per step but achieves **5th-order local accuracy**, meaning 20 RK4 steps ≈ 80 Euler steps in trajectory fidelity.

### 2. Cosine-Dilated Timestep Schedule

Rather than uniform spacing, we warp the timesteps with a cosine dilation:

```
u(i) = i / N
t(i) = σ_max + (σ_min − σ_max) · [(1−d)·u + d·(1 − cos(πu))/2]
```

where **d** = `dilation_strength`. This packs more evaluations into the mid-trajectory region where the velocity field has highest curvature, while coasting through the nearly-linear start and end.

### 3. FFT High-Frequency Detail Injection

At each step:
1. Compute `X = FFT2(x)`
2. Build a smooth Butterworth-style radial high-pass mask `H(r)`
3. Apply: `X' = X · (1 + α · H(r))`  where α = `fft_injection_strength`
4. `x' = IFFT2(X')`

This selectively amplifies the high-frequency micro-detail that the model predicts but that first-order solvers tend to smooth away.

### 4. Continuous Covariance Matching (Inpainting)

When a mask is present, at each solver step we match the per-channel statistics of the generated region to the original:

```
x_matched = (σ_orig / σ_gen) · (x_gen − μ_gen) + μ_orig
```

Blended with weight **w** = `covariance_weight`:
```
x_out = (1 − w) · x_gen + w · x_matched    (inside mask only)
```

This forces color temperature, brightness, and contrast coherence dynamically as content is being generated.

---

## 📦 Installation

### Option A: Git Clone (recommended)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/DFM-Solver.git
```

### Option B: Manual Copy

Copy the entire `DFM-Solver` folder into `ComfyUI/custom_nodes/`.

Then restart ComfyUI. The nodes will appear under **sampling/dfm** in the node menu.

---

## 🔌 Nodes

### DFM Sampler

General-purpose sampler — drop-in replacement for KSampler with flow-matching models.

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | Your loaded flow-matching model |
| `positive` | CONDITIONING | Positive prompt conditioning |
| `negative` | CONDITIONING | Negative prompt conditioning |
| `latent_image` | LATENT | Empty latent or VAE-encoded image |
| `seed` | INT | Random seed |
| `steps` | INT | Solver steps (default: 20) |
| `cfg` | FLOAT | CFG scale (default: 7.5) |
| `dilation_strength` | FLOAT | Timestep dilation (0–1, default: 0.7) |
| `fft_injection_strength` | FLOAT | Detail sharpening (0–1, default: 0.15) |
| `fft_highpass_ratio` | FLOAT | High-pass cutoff (0.05–0.95, default: 0.35) |
| `use_rk4` | BOOLEAN | RK4 or Euler fallback (default: True) |

**Output:** `LATENT`

### DFM Inpaint Sampler

Adds mask-aware covariance matching for inpainting.

| Additional Input | Type | Description |
|---|---|---|
| `mask` | MASK | Inpainting mask (white = generate) |
| `covariance_weight` | FLOAT | Color/lighting matching strength (0–1, default: 0.6) |

**Output:** `LATENT`

---

## 🎛️ Recommended Settings

### Text-to-Image
| Parameter | Value |
|---|---|
| steps | 20–30 |
| cfg | 5.0–7.5 |
| dilation_strength | 0.6–0.8 |
| fft_injection_strength | 0.10–0.20 |
| use_rk4 | ✅ |

### Inpainting
| Parameter | Value |
|---|---|
| steps | 25–35 |
| cfg | 5.0–7.0 |
| dilation_strength | 0.7 |
| fft_injection_strength | 0.10–0.15 |
| covariance_weight | 0.5–0.7 |
| use_rk4 | ✅ |

### Low-VRAM / Speed Priority
Disable RK4 (`use_rk4 = false`) to fall back to Euler (1 model eval/step instead of 4). Increase `steps` to compensate.

---

## 🧪 Example Workflow (ComfyUI)

```
[Load Checkpoint] → model ─────────────────┐
                                            ▼
[CLIP Text Encode] → positive ────────► [DFM Sampler] → latent ─► [VAE Decode] → [Save Image]
[CLIP Text Encode] → negative ────────┘         ▲
[Empty Latent Image] → latent_image ────────────┘
```

For inpainting, wire a `[Load Image]` + `[Create Mask]` into the DFM Inpaint Sampler's `mask` port, and use the VAE-encoded original as `latent_image`.

---

## 📂 Repository Structure

```
DFM-Solver/
├── __init__.py          # Node registration
├── dfm_sampler.py       # Core solver (RK4, FFT, covariance matching)
├── nodes.py             # ComfyUI node class definitions
├── requirements.txt     # Dependencies (torch only)
└── README.md            # This file
```

---

## 📄 License

MIT License — free to use, modify, and distribute.
