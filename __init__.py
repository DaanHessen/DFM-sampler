"""
DFM-Solver — Dilated Flow-Matching Solver for ComfyUI
======================================================

A custom node pack providing a higher-order ODE sampler with spectral detail
injection and covariance-matched inpainting, optimised for MMDiT / flow-matching
models.
"""

from .nodes import DFMSamplerNode, DFMInpaintSamplerNode

NODE_CLASS_MAPPINGS = {
    "DFMSampler": DFMSamplerNode,
    "DFMInpaintSampler": DFMInpaintSamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DFMSampler": "DFM Sampler",
    "DFMInpaintSampler": "DFM Inpaint Sampler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
