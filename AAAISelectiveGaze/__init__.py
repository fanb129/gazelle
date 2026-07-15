"""Selective gaze-following pilot components.

The package intentionally imports no Gazelle or PyTorch modules at import time so
that split construction and metric evaluation remain usable on CPU-only hosts.
"""

__version__ = "0.1.0"

FIXED_DINOV3_VITB_LAYERS = (2, 5, 8, 11)

__all__ = ["FIXED_DINOV3_VITB_LAYERS", "__version__"]
