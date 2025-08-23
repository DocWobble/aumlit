"""Sock-puppet tensor emitters used by the probing loop.

These helpers generate tiny tensors with the correct geometry for various
model inputs.  They guarantee shapes but not semantics.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np

try:  # optional dependency
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover - fallback
    torch = None
    _HAS_TORCH = False

__all__ = ["text_emb", "latent", "vision_grid", "kv_cache_probe", "audio_mel"]


def _randn(
    shape: Tuple[int, ...],
    dtype: str | None = None,
    seed: int | None = None,
):
    """Generate random data with optional dtype and seed hints."""
    if seed is not None:
        np.random.seed(seed)
        if _HAS_TORCH:
            torch.manual_seed(seed)
    if _HAS_TORCH:
        torch_dtype = torch.float16 if dtype == "f16" else torch.float32
        return torch.randn(shape, dtype=torch_dtype)
    np_dtype = np.float16 if dtype == "f16" else np.float32
    return np.random.randn(*shape).astype(np_dtype)


def text_emb(d: int, seed: int | None = None):
    """Dummy text embedding ``[1, 2, d]``."""
    return _randn((1, 2, int(d)), seed=seed)


def latent(c: int, scale: int, base_hw: int = 64, seed: int | None = None):
    """Dummy latent ``[1, C, H/scale, W/scale]`` for base ``64×64`` images."""
    h = w = base_hw // int(scale)
    return _randn((1, int(c), h, w), seed=seed)


def vision_grid(profile: str, seed: int | None = None):
    """Dummy image tensor for common vision backbones."""
    shapes = {
        "ViT-L/14-grid": (1, 3, 224, 224),
        "ViT-H/14-grid": (1, 3, 224, 224),
        "SigLIP-H-map": (1, 3, 256, 256),
    }
    shape = shapes.get(profile)
    if shape is None:
        raise KeyError(f"unknown vision profile: {profile}")
    return _randn(shape, seed=seed)


def audio_mel(
    frame_shape: Tuple[int, int, int],
    seed: int | None = None,
):
    """Dummy audio mel-spectrogram ``[1, C, T, F]``."""
    c, t, f = (int(x) for x in frame_shape)
    return _randn((1, c, t, f), seed=seed)


def kv_cache_probe(
    n_heads: int,
    d_head: int,
    t: int = 2,
    dtype: str | None = None,
    seed: int | None = None,
):
    """Dummy key/value cache tensor for LLM probes.

    Parameters
    ----------
    n_heads:
        Number of attention heads.
    d_head:
        Dimension per head.
    t:
        Sequence length (default ``2``).
    dtype:
        Optional key/value dtype hint (e.g. ``"f16"`` or ``"f32"``).
    """

    shape = (2, int(n_heads), t, int(d_head))
    return _randn(shape, dtype, seed)
