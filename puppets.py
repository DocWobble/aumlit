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

__all__ = ["text_emb", "latent", "vision_grid", "kv_cache_probe"]


def _randn(shape: Tuple[int, ...]):
    if _HAS_TORCH:
        return torch.randn(shape)
    return np.random.randn(*shape).astype(np.float32)


def text_emb(d: int):
    """Dummy text embedding ``[1, 2, d]``."""
    return _randn((1, 2, int(d)))


def latent(c: int, scale: int, base_hw: int = 64):
    """Dummy latent ``[1, C, H/scale, W/scale]`` for base ``64×64`` images."""
    h = w = base_hw // int(scale)
    return _randn((1, int(c), h, w))


def vision_grid(profile: str):
    """Dummy image tensor for common vision backbones."""
    shapes = {
        "ViT-L/14-grid": (1, 3, 224, 224),
        "ViT-H/14-grid": (1, 3, 224, 224),
        "SigLIP-H-map": (1, 3, 256, 256),
    }
    shape = shapes.get(profile)
    if shape is None:
        raise KeyError(f"unknown vision profile: {profile}")
    return _randn(shape)


def kv_cache_probe(n_heads: int, d_head: int, t: int = 2):
    """Dummy key/value cache tensor for LLM probes."""
    shape = (2, int(n_heads), t, int(d_head))
    return _randn(shape)
