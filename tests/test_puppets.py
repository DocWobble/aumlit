from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch optional
    torch = None
    _HAS_TORCH = False

from puppets import audio_mel, text_emb


def _to_numpy(x):
    return x.detach().cpu().numpy() if _HAS_TORCH else x


def test_audio_mel_shape():
    t = audio_mel((2, 3, 4))
    assert t.shape == (1, 2, 3, 4)


def test_seed_reproducible():
    a = _to_numpy(text_emb(4, seed=123))
    b = _to_numpy(text_emb(4, seed=123))
    assert np.array_equal(a, b)


def test_seed_changes_values():
    a = _to_numpy(text_emb(4, seed=1))
    b = _to_numpy(text_emb(4, seed=2))
    assert not np.array_equal(a, b)
