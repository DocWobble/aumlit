from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from puppets import audio_mel


def test_audio_mel_shape():
    t = audio_mel((2, 3, 4))
    assert t.shape == (1, 2, 3, 4)
