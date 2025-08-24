from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import headers
import pytest


def test_read_safetensors_header_malformed(tmp_path):
    bad_json = b'{"foo": 1'
    path = tmp_path / "bad.safetensors"
    path.write_bytes(len(bad_json).to_bytes(8, "little") + bad_json)
    with pytest.raises(RuntimeError, match="Malformed safetensors header"):
        headers.read_safetensors_header(path)


def test_class_key_deterministic():
    meta1 = headers.Meta(
        tensors=[headers.TensorInfo("a", (1, 2)), headers.TensorInfo("b", (3,))],
        hints={"x": 1},
    )
    meta2 = headers.Meta(
        tensors=[headers.TensorInfo("b", (3,)), headers.TensorInfo("a", (1, 2))],
        hints={"x": 1},
    )
    meta3 = headers.Meta(
        tensors=[headers.TensorInfo("a", (1, 2)), headers.TensorInfo("b", (4,))],
        hints={"x": 1},
    )
    key1 = headers.class_key(meta1)
    key2 = headers.class_key(meta2)
    key3 = headers.class_key(meta3)
    assert key1 == key2
    assert key1 != key3

