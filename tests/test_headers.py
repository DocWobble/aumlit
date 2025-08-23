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

