import base64
import io
import tarfile
from pathlib import Path


FIXTURES_DIR = Path(__file__).parent / "fixtures"
ARCHIVE_B64 = FIXTURES_DIR / "fixtures.tar.gz.base64"


def pytest_sessionstart(session):
    """Unpack archived binary fixtures for tests."""
    onnx_file = FIXTURES_DIR / "linear4.onnx"
    gguf_file = FIXTURES_DIR / "tiny.gguf"
    if onnx_file.exists() and gguf_file.exists():
        return
    data = base64.b64decode(ARCHIVE_B64.read_text())
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(FIXTURES_DIR)

