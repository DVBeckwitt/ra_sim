from __future__ import annotations

import struct

import numpy as np
import pytest

from ra_sim.io.osc_reader import ShapeError, read_osc


def _write_fake_osc(path, *, width: int, height: int, pixels: np.ndarray) -> None:
    raw = bytearray(6000 + (width * height * 2))
    raw[:5] = b"RAXIS"
    raw[768:772] = struct.pack(">I", width)
    raw[772:776] = struct.pack(">I", height)
    raw[796:800] = struct.pack(">I", 1)
    raw[6000:] = np.asarray(pixels, dtype=">u2").tobytes()
    path.write_bytes(raw)


def test_read_osc_reads_detector_image(tmp_path) -> None:
    path = tmp_path / "sample.osc"
    expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    _write_fake_osc(path, width=3, height=2, pixels=expected)

    loaded = read_osc(path)

    assert loaded.dtype == np.int32
    assert np.array_equal(loaded, expected)


def test_read_osc_rejects_invalid_signature(tmp_path) -> None:
    path = tmp_path / "invalid.osc"
    path.write_bytes(b"NOTRA" + (b"\x00" * 64))

    with pytest.raises(IOError, match="RAXIS"):
        read_osc(path)


def test_read_osc_raises_shape_error_for_truncated_payload(tmp_path) -> None:
    path = tmp_path / "truncated.osc"
    raw = bytearray(6000 + 2)
    raw[:5] = b"RAXIS"
    raw[768:772] = struct.pack(">I", 2)
    raw[772:776] = struct.pack(">I", 2)
    raw[796:800] = struct.pack(">I", 1)
    path.write_bytes(raw)

    with pytest.raises(ShapeError, match="2x2"):
        read_osc(path)
