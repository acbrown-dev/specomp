import json
from pathlib import Path

import numpy as np
import pytest

import specomp.compressors  # noqa: F401
from specomp.abstract.compressors import Compressor
from specomp.compressors import SimpleDeltaEncoderCompressor
from specomp.io.envi_file import SpecompEnviFile
from specomp.io.sides import deserialize_sides, serialize_sides


def _write_minimal_specomp_files(tmp_path: Path, cube: np.ndarray) -> Path:
    compressor = SimpleDeltaEncoderCompressor(zstd_level=3)
    payload, sides = compressor.compress(cube)
    hdr_path = tmp_path / "scene.hdr"
    spec_path = tmp_path / "scene.spec"
    header = {
        "lines": str(cube.shape[0]),
        "samples": str(cube.shape[1]),
        "bands": str(cube.shape[2]),
        "header offset": "0",
        "file type": "ENVI Standard",
        "data type": SpecompEnviFile.envi_dtype_code(cube.dtype),
        "interleave": "bip",
        "byte order": "0",
        "specomp compressed": "true",
        "specomp compressor": type(compressor).__name__,
        "specomp compressor params": json.dumps(compressor.config()),
        "specomp sides": serialize_sides(sides),
        "specomp version": "0.1.0",
    }
    spec_path.write_bytes(payload)
    with hdr_path.open("w", encoding="utf-8") as hdr_file:
        hdr_file.write("ENVI\n")
        for key, value in header.items():
            hdr_file.write(f"{key} = {value}\n")
    return hdr_path


def test_specomp_envi_file_load_round_trip(tmp_path):
    cube = np.arange(32 * 32 * 16, dtype=np.uint16).reshape(32, 32, 16)
    hdr_path = _write_minimal_specomp_files(tmp_path, cube)
    header = {
        line.split(" = ", 1)[0]: line.split(" = ", 1)[1]
        for line in hdr_path.read_text(encoding="utf-8").splitlines()[1:]
        if " = " in line
    }
    compressor = Compressor.subclass_registry[header["specomp compressor"]].from_config(
        json.loads(header["specomp compressor params"])
    )
    sides = deserialize_sides(header["specomp sides"])
    with SpecompEnviFile(tmp_path / "scene.spec", header, compressor, sides) as img:
        restored = img.load()
        assert img.shape == cube.shape
    assert img._closed
    np.testing.assert_array_equal(restored, cube)
