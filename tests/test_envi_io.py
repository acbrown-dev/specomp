import numpy as np
import pytest
from spectral.io.envi import read_envi_header, write_envi_header

from specomp.compressors import SimpleDeltaEncoderCompressor
from specomp.io import envi
from specomp.io.envi import (
    SPECOMP_COMPRESSED_KEY,
    SPECOMP_COMPRESSOR_KEY,
    SPECOMP_COMPRESSOR_PARAMS_KEY,
    SPECOMP_SIDES_KEY,
    SpecompEnviError,
    _decode_header_json,
    _decode_header_text,
)
from specomp.io.sides import deserialize_sides


@pytest.fixture
def cube():
    return np.arange(32 * 32 * 16, dtype=np.uint16).reshape(32, 32, 16)


def test_round_trip(tmp_path, cube):
    hdr_path = tmp_path / "scene.hdr"
    envi.save_image(str(hdr_path), cube)
    with envi.open(str(hdr_path)) as img:
        restored = img.load()
    np.testing.assert_array_equal(restored, cube)


def test_header_persistence(tmp_path, cube):
    hdr_path = tmp_path / "scene.hdr"
    compressor = SimpleDeltaEncoderCompressor(zstd_level=5)
    envi.save_image(str(hdr_path), cube, compressor=compressor)

    header = read_envi_header(str(hdr_path))
    assert header[SPECOMP_COMPRESSED_KEY] == "true"
    assert header[SPECOMP_COMPRESSOR_KEY] == "SimpleDeltaEncoderCompressor"
    assert _decode_header_json(header[SPECOMP_COMPRESSOR_PARAMS_KEY]) == {"zstd_level": 5}

    sides = deserialize_sides(_decode_header_text(header[SPECOMP_SIDES_KEY]))
    assert len(sides) == 3
    assert sides[1][0] == np.dtype(np.uint16)
    assert sides[1][1] == cube.shape


def test_metadata_preservation(tmp_path, cube):
    hdr_path = tmp_path / "scene.hdr"
    wavelengths = list(range(cube.shape[2]))
    metadata = {"wavelength": wavelengths, "wavelength units": "nm"}
    envi.save_image(str(hdr_path), cube, metadata=metadata)

    with envi.open(str(hdr_path)) as img:
        assert img.metadata["wavelength"] == [str(w) for w in wavelengths]
        assert img.bands.centers == [float(w) for w in wavelengths]
        assert img.bands.band_unit == "nm"


def test_default_compressor(tmp_path, cube):
    hdr_path = tmp_path / "scene.hdr"
    envi.save_image(str(hdr_path), cube)
    with envi.open(str(hdr_path)) as img:
        restored = img.load()
    np.testing.assert_array_equal(restored, cube)


def test_open_rejects_uncompressed_envi(tmp_path):
    hdr_path = tmp_path / "plain.hdr"
    img_path = tmp_path / "plain.img"
    header = {
        "lines": "2",
        "samples": "2",
        "bands": "1",
        "header offset": "0",
        "file type": "ENVI Standard",
        "data type": "12",
        "interleave": "bip",
        "byte order": "0",
    }
    write_envi_header(str(hdr_path), header)
    img_path.write_bytes(np.zeros(8, dtype=np.uint16).tobytes())

    with pytest.raises(SpecompEnviError, match="not a specomp-compressed"):
        envi.open(str(hdr_path))


def test_rejects_unsupported_dtype(tmp_path):
    hdr_path = tmp_path / "scene.hdr"
    cube = np.ones((4, 4, 4), dtype=np.float32)
    with pytest.raises(TypeError, match="not accepted"):
        envi.save_image(str(hdr_path), cube)
