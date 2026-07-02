from __future__ import annotations

import base64
import builtins
import json
import os
import sys

import numpy as np
import specomp.compressors  # noqa: F401 — populate Compressor.subclass_registry
from spectral.io.envi import (
    EnviDataFileNotFoundError,
    add_band_info_to_metadata,
    add_image_info_to_metadata,
    check_compatibility,
    check_new_filename,
    dtype_to_envi,
    read_envi_header,
    write_envi_header,
)
from spectral.io.envi import _validate_dtype
from spectral.io.spyfile import SpyFile, find_file_path, interleave_transpose

from specomp.abstract.compressors import Compressor
from specomp.compressors import SimpleDeltaEncoderCompressor
from specomp.io.envi_file import SpecompEnviFile
from specomp.io.sides import deserialize_sides, serialize_sides

SPECOMP_VERSION = "0.1.0"
SPECOMP_COMPRESSED_KEY = "specomp compressed"
SPECOMP_COMPRESSOR_KEY = "specomp compressor"
SPECOMP_COMPRESSOR_PARAMS_KEY = "specomp compressor params"
SPECOMP_SIDES_KEY = "specomp sides"
SPECOMP_VERSION_KEY = "specomp version"
DEFAULT_DATA_EXT = ".spec"


class SpecompEnviError(ValueError):
    pass


def _encode_header_json(value) -> str:
    return base64.b64encode(json.dumps(value).encode("utf-8")).decode("ascii")


def _decode_header_json(value) -> object:
    if not isinstance(value, str):
        raise SpecompEnviError(f"Invalid encoded header JSON: {value!r}")
    return json.loads(base64.b64decode(value.encode("ascii")))


def _encode_header_text(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _decode_header_text(value: str) -> str:
    if not isinstance(value, str):
        raise SpecompEnviError(f"Invalid encoded header text: {value!r}")
    return base64.b64decode(value.encode("ascii")).decode("utf-8")


def _validate_cube(cube: np.ndarray, compressor: Compressor) -> None:
    if any(input_type.validate(cube) for input_type in compressor.accepted_inputs):
        return
    accepted = ", ".join(cls.__name__ for cls in compressor.accepted_inputs)
    raise TypeError(
        f"Cube dtype/shape not accepted by {type(compressor).__name__}. "
        f"Accepted input types: {accepted}"
    )


def _prepared_data_and_metadata(
    image,
    *,
    metadata: dict | None,
    interleave: str,
    dtype,
):
    if isinstance(image, np.ndarray):
        data = image
        src_interleave = "bip"
        header = {}
    elif isinstance(image, SpyFile):
        data = image.load(dtype=image.dtype, scale=False)
        src_interleave = {0: "bsq", 1: "bil", 2: "bip"}[image.interleave]
        header = image.metadata.copy()
    elif hasattr(image, "load"):
        data = image.load()
        src_interleave = "bip"
        header = getattr(image, "metadata", {}).copy()
    else:
        raise TypeError("image must be a numpy.ndarray or an object with load()")

    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    if metadata:
        header.update(metadata)

    if hasattr(image, "bands"):
        add_band_info_to_metadata(image.bands, header)

    target_dtype = np.dtype(dtype if dtype is not None else data.dtype)
    _validate_dtype(target_dtype)
    if target_dtype != data.dtype:
        data = data.astype(target_dtype)

    interleave = interleave.lower()
    if interleave not in {"bil", "bip", "bsq"}:
        raise ValueError(f"Invalid interleave: {interleave}")
    if interleave != src_interleave:
        data = data.transpose(interleave_transpose(src_interleave, interleave))

    endian_out = sys.byteorder
    header["interleave"] = interleave
    header["byte order"] = 1 if endian_out == "big" else 0
    header["data type"] = dtype_to_envi[data.dtype.char]
    add_image_info_to_metadata(data, header)
    return data, header


def _resolve_data_path(header_path: str, image: str | None) -> str:
    if image is not None:
        return find_file_path(image)

    base, ext = os.path.splitext(header_path)
    if ext.lower() != ".hdr":
        raise SpecompEnviError('Header file name must end in ".hdr".')

    candidates = [f"{base}{DEFAULT_DATA_EXT}", f"{base}{DEFAULT_DATA_EXT.upper()}"]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    msg = (
        "Unable to determine the specomp data file name for the given header. "
        f"Expected {DEFAULT_DATA_EXT} alongside the header, or pass `image=`."
    )
    raise EnviDataFileNotFoundError(msg)


def save_image(
    hdr_file: str,
    image,
    *,
    compressor: Compressor | None = None,
    ext: str = DEFAULT_DATA_EXT,
    force: bool = False,
    metadata: dict | None = None,
    interleave: str = "bip",
    dtype=None,
    **kwargs,
):
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs)}")

    if compressor is None:
        compressor = SimpleDeltaEncoderCompressor()

    data, header = _prepared_data_and_metadata(
        image,
        metadata=metadata,
        interleave=interleave,
        dtype=dtype,
    )
    _validate_cube(data, compressor)

    payload, sides = compressor.compress(data)
    hdr_path, img_path = check_new_filename(hdr_file, ext, force)

    header["file type"] = "ENVI Standard"
    header[SPECOMP_COMPRESSED_KEY] = "true"
    header[SPECOMP_COMPRESSOR_KEY] = type(compressor).__name__
    header[SPECOMP_COMPRESSOR_PARAMS_KEY] = _encode_header_json(compressor.config())
    header[SPECOMP_SIDES_KEY] = _encode_header_text(serialize_sides(sides))
    header[SPECOMP_VERSION_KEY] = SPECOMP_VERSION

    check_compatibility(header)
    with builtins.open(img_path, "wb") as data_file:
        data_file.write(payload)
    write_envi_header(hdr_path, header, is_library=False)


def open(hdr_file: str, image: str | None = None) -> SpecompEnviFile:
    header_path = find_file_path(hdr_file)
    header = read_envi_header(header_path)

    if header.get(SPECOMP_COMPRESSED_KEY, "").lower() != "true":
        raise SpecompEnviError(
            "File is not a specomp-compressed ENVI image "
            f'(missing "{SPECOMP_COMPRESSED_KEY}" header key).'
        )

    check_compatibility(header)
    data_path = _resolve_data_path(header_path, image)

    compressor_name = header[SPECOMP_COMPRESSOR_KEY]
    compressor_cls = Compressor.subclass_registry.get(compressor_name)
    if compressor_cls is None:
        raise SpecompEnviError(f"Unknown compressor: {compressor_name}")

    compressor = compressor_cls.from_config(
        _decode_header_json(header[SPECOMP_COMPRESSOR_PARAMS_KEY])
    )
    sides = deserialize_sides(_decode_header_text(header[SPECOMP_SIDES_KEY]))
    return SpecompEnviFile(data_path, header, compressor, sides)
