from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from spectral import BandInfo
from spectral.io.envi import gen_params

if TYPE_CHECKING:
    from specomp.abstract.compressors import Compressor


class SpecompEnviFile:
    def __init__(
        self,
        data_path: str | Path,
        header: dict,
        compressor: Compressor,
        sides: list,
    ) -> None:
        self.filename = str(data_path)
        self.metadata = header.copy()
        self.compressor = compressor
        self.sides = sides
        self._closed = False

        params = gen_params(header)
        self.nrows = params.nrows
        self.ncols = params.ncols
        self.nbands = params.nbands
        self.dtype = np.dtype(params.dtype)
        self.shape = (self.nrows, self.ncols, self.nbands)

        self.bands = BandInfo()
        if "wavelength" in header:
            self.bands.centers = [float(w) for w in header["wavelength"]]
        if "fwhm" in header:
            self.bands.bandwidths = [float(f) for f in header["fwhm"]]
        self.bands.band_unit = header.get("wavelength units")

    def load(self) -> np.ndarray:
        if self._closed:
            raise ValueError("Cannot load from a closed SpecompEnviFile.")
        with open(self.filename, "rb") as data_file:
            compressed = data_file.read()
        return self.compressor.decompress(compressed, self.sides)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> SpecompEnviFile:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def dtype_from_header(header: dict) -> np.dtype:
        params = gen_params(header)
        return np.dtype(params.dtype)

    @staticmethod
    def envi_dtype_code(dtype: np.dtype) -> str:
        from spectral.io.envi import dtype_to_envi

        return dtype_to_envi[np.dtype(dtype).char]
