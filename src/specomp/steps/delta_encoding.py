from specomp.abstract.steps import LosslessStep
import numpy as np
from specomp.dtypes.compressor_inputs import UInt16Cube, UInt32Cube

class UnsignedDeltaEncoding(LosslessStep):
    domain = (UInt16Cube, UInt32Cube)

    def __init__(self, spectral_dim: int = 2):
        self.spectral_dim = spectral_dim


    def forward(self, x: np.ndarray) -> tuple[np.ndarray, None]:
        if x.dtype not in [dom.dtype for dom in self.domain]:
            raise TypeError(f"unsigned delta encoding requires uint16/uint32, got {x.dtype}")

        first = np.expand_dims(
            np.take(x, 0, axis=self.spectral_dim),
            axis=self.spectral_dim,
        )
        wrapped_diff = np.diff(x, axis=self.spectral_dim)
        return np.concatenate([first, wrapped_diff], axis=self.spectral_dim), None

    def inverse(self, x: np.ndarray, _: object) -> np.ndarray:
        # cumsum widens internally; final cast recovers mod 2^n values
        return np.cumsum(x, axis=self.spectral_dim).astype(x.dtype, copy=False)


        
