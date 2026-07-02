from specomp.abstract.steps import LosslessStep
import numpy as np
from specomp.dtypes.compressor_inputs import UINT_ARRAYS_3D, Bytes
import zstandard

class IdentityStep(LosslessStep):
    domain = UINT_ARRAYS_3D
    range = UINT_ARRAYS_3D

    def forward(self, x):
        return x, None

    def inverse(self, x, _):
        return x

class ArrToByteStep(LosslessStep):
    domain = UINT_ARRAYS_3D
    range = (Bytes,)

    def forward(self, x : np.ndarray):
        return x.tobytes(), (x.dtype, x.shape)

    def inverse(self, x : bytes, dtype_and_shape : tuple) -> np.ndarray:
        dtype, shape = dtype_and_shape
        return np.frombuffer(x,dtype).reshape(shape)

class ZstdStep(LosslessStep):
    domain = (Bytes,)
    range = (Bytes,)
    def __init__(self, level=3) -> None:
        super().__init__()
        self.level = level

    def forward(self, input_bytes):
        return zstandard.compress(input_bytes,level=self.level), None

    def inverse(self, compressed_bytes, _):
        return zstandard.decompress(compressed_bytes)

