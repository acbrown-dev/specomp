from specomp.abstract.steps import LosslessStep
import numpy as np

_DTYPE_UPCAST: dict[np.dtype, np.dtype] = {
    np.dtype(np.uint8):  np.dtype(np.int16),
    np.dtype(np.int8):   np.dtype(np.int16),
    np.dtype(np.uint16): np.dtype(np.int32),
    np.dtype(np.int16):  np.dtype(np.int32),
    np.dtype(np.uint32): np.dtype(np.int64),
    np.dtype(np.int32):  np.dtype(np.int64),
    np.dtype(np.float16): np.dtype(np.float32),
    np.dtype(np.float32): np.dtype(np.float32),
    np.dtype(np.float64): np.dtype(np.float64),
}

def _get_dtype_upcast(dtype : np.dtype) -> np.dtype:
    if dtype not in _DTYPE_UPCAST:
        raise TypeError("This dtype is not supported for upcasting in DeltaEncoding")
    return _DTYPE_UPCAST[dtype]

class DeltaEncoding(LosslessStep):
    def __init__(self, spectral_dim = 2):
        self.spectral_dim = spectral_dim

    @property
    def domain(self):
        return np.ndarray

    @property
    def range(self):
        return np.ndarray

    def forward(self, x : np.ndarray) -> tuple[np.ndarray, np.dtype]:
        original_dtype = x.dtype
        x = x.astype(_get_dtype_upcast(original_dtype))
        diff = np.diff(x, axis=self.spectral_dim)
        first_channel = np.expand_dims(
            np.take(x, 0, axis=self.spectral_dim),
            axis=self.spectral_dim,
        )
        return np.concatenate([first_channel,diff], axis=self.spectral_dim), original_dtype
    
    def inverse(self, x, original_dtype : np.dtype) -> np.ndarray:
        undiffed = np.cumulative_sum(x,axis=self.spectral_dim)
        original_x = undiffed.astype(original_dtype)
        return original_x


        
