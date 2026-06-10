from specomp.abstract.steps import LosslessStep, LossyStep
import numpy as np
from specomp.dtypes.compressor_inputs import ARRAYS_3D

class IdentityStep(LosslessStep):
    domain = ARRAYS_3D

    def forward(self, x):
        return x, None

    def inverse(self, x, _):
        return x

class ArrToByteStep(LosslessStep):
    domain = ARRAYS_3D

    def forward(self, x : np.ndarray):
        return x.tobytes(), (x.dtype, x.shape)

    def inverse(self, x : bytes, dtype_and_shape : tuple) -> np.ndarray:
        dtype, shape = dtype_and_shape
        return np.frombuffer(x,dtype).reshape(shape)
