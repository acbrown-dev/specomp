from specomp.abstract.steps import LosslessStep, LossyStep
import numpy as np

class IdentityStep(LosslessStep):
    
    @property
    def domain(self):
        return np.ndarray

    @property
    def range(self):
        return np.ndarray

    def forward(self, x):
        return x, None

    def inverse(self, x, _):
        return x

class ArrToByteStep(LosslessStep):
    @property
    def domain(self):
        return np.ndarray

    @property
    def range(self):
        return bytes

    def forward(self, x : np.ndarray):
        return x.tobytes(), (x.dtype, x.shape)

    def inverse(self, x : bytes, dtype_and_shape : tuple) -> np.ndarray:
        dtype, shape = dtype_and_shape
        return np.frombuffer(x,dtype).reshape(shape)