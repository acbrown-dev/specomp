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
        x.tobytes(), x.dtype

    def inverse(self, x : bytes, dtype : np.dtype) -> np.ndarray:
        return np.from_buffer(x,dtype)