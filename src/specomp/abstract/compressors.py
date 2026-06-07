from abc import ABC, abstractmethod
import numpy as np
from specomp.abstract.steps import Step
import inspect
'''
The idea is to make tests very DRY and the addition of new tests for new compressors mostly automatic.

The tests will use class name and inspect function signatures to determine what specific inputs to test and how to validate decompression.
'''
class Compressor(ABC):
    subclass_registry = dict[str,type["Compressor"]]()

    @property
    @abstractmethod
    def pipeline_steps(self) -> tuple[Step]:
        pass
    
    def compress(self, input_data : np.ndarray):
        x = input_data
        sides = []
        for step in self.pipeline_steps:
            x, side = step.forward(x)
            sides.append(side)
        return x, sides
        
    def decompress(self, compressed_data : bytes, sides):
        x = compressed_data
        for step, side in reversed(zip(self.pipeline_steps, sides)):
            x = step.inverse(x, side)
        return x

    def __init_subclass__(cls, **kwargs):
        super().__init__subclass(**kwargs)
        Compressor.registry[cls.__name__] = cls

class LosslessCompressor(Compressor):
    pass
        
class LossyCompressor(Compressor):

    @property
    @abstractmethod
    def error_bound(self) -> float | None:
        """Guaranteed upper bound on reconstruction error, or None for no guarantee"""
        pass