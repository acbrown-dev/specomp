from abstract import ABC, abstractmethod
import numpy as np
'''
The idea is to make tests very DRY and the addition of new tests for new compressors mostly automatic.

The tests will use class name and inspect function signatures to determine what specific inputs to test and how to validate decompression.
'''
class Compressor(ABC):

    @abstractmethod
    def compress(self, input_data : np.NDArray):
        pass

    @abstractmethod
    def decompress():
        pass


class LosslessCompressor(Compressor, ABC):
    pass
        
class LossyCompressor(Compressor, ABC):

    @abstractmethod
    def error_bound(self) -> float | None:
        """Guaranted upper bound on reconstruction error, or None for no guarantee"""
