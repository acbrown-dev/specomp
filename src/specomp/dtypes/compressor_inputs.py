from abc import ABC, abstractmethod
import inspect
import numpy as np

# base classes
class InputTypeBase(ABC):
    subclass_registry = dict()

    @classmethod
    @abstractmethod
    def generate_example(cls):
        pass

    @classmethod
    @abstractmethod
    def validate(cls, data) -> bool:
        pass

    def __init_subclass__(cls) -> None:
        if not inspect.isabstract(cls):
            InputTypeBase.subclass_registry[cls.__name__] = cls
        

class DType3DArr(InputTypeBase):
    dtype: np.dtype
    @classmethod
    def generate_example(cls):
        n = 384 * 384 * 288
        return np.arange(n, dtype=cls.dtype).reshape(384, 384, 288)
    @classmethod
    def validate(cls, data: np.ndarray) -> bool:
        return (
            isinstance(data, np.ndarray)
            and data.dtype == cls.dtype
            and data.ndim == 3
        )
# concrete classes
class UInt8Cube(DType3DArr):
    dtype = np.uint8

class UInt16Cube(DType3DArr):
    dtype = np.uint16

class UInt32Cube(DType3DArr):
    dtype = np.uint32

UINT_ARRAYS_3D = (UInt8Cube, UInt16Cube, UInt32Cube )

class Bytes(InputTypeBase):
    @classmethod
    def generate_example(cls):
        n = 100
        return np.arange(0,n,dtype=np.int32).tobytes()
    @classmethod
    def validate(cls, data: bytes):
        return isinstance(data, bytes)

    
