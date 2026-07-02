from abc import ABC, abstractmethod
import dataclasses
import inspect
import numpy as np
from specomp.abstract.steps import Step

class Compressor(ABC):
    accepted_inputs : tuple
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
        steps_and_sides = list(zip(self.pipeline_steps,sides))
        for step, side in reversed(steps_and_sides):
            x = step.inverse(x, side)
        return x

    def config(self) -> dict:
        if not dataclasses.is_dataclass(self):
            return {}
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.init
        }

    @classmethod
    def from_config(cls, params: dict):
        return cls(**params)

    def __init_subclass__(cls):
        if not inspect.isabstract(cls):
            Compressor.subclass_registry[cls.__name__] = cls
            if "accepted_inputs" not in cls.__dict__:
                raise TypeError(f"{cls.__name__} must define accepted_inputs")

class LosslessCompressor(Compressor):
    pass
        
class LossyCompressor(Compressor):

    @property
    @abstractmethod
    def error_bound(self) -> float | None:
        """Guaranteed upper bound on reconstruction error, or None for no guarantee"""
        pass