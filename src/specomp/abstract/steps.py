from abc import ABC, abstractmethod
import inspect
from typing import Type

class Step(ABC):
    subclass_registry = dict[str,Type["Step"]]()
    domain : list[Type]
    range : list[Type]

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def inverse(self, x, side):
        pass

    def __init_subclass__(cls):
        if not inspect.isabstract(cls):
            Step.subclass_registry[cls.__name__] = cls


class LosslessStep(Step):
    pass

class LossyStep(Step):
    pass