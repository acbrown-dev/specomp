from abc import ABC, abstractmethod
from typing import Type

class Step(ABC):
    subclass_registry = dict()

    @property
    @abstractmethod
    def domain(self) -> Type:
        pass

    @property
    @abstractmethod
    def range(self) -> Type:
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def inverse(self, x, side):
        pass

    def __init_subclass__(cls):
        Step.subclass_registry[cls.__name__] = cls


class LosslessStep(Step):
    pass

class LossyStep(Step):
    pass