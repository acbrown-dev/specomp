from abc import ABC, abstractmethod
from typing import Type

class Step(ABC):
    subclass_registry = []

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

    def __init_subclass__(cls, **kwargs):
        super().__init__subclass(**kwargs)
        Step.registry[cls.__name__] = cls


class LosslessStep(Step):
    pass

class LossyStep(Step):
    pass