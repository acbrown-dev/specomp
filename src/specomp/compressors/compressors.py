from specomp.abstract.compressors import LosslessCompressor, LossyCompressor
from specomp.abstract.steps import Step
from specomp.steps import IdentityStep, ArrToByteStep
import numpy as np

class IdentityCompressor(LosslessCompressor):

    @property
    def pipeline_steps(self) -> tuple[Step]:
        return (IdentityStep(), ArrToByteStep())

