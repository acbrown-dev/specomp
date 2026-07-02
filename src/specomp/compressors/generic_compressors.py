from dataclasses import dataclass

from specomp.abstract.compressors import LosslessCompressor
from specomp.abstract.steps import Step
from specomp.steps import IdentityStep, ArrToByteStep
from specomp.dtypes.compressor_inputs import UInt16Cube, UInt32Cube


@dataclass
class IdentityCompressor(LosslessCompressor):
    accepted_inputs = (UInt16Cube, UInt32Cube)

    @property
    def pipeline_steps(self) -> tuple[Step]:
        return (IdentityStep(), ArrToByteStep())
