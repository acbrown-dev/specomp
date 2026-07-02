from dataclasses import dataclass

from specomp.steps.delta_encoding import UnsignedDeltaEncoding
from specomp.steps.generic_steps import ArrToByteStep, ZstdStep
from specomp.abstract.compressors import LosslessCompressor
from specomp.dtypes.compressor_inputs import UINT_ARRAYS_3D


@dataclass
class SimpleDeltaEncoderCompressor(LosslessCompressor):
    zstd_level: int = 3
    accepted_inputs = UINT_ARRAYS_3D

    @property
    def pipeline_steps(self):
        return (UnsignedDeltaEncoding(), ArrToByteStep(), ZstdStep(self.zstd_level))
