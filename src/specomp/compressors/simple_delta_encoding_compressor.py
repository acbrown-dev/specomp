from specomp.steps.delta_encoding import UnsignedDeltaEncoding
from specomp.steps.generic_steps import ArrToByteStep, ZstdStep
from specomp.abstract.compressors import LosslessCompressor
from specomp.dtypes.compressor_inputs import UINT_ARRAYS_3D

class SimpleDeltaEncoderCompressor(LosslessCompressor):

    accepted_inputs = UINT_ARRAYS_3D

    def __init__(self, zstd_level = 3) -> None:
        super().__init__()
        self._zstd_level = zstd_level

    @property
    def pipeline_steps(self):
        return (UnsignedDeltaEncoding(),ArrToByteStep(),ZstdStep(self._zstd_level))