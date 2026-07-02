import numpy as np
import pytest

import specomp.compressors  # noqa: F401
from specomp.abstract.compressors import Compressor, LossyCompressor
from specomp.compressors import SimpleDeltaEncoderCompressor


def _round_trip_cases():
    for cls in Compressor.subclass_registry.values():
        for input_type in cls.accepted_inputs:
            yield cls, input_type


@pytest.mark.parametrize(
    "cls,input_type",
    _round_trip_cases(),
    ids=lambda v: v.__name__,
)
def test_round_trip_on_example_data(cls, input_type):
    c = cls()
    arr = input_type.generate_example()
    assert input_type.validate(arr)

    payload, sides = c.compress(arr)
    out = c.decompress(payload, sides)

    assert input_type.validate(out)
    if issubclass(cls, LossyCompressor):
        assert np.max(np.abs(out.astype(float) - arr)) <= c.error_bound
    else:
        np.testing.assert_array_equal(out, arr)


def test_compressor_config_round_trip():
    original = SimpleDeltaEncoderCompressor(zstd_level=7)
    restored = SimpleDeltaEncoderCompressor.from_config(original.config())
    assert restored.zstd_level == 7