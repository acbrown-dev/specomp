import numpy as np
import pytest

import specomp.compressors  # noqa: F401
from specomp.abstract.compressors import Compressor, LosslessCompressor, LossyCompressor
import inspect

_ABSTRACT = {Compressor, LosslessCompressor, LossyCompressor}

def _concrete():
    return [
        cls for cls in Compressor.subclass_registry.values()
        if cls not in _ABSTRACT and not inspect.isabstract(cls)
    ]

@pytest.mark.parametrize("cls", _concrete(), ids=lambda c: c.__name__)
def test_round_trip(cls):
    c = cls()
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    payload, sides = c.compress(arr)
    out = c.decompress(payload, sides)
    np.testing.assert_array_equal(out, arr)