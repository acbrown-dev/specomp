import numpy as np
import pytest

from specomp.io.sides import deserialize_sides, serialize_sides


def test_round_trip_none_and_dtype_shape():
    sides = [None, (np.dtype(np.uint16), (32, 32, 16)), None]
    restored = deserialize_sides(serialize_sides(sides))
    assert restored[0] is None
    assert restored[1][0] == np.dtype(np.uint16)
    assert restored[1][1] == (32, 32, 16)
    assert restored[2] is None


def test_serialize_rejects_unknown_side_type():
    with pytest.raises(TypeError, match="Unsupported side type"):
        serialize_sides([{"unexpected": True}])
